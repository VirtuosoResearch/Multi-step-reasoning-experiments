import argparse
import logging
import os
import wandb

from src.custom.clrs_text_task_data_module import TextCLRSDataModule
from src.custom.clrs_text_task_graph_data_module import TextGraphCLRSDataModule

from src.custom.multitask_model import MultitaskModel
from src.model.GraphLlama import GraphLlamaForCausalLM
from src.model.gnn_models.config import load_cfg

from functools import partial
from pytorch_lightning.trainer.states import RunningStage, TrainerFn

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pytorch_lightning as pl
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.nn import Embedding

from peft import get_peft_model, LoraConfig
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
from collections import defaultdict
import time
import re

from adapters import SeqBnInvConfig, PrefixTuningConfig, BnConfig, DoubleSeqBnConfig, SeqBnConfig
from adapters import AutoAdapterModel,list_adapters, BnConfig
from torch._inductor.async_compile import AsyncCompile

logging.basicConfig(level=logging.INFO, force=True)
torch.set_float32_matmul_precision("high")

def add_result_to_csv(result_datapoint, file_name):
    for key, val in result_datapoint.items():
        result_datapoint[key] = [val, ]
    
    if os.path.exists(file_name):
        result_df = pd.read_csv(file_name, index_col=0)
        tmp_df = pd.DataFrame(result_datapoint)
        result_df = pd.concat([result_df, tmp_df], ignore_index = True)
        result_df.to_csv(file_name)
    else:
        result_df = pd.DataFrame(result_datapoint)  
        result_df.to_csv(file_name)   

def layer_index_from_name(name, default=0):
    _LAYER_PATTERNS = [
        re.compile(r"(?:^|\.)(layers)\.(\d+)(?:\.|$)"),  # LLaMA-style: ...layers.{idx}....
        re.compile(r"(?:^|\.)(h)\.(\d+)(?:\.|$)"),       # GPT-2/NeoX-style: ...h.{idx}....
        re.compile(r"(?:^|\.)(layer)\.(\d+)(?:\.|$)"),   # BERT-style: ...layer.{idx}....
    ]
    for pat in _LAYER_PATTERNS:
        m = pat.search(name)
        if m:
            return int(m.group(2))
    return default


def initialize_model(args):
    model_key = args.model_key.replace("/", "-").replace("..", "")
    if "gpt" in args.model_key or "Llama" in model_key \
        or "bloomz" in model_key or "gemma" in model_key or "Mistral" in model_key:
        hf_key = args.model_key.replace("_", "-")
        tokenizer = AutoTokenizer.from_pretrained(hf_key)
        tokenizer.padding_side = 'right'
        if args.use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
                )
            model = AutoModelForCausalLM.from_pretrained(hf_key, quantization_config=quantization_config, torch_dtype=torch.bfloat16, device_map={"": args.devices[0]}) 
        else:
            model = AutoModelForCausalLM.from_pretrained(hf_key, torch_dtype=torch.bfloat16)
        model_type = "decoder"
        append_eos = True
    elif "flan" in model_key:
        hf_key = "google/{}".format(model_key.replace("_", "-"))
        model = AutoModelForSeq2SeqLM.from_pretrained(hf_key)
        tokenizer = AutoTokenizer.from_pretrained(hf_key, model_max_length=512)
        model_type = "encoder_decoder"
        append_eos = False  # t5 tokenizers already append eos
    elif "Qwen" in model_key:
        hf_key = args.model_key.replace("_", "-")
        if args.use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
                )
            model = AutoModelForCausalLM.from_pretrained(hf_key, quantization_config=quantization_config, torch_dtype=torch.bfloat16, device_map={"": args.devices[0]}) 
        else:
            model = AutoModelForCausalLM.from_pretrained(hf_key)
        tokenizer = AutoTokenizer.from_pretrained(hf_key, model_max_length=args.max_length+ args.max_output_length)
        model_type = "decoder"
        append_eos = True
    else:
        raise NotImplementedError(args.model_key)
    
    if args.use_graph_llama:
        cfg = load_cfg("./src/model/gnn_models/configs/SAGE.yml")
        specs = np.load(f".//src/model/specs/{args.task_names[0]}_specs.npy", allow_pickle=True).item()

        model.get_model().initialize_graph_modules(
            graph_tower="SAGE",
            specs=specs, cfg=cfg, use_cross_attn=args.use_cross_attn, 
            add_output_projection=args.add_output_projection,
            alignment_loss_weight=args.alignment_loss_weight,
            test_classifier_before_cross_attn=args.test_classifier_before_cross_attn
        )
        model.requires_grad_(False)
        # only the graph tower is trainable
        if not args.freeze_graph_tower:
            for p in model.get_model().get_graph_tower().parameters():
                p.requires_grad = True
        for p in model.get_model().graph_projector.parameters():
            p.requires_grad = True
        if args.use_cross_attn:
            for p in model.get_model().graph_token_cross_attn.parameters():
                p.requires_grad = True

        model.set_if_only_train_graph(args.only_train_graph)

    if args.train_adapter:
        
        if args.use_qadapter:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4' 
            )

            model = AutoAdapterModel.from_pretrained(
                hf_key, 
                quantization_config=quantization_config, 
                torch_dtype=torch.bfloat16, 
                device_map={"": args.devices[0]}
            )
        
        else: model = AutoAdapterModel.from_pretrained(hf_key)

        bottleneck_config = DoubleSeqBnConfig(
            mh_adapter=True,    
            output_adapter=True,    
            reduction_factor=args.reduction_factor,     
            non_linearity="relu"     
        )

        model.add_adapter(adapter_name="seq_bn",config=bottleneck_config)

        for name, param in model.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False

        model.set_active_adapters("seq_bn")
        trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params_count = sum(p.numel() for p in model.parameters())

        print(f"Trainable parameters: {trainable_params_count} || All parameters: {all_params_count} || ratio: {trainable_params_count/all_params_count}")
        print("-"*20,"Bottleneck_Adapter","-"*20)
    
    elif args.use_3bit or args.use_2bit:
        ''' deprecated '''
        from src.lqlora_utils import lora_utils
        model = lora_utils.prepare_model_for_lora(
            model=model,
            num_ranks=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            use_gradient_checkpointing=True)

        lora_utils.transform_lora_layers(
            lpq=False,
            model=model,
            model_name="nf3" if args.use_3bit else "nf2",
            device=f"cuda:{args.devices[0]}")
        model.to(f"cuda:{args.devices[0]}")        

    elif args.train_lora:
        if args.model_key == "gpt2": # for gpt2, we generally use full model
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["c_attn", "c_proj", "c_fc"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        elif args.model_key == "EleutherAI/gpt-neox-20b":
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["query_key_value"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        elif "flan" in args.model_key:
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["q", "k", "v"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        else:
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        model = get_peft_model(model, config)
        if args.use_graph_llama:
            # only the graph tower is trainable
            if not args.freeze_graph_tower:
                for p in model.model.get_model().get_graph_tower().parameters():
                    p.requires_grad = True
            for p in model.model.get_model().graph_projector.parameters():
                p.requires_grad = True
            if args.use_cross_attn:
                for p in model.model.get_model().graph_token_cross_attn.parameters():
                    p.requires_grad = True

        model.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, hf_key, model_type, append_eos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_names", type=str, nargs="+", default=['dfs']) 
    
    parser.add_argument("--model_key", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--inference_batch_size", type=int, default=None)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--disable_checkpointing", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_output_length", type=int, default=64)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--optimizer", type=str, default="adamw")

    parser.add_argument("--eval_split", type=float, default=0.2)
    parser.add_argument("--downsample_ratio", type=float, default=1.0)
    parser.add_argument("--minimum_samples", type=int, default=1e6)
    parser.add_argument("--minimum_samples_validation", type=int, default=1e6)
    parser.add_argument("--train_lengths", type=int, nargs="+", default=[4])
    parser.add_argument("--test_lengths", type=int, nargs="+", default=[4])
    parser.add_argument("--few_shot_k", type=int, default=0)
    parser.add_argument("--only_evaluate_test_set", action="store_true")
    parser.add_argument("--only_load_last_output", action="store_true")  # for graph-llama, only load the last output
    parser.add_argument("--only_answer_output", action="store_true") # only load the last step
    parser.add_argument("--eval_last_step", action="store_true") # only evaluate the last step of the output

    parser.add_argument("--use_graph_llama", action="store_true")
    parser.add_argument("--only_train_graph", action="store_true") # pretraining gnn 
    parser.add_argument("--test_classifier_before_cross_attn", action="store_true") # pretraining gnn
    parser.add_argument("--freeze_graph_tower", action="store_true")
    parser.add_argument("--use_cross_attn", action="store_true")
    parser.add_argument("--add_output_projection", action="store_true")
    parser.add_argument("--alignment_loss_weight", type=float, default=0.0)

    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)

    parser.add_argument("--train_adapter", action="store_true")
    parser.add_argument("--reduction_factor", type=int, default=128)
    parser.add_argument("--use_qadapter", action="store_true")

    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--use_3bit", action="store_true")
    parser.add_argument("--use_2bit", action="store_true")
    
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--runs", type=int, default=3)

    parser.add_argument("--load_model_dir", type=str, default="test")
    parser.add_argument("--load_branching_config", type=str)
    parser.add_argument("--task_branching_config_dir", type=str)
    parser.add_argument("--write_results", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--generate_output", action="store_true")

    args = parser.parse_args()
    args.enable_checkpointing = not args.disable_checkpointing
    print("arguments".upper().center(80, "-"))
    print(args)
    print("-" * 80)

    model_key = args.model_key.replace("/", "-").replace("..", "")
    save_name = model_key + \
                (f"_{args.save_name}" if args.save_name else "") + \
                (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                (f"_use_only_answer_output" if args.only_answer_output else "")
    file_dir = os.path.join("./results/", save_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    metrics = {}
    for run in range(args.runs):
        model, tokenizer, hf_key, model_type, append_eos = initialize_model(args)

        batch_size = args.batch_size
        if args.inference_batch_size is None:
            inference_batch_size = batch_size
        else:
            inference_batch_size = args.inference_batch_size
        
        if args.use_graph_llama:
            data_module = TextGraphCLRSDataModule(
                task_names=args.task_names,
                tokenizer=tokenizer,
                batch_size=batch_size,
                inference_batch_size=inference_batch_size,
                max_input_length=args.max_length,
                max_output_length=args.max_output_length,
                eval_all=True,
                eval_split=args.eval_split,
                downsample_ratio=args.downsample_ratio,
                minimum_samples=args.minimum_samples,
                minimum_samples_validation=args.minimum_samples_validation,
                train_lengths=args.train_lengths,
                test_lengths=args.test_lengths,
                use_few_shot=(args.few_shot_k > 0), 
                few_shot_k=args.few_shot_k,
                load_only_last_output=args.only_load_last_output)
        else:
            data_module = TextCLRSDataModule(
                    task_names=args.task_names,
                    tokenizer=tokenizer,
                    batch_size=batch_size,
                    inference_batch_size=inference_batch_size,
                    max_input_length=args.max_length,
                    max_output_length=args.max_output_length,
                    eval_all=True,
                    eval_split=args.eval_split,
                    downsample_ratio=args.downsample_ratio,
                    minimum_samples=args.minimum_samples,
                    minimum_samples_validation=args.minimum_samples_validation,
                    train_lengths=args.train_lengths,
                    test_lengths=args.test_lengths,
                    use_few_shot=(args.few_shot_k > 0), 
                    few_shot_k=args.few_shot_k,
                    only_answer_output=args.only_answer_output)
        data_module.setup(stage="fit")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

        extended_task_names = [f"{task_name}" for task_name in args.task_names]
        lm = MultitaskModel(model, tokenizer, model_type, use_cpu_offload=False,
                        lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, max_output_length=args.max_output_length, use_wandb=args.use_wandb, 
                        optimizer=args.optimizer, generate_output=args.generate_output, task_names=extended_task_names, eval_clrs=args.eval_last_step)
        
        load_model_dir = args.load_model_dir
        load_model_dir = os.path.join("external_lightning_logs", load_model_dir)
        if load_model_dir is not None:
            if ("ckpt" in load_model_dir) and os.path.exists(load_model_dir):
                lm = MultitaskModel.load_from_checkpoint(load_model_dir, model=model, tokenizer=tokenizer, model_type=model_type,
                        lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, max_output_length=args.max_output_length, use_wandb=args.use_wandb,
                        optimizer=args.optimizer, generate_output=args.generate_output, task_names=extended_task_names, eval_clrs=args.eval_last_step)
                print(f"Loaded model from {load_model_dir}")
            elif ("pt" in load_model_dir) and os.path.exists(load_model_dir):
                if args.use_graph_llama:
                    print(model.model.load_state_dict(torch.load(load_model_dir), strict=False))
                else:
                    print(model.load_state_dict(torch.load(load_model_dir), strict=False))
                print(f"Loaded model from {load_model_dir}")

        if args.load_branching_config:
            # load weights from different trained adapters
            if args.task_branching_config_dir is not None:
                task_branching_config_dir = os.path.join("branching_configs", args.task_branching_config_dir)
                with open(task_branching_config_dir, "r") as f:
                    for line in f.readlines():
                        layers, checkpoint_dir = line.split(":")
                        layers = layers.split(",")
                        checkpoint = torch.load(load_model_dir)
                        new_checkpoint = {}
                        for key, val in checkpoint.items():
                            if str(layer_index_from_name(key)) in layers:
                                new_checkpoint[key] = val
                        model.load_state_dict(new_checkpoint, strict=False)

        if not os.path.exists("external_lightning_logs"):
            raise Exception("external_lightning_logs/ does not exist")
        default_root_dir = os.path.join("external_lightning_logs", 
                                        f"{model_key}_" + \
                                        ("_".join(extended_task_names) if len("_".join(extended_task_names)) <= 100 else "{}_tasks".format(len(extended_task_names))) + \
                                        (f"_use_only_answer_output" if args.only_answer_output else "") + \
                                        (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                                        (f"_{args.save_name}" if args.save_name else "") + \
                                        f"_run_{run}"
                                        )
        # # remove previous checkpoints
        # if args.save_name and os.path.exists(default_root_dir):
        #     os.system(f"rm -rf {default_root_dir}")
        
        checkpoint_callback = ModelCheckpoint(
            monitor="accuracy",
            dirpath=default_root_dir,
            filename="epoch_{epoch}",
            save_top_k=(-1 if args.save_every_epoch else 1),
            mode="max",
        )

        trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                            default_root_dir=default_root_dir, min_epochs=args.epochs, max_epochs=args.epochs,
                            accumulate_grad_batches=args.accumulate, precision=args.precision,
                            enable_checkpointing=args.enable_checkpointing,
                            callbacks=[checkpoint_callback]
                            )
        # save initial weights
        if args.train_lora:
            if not os.path.exists(default_root_dir):
                os.makedirs(default_root_dir)
            model_path = default_root_dir + "/initial_weights.pt"
            state_dict = model.state_dict()
            state_dict = {k: v.clone() for k, v in state_dict.items() if "lora" in k}
            torch.save(state_dict, model_path)

        start_time = time.time()
        if args.epochs > 0:
            trainer.fit(lm, datamodule=data_module)
            if args.use_graph_llama:
                from lightning_fabric.utilities.cloud_io import _load as pl_load
                checkpoint = pl_load(checkpoint_callback.best_model_path, map_location=lm.device)
                state_dict = checkpoint["state_dict"]
                state_dict = {k[6:]: v for k, v in state_dict.items() if ("graph" in k or "lora" in k)}
                torch.save(state_dict, checkpoint_callback.best_model_path.replace(".ckpt", ".pt"))
            elif args.train_lora:
                from lightning_fabric.utilities.cloud_io import _load as pl_load
                checkpoint = pl_load(checkpoint_callback.best_model_path, map_location=lm.device)
                state_dict = checkpoint["state_dict"]
                state_dict = {k[6:]: v for k, v in state_dict.items() if "lora" in k}
                torch.save(state_dict, checkpoint_callback.best_model_path.replace(".ckpt", ".pt"))
            elif args.train_adapter:
                from lightning_fabric.utilities.cloud_io import _load as pl_load
                checkpoint = pl_load(checkpoint_callback.best_model_path, map_location=lm.device)
                state_dict = checkpoint["state_dict"]
                state_dict = {k[6:]: v for k, v in state_dict.items() if ("adapter" in k or "head" in k)}
                torch.save(state_dict, checkpoint_callback.best_model_path.replace(".ckpt", ".pt"))
        end_time = time.time()
        print(f"Training time: {end_time - start_time}")

        # evaluate the best checkpoint
        start_time = time.time()
            
        if args.epochs > 0:
            if args.use_graph_llama or args.train_lora or args.train_adapter or \
                args.use_qadapter or args.use_qlora or \
                args.use_3bit or args.use_2bit:                         
                model, tokenizer, hf_key, model_type, append_eos = initialize_model(args)
                model.load_state_dict(state_dict, strict=False)
                lm = MultitaskModel(model, tokenizer, model_type, use_cpu_offload=False,
                        lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, max_output_length=args.max_output_length, use_wandb=args.use_wandb,
                        optimizer=args.optimizer, generate_output=args.generate_output, task_names=extended_task_names, eval_clrs=args.eval_last_step)
                
                if args.use_3bit or args.use_2bit:
                    trainer.validate_loop.trainer_fn = TrainerFn.FITTING
                    trainer.validate_loop.inference_mode = False
                
                summary = trainer.validate(lm, dataloaders=data_module.test_dataloader())[0]
                summary = {f"test_{key}": val for key, val in summary.items()}
                summary.update(trainer.validate(lm, dataloaders=data_module.val_dataloader())[0])
            else:
                summary = trainer.validate(lm, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)[0]
            logging.info(summary)
        else:
            if args.use_3bit or args.use_2bit:
                trainer.validate_loop.trainer_fn = TrainerFn.FITTING
                trainer.validate_loop.inference_mode = False

            summary = {}
            if args.only_evaluate_test_set:
                summary = trainer.validate(lm, dataloaders=data_module.test_dataloader())[0]
                summary = {f"test_{key}": val for key, val in summary.items()}
            else:
                summary = trainer.validate(lm, dataloaders=data_module.test_dataloader())[0]
                summary = {f"test_{key}": val for key, val in summary.items()}
                summary.update(trainer.validate(lm, dataloaders=data_module.val_dataloader())[0])
            logging.info(summary)
        end_time = time.time()
        print(f"Evaluation time: {end_time - start_time}")
            
        for key in summary:
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(summary[key])

        # delete the whole model checkpoint and only keep the lora parameters
        if args.use_graph_llama or args.train_lora or args.train_adapter:
            os.system(f"rm {checkpoint_callback.best_model_path}")
    
    for key in metrics:
        logging.info("{}: {:.4f} +/- {:.4f}".format(key, np.mean(metrics[key]), np.std(metrics[key])))
    
    # save indexes 
    if args.write_results:
        for task_name in extended_task_names:
            result_datapoint = {
                "Task name": task_name,
                "Trained with": " ".join(extended_task_names),
            }
            for key, val in metrics.items():
                if task_name in key:
                    tmp_key = key.replace(f"{task_name}_", "")
                    result_datapoint[tmp_key] = np.mean(val)
            file_name = os.path.join(file_dir, "results.csv")
            add_result_to_csv(result_datapoint, file_name)