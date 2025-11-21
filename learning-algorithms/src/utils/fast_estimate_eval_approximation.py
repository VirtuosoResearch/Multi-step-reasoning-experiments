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
from sklearn.linear_model import LogisticRegression

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
    
    # if args.use_graph_llama:
    #     cfg = load_cfg("./src/model/gnn_models/configs/SAGE.yml")
    #     specs = np.load(f".//src/model/specs/{args.task_names[0]}_specs.npy", allow_pickle=True).item()

    #     model.get_model().initialize_graph_modules(
    #         graph_tower="SAGE",
    #         specs=specs, cfg=cfg, use_cross_attn=args.use_cross_attn, 
    #         add_output_projection=args.add_output_projection,
    #         alignment_loss_weight=args.alignment_loss_weight,
    #         test_classifier_before_cross_attn=args.test_classifier_before_cross_attn
    #     )
    #     model.requires_grad_(False)
    #     # only the graph tower is trainable
    #     if not args.freeze_graph_tower:
    #         for p in model.get_model().get_graph_tower().parameters():
    #             p.requires_grad = True
    #     for p in model.get_model().graph_projector.parameters():
    #         p.requires_grad = True
    #     if args.use_cross_attn:
    #         for p in model.get_model().graph_token_cross_attn.parameters():
    #             p.requires_grad = True

    #     model.set_if_only_train_graph(args.only_train_graph)

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

        if args.freeze_layers > 1:
            for name, param in model.named_parameters():
                if layer_index_from_name(name) < args.freeze_layers:
                    param.requires_grad = False

        model.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, hf_key, model_type, append_eos


def compute_norm(state_dict, use_lora = True, removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "embed_tokens", "norm", "word_embeddings", "quant", "absmax"]):
    norm = 0
    for key, val in state_dict.items():
        if use_lora:
            if "lora" in key:
                norm += val.clone().square().sum().item()
        else:
            if any([rkey in key for rkey in removing_keys]):
                    continue
            norm += val.clone().square().sum().item()
    return np.sqrt(norm)

def generate_state_dict(model, state_dict, coef, device="cpu", removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "embed_tokens", "norm", "word_embeddings" ]):
    new_state_dict = {}; cur_len = 0
    for key, param in model.named_parameters():
        if not param.requires_grad: continue
        param_len = param.numel()
        if any([rkey in key for rkey in removing_keys]):
            continue
            # new_state_dict[key] = state_dict[key].clone()
        else:
            new_state_dict[key] = state_dict[key].clone().to(device) + \
                torch.Tensor(coef[cur_len:cur_len+param_len].reshape(param.shape)).to(device)
            cur_len += param_len
    return new_state_dict

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

    # compute gradient arguments
    parser.add_argument("--number_of_subsets", type=int, default=1)
    parser.add_argument("--subset_seed", type=int, default=0)
    parser.add_argument("--subset_size", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--lr_regularization_lambda", type=float, default=1)
    parser.add_argument("--freeze_layers", type=int, default=0)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--compute_gradient_steps", type=int, default=1e7)
    parser.add_argument("--compute_gradients_seed", type=int, default=0)
    parser.add_argument("--project_gradients_dim", type=int, default=-1)

    parser.add_argument("--load_model_dir", type=str, default="test")
    parser.add_argument("--write_results", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--generate_output", action="store_true")

    args = parser.parse_args()
    args.enable_checkpointing = not args.disable_checkpointing
    print("arguments".upper().center(80, "-"))
    print(args)
    print("-" * 80)

    extended_task_names = [f"{task_name}" for task_name in args.task_names]
    model_key = args.model_key.replace("/", "-").replace("..", "")
    save_name = model_key + \
                (f"_{args.save_name}" if args.save_name else "") + \
                ("_".join(extended_task_names) if len("_".join(extended_task_names)) <= 100 else "{}_tasks".format(len(extended_task_names))) + \
                (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                (f"_use_only_answer_output" if args.only_answer_output else "")
    gradients_dir = save_name + f"_dim_{args.project_gradients_dim}_seed_{str(args.compute_gradients_seed)}"
    file_dir = os.path.join("./results/", save_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    metrics = {}
    model, tokenizer, hf_key, model_type, append_eos = initialize_model(args)

    batch_size = args.batch_size
    if args.inference_batch_size is None:
        inference_batch_size = batch_size
    else:
        inference_batch_size = args.inference_batch_size

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

    lm = MultitaskModel(model, tokenizer, model_type, use_cpu_offload=False,
                    lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, max_output_length=args.max_output_length, use_wandb=args.use_wandb, 
                    optimizer=args.optimizer, generate_output=args.generate_output, task_names=extended_task_names, eval_clrs=args.eval_last_step,
                    compute_gradients=True, gradients_dir=gradients_dir,
                    project_gradients_dim=args.project_gradients_dim, compute_gradients_seed=args.compute_gradients_seed, 
                    compute_gradients_steps=args.compute_gradient_steps, start_step=args.start_step, only_compute_outputs=True)
    
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

    if not os.path.exists("external_lightning_logs"):
        raise Exception("external_lightning_logs/ does not exist")
    default_root_dir = os.path.join("external_lightning_logs", 
                                    f"{model_key}_" + \
                                    ("_".join(extended_task_names) if len("_".join(extended_task_names)) <= 100 else "{}_tasks".format(len(extended_task_names))) + \
                                    (f"_use_only_answer_output" if args.only_answer_output else "") + \
                                    (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                                    (f"_{args.save_name}" if args.save_name else "")
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
                        callbacks=[checkpoint_callback], use_distributed_sampler=False, inference_mode=False
                        )
    # save initial weights
    # if args.train_lora:
    #     if not os.path.exists(os.path.join("gradients", gradients_dir)):
    #         os.makedirs(os.path.join("gradients", gradients_dir))
    #     model_path = os.path.join("gradients", gradients_dir) + "/initial_weights.pt"
    #     state_dict = model.state_dict()
    #     state_dict = {k: v.clone() for k, v in state_dict.items() if "lora" in k}
    #     torch.save(state_dict, model_path)

    state_dict = {key: val.clone() for key, val in lm.model.state_dict().items() if ("quant" not in key) and ("absmax" not in key)}
    pretrain_norm = compute_norm(state_dict, use_lora = True)
    print("Norm of the original model", pretrain_norm)

    def fit_linear_model(gradients, outputs=None, labels=None):
        # randomly assign labels as 0 or 1
        labels = np.random.binomial(n=1, p=0.7, size=gradients.shape[0])
        # reverse the gradients for the 0 labels
        mask = np.copy(labels)
        mask[labels == 0] = -1
        mask = mask.reshape(-1, 1)
        gradients = gradients*mask

        if outputs is None:
            # estimate parameters: train a logistic regression model
            clf = LogisticRegression(penalty='l2',  solver='lbfgs', C=1/args.lr_regularization_lambda) 
            clf.fit(gradients, labels)
            print("Linear regression score: ", clf.score(gradients, labels))
            proj_coef = clf.coef_.copy().flatten().reshape(-1, 1)
            print("L2 norm before projection", np.linalg.norm(proj_coef))
        else:
            # concatenate outputs
            print("Also using outputs for linear regression")
            outputs = outputs*mask.flatten()
            gradients = np.concatenate([gradients, -outputs.reshape(-1, 1)], axis=1)
            # estimate parameters: train a logistic regression model
            clf = LogisticRegression(penalty='l2',  solver='lbfgs', C=1/args.lr_regularization_lambda, fit_intercept=False) 
            clf.fit(gradients, labels)
            print("Linear regression score: ", clf.score(gradients, labels))
            proj_coef = clf.coef_.copy().flatten().reshape(-1, 1)[:-1] # remove the last column corresponding to the output
            print("L2 norm before projection", np.linalg.norm(proj_coef))
        
        # convert the coefficients to the original space
        project_matrix = lm.project_matrix
        if project_matrix is not None:
            coef = project_matrix @ proj_coef.flatten()
        else:
            coef = proj_coef.flatten()
        print("L2 norm before scaling", np.linalg.norm(coef))

        return coef

    def load_gradient_features(seed, task_name, gradients_dir):
        cur_gradients_dir = gradients_dir

        gradients = []
        for filename in os.listdir(f"./gradients/{cur_gradients_dir}/{task_name}"):
            if filename.endswith("_gradients.npy"):
                gradients.append(np.load(f"./gradients/{cur_gradients_dir}/{task_name}/{filename}"))
        gradients = np.concatenate(gradients, axis=0)
        return gradients
    
    def load_outputs(seed, task_name, gradients_dir):
        cur_gradients_dir = gradients_dir

        outputs = []
        for filename in os.listdir(f"./gradients/{cur_gradients_dir}/{task_name}"):
            if filename.endswith("_outputs.npy"):
                outputs.append(np.load(f"./gradients/{cur_gradients_dir}/{task_name}/{filename}"))
        outputs = np.concatenate(outputs, axis=0)
        outputs = outputs[outputs != 0] # in case the outputs are cut off by the maximum length
        return outputs

    def gradient_based_estimation(task_idxes, scale, seed=0):
        # load gradients
        all_gradients = []; all_outputs = []
        for task_idx in task_idxes:
            task_gradients = load_gradient_features(seed, args.task_names[task_idx], gradients_dir)
            task_outputs = load_outputs(seed, args.task_names[task_idx], gradients_dir)
            all_gradients.append(task_gradients); all_outputs.append(task_outputs)
        all_gradients = np.concatenate(all_gradients, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0) if all_outputs[0] is not None else None

        if len(all_gradients) == 0:
            return {}
        
        gradients = np.copy(all_gradients)
        outputs = np.copy(all_outputs) if all_outputs is not None else None
        print("Number of gradients for logistic regression", gradients.shape)
        coef = fit_linear_model(gradients, outputs=outputs)
        coef = coef * scale / np.linalg.norm(coef) 
        print("L2 norm after scaling", np.linalg.norm(coef))
        
        # evaluate task performances
        new_state_dict = generate_state_dict(lm.model, state_dict, coef, device=lm.model.device)
    
        return new_state_dict, all_gradients, all_outputs
    
    diff_list = []; inv_project_matrix = np.linalg.pinv(lm.project_matrix) if lm.project_matrix is not None else None
    for i in range(args.number_of_subsets):
        rng = np.random.default_rng(args.subset_seed + i)
        task_idxes = rng.choice(len(args.task_names), size=int(args.subset_size * len(args.task_names)), replace=False)
        scale = pretrain_norm * args.scale
        new_state_dict, gradients, pretrain_outputs = gradient_based_estimation(task_idxes, scale, seed=args.compute_gradients_seed)

        pretrain_state_dict = state_dict
        finetuned_state_dict = new_state_dict
        lm.model.load_state_dict(pretrain_state_dict)
        lm.model.load_state_dict(finetuned_state_dict, strict=False)

        finetuned_vector = [finetuned_state_dict[key]-pretrain_state_dict[key] for key in finetuned_state_dict.keys()]
        finetuned_vector = np.concatenate([vec.flatten().cpu().numpy() for vec in finetuned_vector]).reshape(1,-1)
        print("Finetuned vector shape", finetuned_vector.shape)

        # load compute new outputs
        tmp_task_names = [args.task_names[idx] for idx in task_idxes]
        new_data_module = TextCLRSDataModule(
                task_names=tmp_task_names,
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
        new_data_module.setup(stage="fit")
        tmp_task_names_str = "_".join(tmp_task_names)
        new_gradients_dir = gradients_dir + f"/subsets/{tmp_task_names_str}"
        if not os.path.exists(f"./gradients/{new_gradients_dir}"):
            os.makedirs(f"./gradients/{new_gradients_dir}")
            for task_name in tmp_task_names:
                os.makedirs(f"./gradients/{new_gradients_dir}/{task_name}")
        lm.gradients_dir = f"./gradients/{new_gradients_dir}"
        _ = trainer.predict(lm, dataloaders=new_data_module.train_dataloader())

        finetuned_outputs = []
        for task_name in tmp_task_names:
            tmp_outputs = load_outputs(args.compute_gradients_seed, task_name, new_gradients_dir)
            finetuned_outputs.append(tmp_outputs)
        finetuned_outputs = np.concatenate(finetuned_outputs, axis=0)
        print(pretrain_outputs)
        print("pretrain_outputs.shape",pretrain_outputs.shape)
        print(finetuned_outputs)
        print("finetuned_outputs.shape",finetuned_outputs.shape)

        if inv_project_matrix is not None:
            print(gradients.shape, inv_project_matrix.shape, finetuned_vector.shape)
            gradients = gradients @ inv_project_matrix
        first_order = (gradients * finetuned_vector).sum(axis=1)
        print("First-order term", first_order)

        mask = np.logical_and(pretrain_outputs != 0, finetuned_outputs != 0)
        mask = np.logical_and(mask, ~np.isnan(pretrain_outputs))
        mask = np.logical_and(mask, ~np.isnan(finetuned_outputs))
        pretrain_outputs[~mask] = 0 
        finetuned_outputs[~mask] = 0

        diff = np.abs(pretrain_outputs + first_order - finetuned_outputs) / np.maximum(np.abs(finetuned_outputs), np.abs(pretrain_outputs))
        diff = diff[~np.isnan(diff)]
        diffs = np.square(diff).mean()
        print("Mean Difference:", diffs)
        diff_list.append(diffs)
    print("Average Difference:", np.mean(diff_list), "std:", np.std(diff_list))
