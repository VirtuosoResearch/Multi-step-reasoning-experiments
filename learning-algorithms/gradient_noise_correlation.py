import argparse
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.custom.clrs_text_task_data_module import TextCLRSDataModule
from train_clrs_text import initialize_model


def move_batch_to_device(batch_data: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch_data.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def build_model_kwargs(batch_data: Dict) -> Dict:
    kwargs = {
        "input_ids": batch_data["input_ids"],
        "attention_mask": batch_data["attention_mask"],
        "labels": batch_data["labels"],
    }
    if "decoder_attention_mask" in batch_data:
        kwargs["decoder_attention_mask"] = batch_data["decoder_attention_mask"]
    return kwargs


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")

    x_ranks = np.argsort(np.argsort(x)).astype(np.float64)
    y_ranks = np.argsort(np.argsort(y)).astype(np.float64)

    x_std = x_ranks.std()
    y_std = y_ranks.std()
    if x_std == 0 or y_std == 0:
        return float("nan")

    return float(np.corrcoef(x_ranks, y_ranks)[0, 1])


def get_trainable_params(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Parameter]]:
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def compute_single_example_loss(model: torch.nn.Module, batch_data: Dict) -> torch.Tensor:
    kwargs = build_model_kwargs(batch_data)
    output = model(**kwargs)
    return output["loss"]


def collect_examples(data_module: TextCLRSDataModule, num_examples: int) -> List[Dict]:
    examples = []
    for wrapped_batch in data_module.train_dataloader():
        batch_data = wrapped_batch["data"]
        batch_size = batch_data["input_ids"].shape[0]
        for i in range(batch_size):
            single = {}
            for key, value in batch_data.items():
                if torch.is_tensor(value):
                    single[key] = value[i : i + 1]
                else:
                    single[key] = value
            examples.append(single)
            if len(examples) >= num_examples:
                return examples
    return examples


def maybe_load_weights(model: torch.nn.Module, load_model_path: str, use_graph_llama: bool) -> None:
    if not load_model_path:
        return
    if not os.path.exists(load_model_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_model_path}")

    state_dict = torch.load(load_model_path, map_location="cpu")
    target = model.model if use_graph_llama and hasattr(model, "model") else model
    target.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {load_model_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_names", type=str, nargs="+", default=["dfs"])
    parser.add_argument("--model_key", type=str, default="gpt2")
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--inference_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_output_length", type=int, default=64)

    parser.add_argument("--eval_split", type=float, default=0.2)
    parser.add_argument("--downsample_ratio", type=float, default=1.0)
    parser.add_argument("--minimum_samples", type=int, default=1000)
    parser.add_argument("--minimum_samples_validation", type=int, default=1000)
    parser.add_argument("--train_lengths", type=int, nargs="+", default=[4])
    parser.add_argument("--test_lengths", type=int, nargs="+", default=[4])
    parser.add_argument("--few_shot_k", type=int, default=0)
    parser.add_argument("--only_answer_output", action="store_true")
    parser.add_argument("--reduce_steps_ratio", type=float, default=1.0)
    parser.add_argument("--reduce_steps_equally_spaced", action="store_true")

    parser.add_argument("--num_examples", type=int, default=8)
    parser.add_argument("--num_perturbations", type=int, default=20)
    parser.add_argument("--noise_std", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_model_path", type=str, default=None)

    # Flags needed by initialize_model(args)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--use_graph_llama", action="store_true")
    parser.add_argument("--use_cross_attn", action="store_true")
    parser.add_argument("--add_output_projection", action="store_true")
    parser.add_argument("--alignment_loss_weight", type=float, default=0.0)
    parser.add_argument("--test_classifier_before_cross_attn", action="store_true")
    parser.add_argument("--freeze_graph_tower", action="store_true")
    parser.add_argument("--only_train_graph", action="store_true")
    parser.add_argument("--train_adapter", action="store_true")
    parser.add_argument("--use_qadapter", action="store_true")
    parser.add_argument("--reduction_factor", type=int, default=128)
    parser.add_argument("--use_3bit", action="store_true")
    parser.add_argument("--use_2bit", action="store_true")
    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--intrinsic_dim", type=int, default=0)
    parser.add_argument("--intrinsic_mode", type=str, default="rdkronqr")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.devices[0]}")
    else:
        device = torch.device("cpu")

    model, tokenizer, _, model_type, _ = initialize_model(args)
    maybe_load_weights(model, args.load_model_path, args.use_graph_llama)
    model.to(device)
    model.eval()

    data_module = TextCLRSDataModule(
        task_names=args.task_names,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        inference_batch_size=args.inference_batch_size,
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
        only_answer_output=args.only_answer_output,
        reduce_steps_ratio=args.reduce_steps_ratio,
        reduce_steps_equally_spaced=args.reduce_steps_equally_spaced,
    )
    data_module.setup(stage="fit")

    examples = collect_examples(data_module, args.num_examples)
    if len(examples) == 0:
        raise RuntimeError("No training examples were collected. Check task/data settings.")

    trainable_params = get_trainable_params(model)
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found.")

    grad_norms = []
    mean_loss_deltas = []

    for idx, example in enumerate(examples):
        single = move_batch_to_device(example, device)

        model.zero_grad(set_to_none=True)
        loss = compute_single_example_loss(model, single)
        original_loss = float(loss.item())
        loss.backward()

        grad_sq_sum = 0.0
        for _, p in trainable_params:
            if p.grad is not None:
                grad_sq_sum += float((p.grad.detach().float() ** 2).sum().item())
        grad_norm = math.sqrt(grad_sq_sum)

        original_weights = {name: p.detach().clone() for name, p in trainable_params}
        deltas = []
        with torch.no_grad():
            for _ in range(args.num_perturbations):
                for name, p in trainable_params:
                    p.copy_(original_weights[name])
                    p.add_(torch.randn_like(p) * args.noise_std)

                perturbed_loss = compute_single_example_loss(model, single)
                deltas.append(float(perturbed_loss.item()) - original_loss)

            for name, p in trainable_params:
                p.copy_(original_weights[name])

        mean_delta = float(np.mean(deltas))
        grad_norms.append(grad_norm)
        mean_loss_deltas.append(mean_delta)

        print(
            f"example={idx:03d} loss={original_loss:.6f} grad_norm={grad_norm:.6f} "
            f"mean_delta={mean_delta:.6f} delta_std={np.std(deltas):.6f}"
        )

    grad_arr = np.array(grad_norms, dtype=np.float64)
    delta_arr = np.array(mean_loss_deltas, dtype=np.float64)

    if len(grad_arr) < 2:
        pearson = float("nan")
        spearman = float("nan")
    else:
        pearson = float(np.corrcoef(grad_arr, delta_arr)[0, 1])
        spearman = spearman_correlation(grad_arr, delta_arr)

    print("\n=== Correlation Summary ===")
    print(f"num_examples: {len(grad_arr)}")
    print(f"num_perturbations: {args.num_perturbations}")
    print(f"noise_std: {args.noise_std}")
    print(f"pearson(grad_norm, mean_delta): {pearson:.6f}")
    print(f"spearman(grad_norm, mean_delta): {spearman:.6f}")

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gradient_noise_correlation.csv")
    np.savetxt(
        output_path,
        np.stack([grad_arr, delta_arr], axis=1),
        delimiter=",",
        header="grad_norm,mean_perturbed_minus_original_loss",
        comments="",
    )
    print(f"Saved per-example values to: {output_path}")


if __name__ == "__main__":
    main()
