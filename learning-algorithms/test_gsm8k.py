import argparse
import logging
import os
import re
from typing import Optional, Tuple

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO, force=True)


def build_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer: #### "


def extract_gsm8k_answer(answer_text: str) -> Optional[str]:
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip().replace(",", "")
    match = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?", answer_text)
    if not match:
        return None
    return match[-1].replace(",", "")


def parse_number(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    if "/" in text:
        parts = text.split("/")
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except ValueError:
                return None
    try:
        return float(text)
    except ValueError:
        return None


def compare_numbers(pred: Optional[str], gold: Optional[str]) -> bool:
    pred_num = parse_number(pred)
    gold_num = parse_number(gold)
    if pred_num is None or gold_num is None:
        return False
    return abs(pred_num - gold_num) < 1e-4


def tokenize_batch(tokenizer, prompts, targets, max_length):
    input_texts = [f"{p} {t}" for p, t in zip(prompts, targets)]
    inputs = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    prompt_tokens = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = inputs["input_ids"].clone()
    prompt_lens = prompt_tokens["attention_mask"].sum(dim=1)
    total_lens = inputs["attention_mask"].sum(dim=1)
    pad_offsets = inputs["input_ids"].size(1) - total_lens
    for i in range(len(prompts)):
        start = pad_offsets[i].item()
        end = min(start + prompt_lens[i].item(), inputs["input_ids"].size(1))
        labels[i, start:end] = -100
    return inputs, labels


def evaluate_loss(model, tokenizer, dataset, batch_size, max_length, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for start in tqdm(range(0, len(dataset), batch_size), desc="Loss", unit="batch"):
        batch = dataset[start:start + batch_size]
        prompts = [build_prompt(q) for q in batch["question"]]
        targets = batch["answer"]
        inputs, labels = tokenize_batch(tokenizer, prompts, targets, max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item()
            total_batches += 1
    return total_loss / max(1, total_batches)


def evaluate_accuracy(model, tokenizer, dataset, batch_size, max_length, max_new_tokens, device):
    model.eval()
    correct = 0
    total = 0
    for start in tqdm(range(0, len(dataset), batch_size), desc="Accuracy", unit="batch"):
        batch = dataset[start:start + batch_size]
        prompts = [build_prompt(q) for q in batch["question"]]
        gold_answers = [extract_gsm8k_answer(a) for a in batch["answer"]]
        inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
        decoded = []
        for output_ids, input_len in zip(outputs, input_lens):
            continuation_ids = output_ids[input_len:]
            decoded.append(tokenizer.decode(continuation_ids, skip_special_tokens=True))
        for pred, gold in zip(decoded, gold_answers):
            pred_ans = extract_gsm8k_answer(pred)
            if compare_numbers(pred_ans, gold):
                correct += 1
            total += 1
    return 100.0 * correct / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_key)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_key)
    model.to(device)

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    loss = evaluate_loss(
        model,
        tokenizer,
        dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    accuracy = evaluate_accuracy(
        model,
        tokenizer,
        dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )

    print(f"Final test loss: {loss:.6f}")
    print(f"Final test accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
