import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader, SequentialSampler, IterableDataset
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import *
from torch.utils.data import BatchSampler

from src.utils.multitask_dataset import MultitaskDataset, MultitaskBatchSampler, MultitaskCollator
from datasets import concatenate_datasets, load_dataset

import torch
import numpy as np

def build_causal_lm_instruction_batch(
    tokenizer,
    converted_batch,
    padding=True,
    max_source_length=None,
    max_target_length=None,
    label_pad_token_id=-100,
    return_tensors="pt",
):
    # prepare input sources
    sources = []; source_lengths = []
    for instance in converted_batch:
        source = instance["input"]
        source = source.replace("\n", " ")
        source = " ".join(source.split())
        tokenized_source = tokenizer(source)["input_ids"]
        if len(tokenized_source) <= max_source_length:
            sources.append(source)
        else:
            sources.append(tokenizer.decode(tokenized_source[:max_source_length], skip_special_tokens=True))
        source_lengths.append(min(len(tokenized_source), max_source_length))

    labels = []; label_lengths = []
    for instance in converted_batch:
        label = instance["output"]
        label = label.replace("\n", " ")
        label = " ".join(label.split())
        tokenized_label = tokenizer(label)["input_ids"]
        if len(tokenized_label) <= max_target_length:
            labels.append(label)
        else:
            labels.append(tokenizer.decode(tokenized_label[:max_target_length], skip_special_tokens=True))
        label_lengths.append(min(len(tokenized_label), max_target_length))

    inputs = [source + " " + label for source, label in zip(sources, labels)]

    model_inputs = tokenizer(
            text = inputs,
            max_length=max_source_length + max_target_length,
            padding=padding,
            return_tensors=return_tensors,
            truncation=True)

    # Lightning calls `.to(device)` on BatchEncoding, which fails once we attach
    # raw Python lists for logging. Convert to a plain dict so device transfer
    # recurses over tensor fields and leaves string lists on CPU.
    model_inputs = dict(model_inputs)

    # prepare labels
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    label_mask = model_inputs["attention_mask"].clone().bool()
    model_inputs["labels"] = model_inputs["labels"].masked_fill(~label_mask, label_pad_token_id)
    for i, length in enumerate(source_lengths):
        model_inputs["labels"][i, :length] = label_pad_token_id

    if "weights" in converted_batch[0]:
        model_inputs["weights"] = torch.Tensor([instance["weights"] for instance in converted_batch])

    if "sample_idx" in converted_batch[0]:
        model_inputs["sample_idx"] = torch.tensor([instance["sample_idx"] for instance in converted_batch], dtype=torch.long)

    if "residuals" in converted_batch[0]:
        model_inputs["residuals"] = torch.Tensor([instance["residuals"] for instance in converted_batch])

    if "length" in converted_batch[0]:
        model_inputs["length"] = torch.tensor([int(instance["length"]) for instance in converted_batch], dtype=torch.long)

    model_inputs["raw_inputs"] = [instance["input"] for instance in converted_batch]
    if "answer" in converted_batch[0]:
        model_inputs["raw_answers"] = [instance["answer"] for instance in converted_batch]
    if "output" in converted_batch[0]:
        model_inputs["raw_outputs"] = [instance["output"] for instance in converted_batch]

    return model_inputs

@dataclass
class CasualLMInstructionCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None # maximum length of the output
    max_target_length: Optional[int] = None # maximum length of the input
    pad_to_multiple_of: Optional[int] = None 
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
                return_tensors = self.return_tensors

        return build_causal_lm_instruction_batch(
            self.tokenizer,
            batch,
            padding=self.padding,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            label_pad_token_id=self.label_pad_token_id,
            return_tensors=return_tensors,
        )

def get_length(question):
    start_index = question.find('A:')
    end_index = question.find(', initial_trace:')
    question = question[start_index:end_index]
    return len(question.split(','))

def get_length_lego(question):
    """Get length for lego dataset from variables field"""
    # Extract variables line: "variables: t u d p s e a n x y i f"
    variables_start = question.find('variables:')
    if variables_start == -1:
        return 0
    variables_line = question[variables_start:].split('\n')[0]
    variables = variables_line.replace('variables:', '').strip().split()
    return len(variables)

class add_length:

    def __init__(self, is_lego=False):
        self.is_lego = is_lego

    def __call__(self, examples):
        if self.is_lego:
            examples["length"] = [get_length_lego(q) for q in examples['question']]
        else:
            examples["length"] = [get_length(q) for q in examples['question']]
        return examples

class convert_format:

    def __init__(self, only_answer_output = False, sample_steps=20, reduce_steps_ratio=1.0, reduce_steps_equally_spaced=False):
        self.only_answer_output = only_answer_output
        self.sample_steps = sample_steps
        self.reduce_steps_ratio = reduce_steps_ratio
        self.reduce_steps_equally_spaced = reduce_steps_equally_spaced

    def _reduce_step_choices(self, choices):
        target_num_steps = max(1, int(len(choices) * self.reduce_steps_ratio))
        if self.reduce_steps_equally_spaced:
            if target_num_steps >= len(choices):
                return choices
            selected_positions = np.linspace(0, len(choices) - 1, target_num_steps)
            selected_positions = np.round(selected_positions).astype(int)
            return choices[selected_positions]

        return np.random.choice(choices, target_num_steps, replace=False)

    def __call__(self, examples):
        examples["input"] = examples["question"][:]
        examples["only_answer"] = [answer.split("|")[-1].strip() for answer in examples["answer"]]
        if self.only_answer_output:
            examples["output"] = examples["only_answer"]
        else:
            only_answers = [answer.split("|")[-1].strip() for answer in examples["answer"]]
            intermediate_steps = [answer.split("|")[0].strip().split(",") for answer in examples["answer"]]
            for i, item in enumerate(intermediate_steps):
                # print("Original number of steps: ", len(item))
                choices = np.random.choice(len(item), min(len(item), self.sample_steps), replace=False) # if there are too many intermediate steps, randomly sample some of them
                if self.reduce_steps_ratio < 1.0:
                    choices = self._reduce_step_choices(choices)  # reduce the number of steps by the given ratio
                # print("Reduced number of steps: ", len(choices))
                choices = sorted(choices)
                intermediate_steps[i] = ", ".join([item[j] for j in choices])
            examples["output"] = [intermediate_steps[i] + " | " + only_answers[i] for i in range(len(only_answers))]
        return examples
    
class convert_few_shot_format:
    
    def __init__(self, train_dataset, only_answer_output=False, k=5, seed=42):
        self.train_dataset = train_dataset
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.only_answer_output = only_answer_output

    def _concat_examples(self, examples, question):
        output = "".join([example["question"] + example["answer"] for example in examples])
        output += question
        return output

    def __call__(self, examples):
        # randomly select from train dataset
        examples["input"] = []
        examples["output"] = []
        for i in range(len(examples["question"])):
            few_shot_examples = self.rng.choice(len(self.train_dataset), self.k, replace=False)
            few_shot_examples = [self.train_dataset[int(few_shot_example)] for few_shot_example in few_shot_examples]
            examples["input"].append(self._concat_examples(few_shot_examples, examples["question"][i]))
            examples["output"].append(examples["answer"][i])
        examples["only_answer"] = [answer.split("|")[-1].strip() for answer in examples["answer"]]
        if self.only_answer_output:
            examples["output"] = examples["only_answer"]
        return examples

class TextCLRSDataModule(pl.LightningDataModule):
    
    def __init__(
        self,
        task_names, 
        tokenizer,
        batch_size=8,
        inference_batch_size=32,
        max_input_length=512,
        max_output_length=64,
        shuffle_train=True,
        eval_all=False,
        train_lengths=[4],
        test_lengths=[4],
        eval_split=0.1,
        downsample_ratio=1.0, # ratio of downsampling
        minimum_samples=100,
        minimum_samples_validation=100,
        downsample_seed=0,
        use_few_shot=False,
        few_shot_k=5,
        only_answer_output=False,
        reduce_steps_ratio=1.0,
        reduce_steps_equally_spaced=False,
        eval_test_during_fit=False,
    ):
        super().__init__()

        self.task_names = task_names

        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.batch_size = batch_size
        if inference_batch_size is None:
            self.inference_batch_size = batch_size
        else:
            self.inference_batch_size = inference_batch_size
        self.shuffle_train = shuffle_train
        self.eval_all = eval_all

        self.train_lengths = train_lengths
        self.test_lengths = test_lengths
        self.eval_split = eval_split # the ratio of the validation set
        self.downsample_rate = downsample_ratio
        self.downsample_seed = downsample_seed
        self.minimum_sample = minimum_samples
        self.minimum_sample_validation = minimum_samples_validation
        self.use_few_shot = use_few_shot
        self.few_shot_k = few_shot_k
        self.only_answer_output = only_answer_output
        self.reduce_steps_ratio = reduce_steps_ratio
        self.reduce_steps_equally_spaced = reduce_steps_equally_spaced
        self.eval_test_during_fit = eval_test_during_fit

    def setup(self, stage=None):
        self.task_to_train_datasets = {}
        self.task_to_valid_datasets = {}
        self.task_to_test_datasets = {}
        self.task_to_collators = {}
        self.task_to_templates = {}
        for i, task_name in enumerate(self.task_names):
            # Check if this is a local dataset task (lego, cyclic, or symmetry)
            is_local_dataset = (task_name == "lego" or task_name.startswith("lego_") or 
                              task_name == "cyclic" or task_name.startswith("cyclic_") or
                              task_name == "symmetric" or task_name.startswith("symmetric_") or
                              task_name == "symmetry" or task_name.startswith("symmetry_"))
            
            if is_local_dataset:
                # Determine dataset directory and file prefix
                # Map task names to dataset directory and file prefix
                if task_name == "lego" or task_name.startswith("lego_"):
                    dataset_dir = "lego_dataset"
                    file_prefix = "lego"
                    file_suffix = "_progressive"  # lego uses _progressive suffix
                elif task_name == "cyclic" or task_name.startswith("cyclic_"):
                    dataset_dir = "lego_dataset"  # cyclic examples are in lego_dataset
                    file_prefix = "lego"
                    file_suffix = "_progressive"
                elif task_name == "symmetric" or task_name.startswith("symmetric_") or \
                     task_name == "symmetry" or task_name.startswith("symmetry_"):
                    dataset_dir = "symmetry_dataset"
                    file_prefix = "symmetry"
                    file_suffix = ""  # symmetry doesn't use suffix
                else:
                    raise ValueError(f"Unknown local dataset task: {task_name}")
                
                # Load dataset from local JSON files
                # Note: local datasets have fixed length for all samples, so we skip length filtering
                train_dataset = load_dataset("json", data_files=f"data/{dataset_dir}/{file_prefix}_train{file_suffix}.json")['train']
                train_dataset = train_dataset.map(add_length(is_lego=True), batched=True)
                # Skip length filtering for local datasets since all samples have the same length
                # convert the input and output format
                sample_steps = 10 if task_name in ["mst_kruskal", "floyd_warshall"] else 20
                train_dataset = train_dataset.map(convert_format(only_answer_output=self.only_answer_output, sample_steps=sample_steps, reduce_steps_ratio=self.reduce_steps_ratio, reduce_steps_equally_spaced=self.reduce_steps_equally_spaced), batched=True, load_from_cache_file=False)
                
                # Load validation dataset
                eval_dataset = load_dataset("json", data_files=f"data/{dataset_dir}/{file_prefix}_val{file_suffix}.json")['train']
                eval_dataset = eval_dataset.map(add_length(is_lego=True), batched=True)
                # Skip length filtering for local datasets since all samples have the same length
                eval_dataset = eval_dataset.map(convert_format(only_answer_output=self.only_answer_output, sample_steps=sample_steps, reduce_steps_ratio=self.reduce_steps_ratio, reduce_steps_equally_spaced=self.reduce_steps_equally_spaced), batched=True, load_from_cache_file=False)
                
                # Load test dataset
                predict_dataset = load_dataset("json", data_files=f"data/{dataset_dir}/{file_prefix}_test{file_suffix}.json")['train']
                predict_dataset = predict_dataset.map(add_length(is_lego=True), batched=True)
                # Skip length filtering for local datasets since all samples have the same length
                # convert the input and output format
                if self.use_few_shot:
                    predict_dataset = predict_dataset.map(convert_few_shot_format(train_dataset, only_answer_output=self.only_answer_output, k=self.few_shot_k), batched=True, load_from_cache_file=False)
                else:
                    predict_dataset = predict_dataset.map(convert_format(only_answer_output=self.only_answer_output, sample_steps=sample_steps, reduce_steps_ratio=self.reduce_steps_ratio, reduce_steps_equally_spaced=self.reduce_steps_equally_spaced), batched=True, load_from_cache_file=False)
            else:
                # Original CLRS dataset loading
                # Split the dataset into train and validation
                train_dataset = load_dataset("tomg-group-umd/CLRS-Text-train")['train']
                train_dataset = train_dataset.filter(lambda x: x['algo_name'] == task_name)
                train_dataset = train_dataset.map(add_length(is_lego=False), batched=True)
                train_dataset = train_dataset.filter(lambda x: x['length'] in self.train_lengths) if task_name != "bridges" else \
                    train_dataset.filter(lambda x: x['length'] in [5])
                # fileter out the examples by the text encoder
                column_names = train_dataset.column_names
                # convert the input and output format
                sample_steps = 10 if task_name in ["mst_kruskal", "floyd_warshall"] else 20
                train_dataset = train_dataset.map(convert_format(only_answer_output=self.only_answer_output, sample_steps=sample_steps, reduce_steps_ratio=self.reduce_steps_ratio, reduce_steps_equally_spaced=self.reduce_steps_equally_spaced), batched=True, load_from_cache_file=False) # remove_columns=column_names
                # split dataset
                tmp_datasets = train_dataset.train_test_split(test_size=self.eval_split, seed=42)
                train_dataset = tmp_datasets['train']
                eval_dataset = tmp_datasets['test']
                
                predict_dataset = load_dataset("tomg-group-umd/CLRS-Text-test")['test_1']
                predict_dataset = predict_dataset.filter(lambda x: x['algo_name'] == task_name)
                predict_dataset = predict_dataset.filter(lambda x: x['length'] in self.test_lengths) if task_name != "bridges" else \
                    predict_dataset.filter(lambda x: x['length'] in [5])
                # fileter out the examples by the text encoder
                column_names = predict_dataset.column_names
                # convert the input and output format
                if self.use_few_shot:
                    predict_dataset = predict_dataset.map(convert_few_shot_format(train_dataset, only_answer_output=self.only_answer_output, k=self.few_shot_k), batched=True, load_from_cache_file=False)
                else:
                    predict_dataset = predict_dataset.map(convert_format(only_answer_output=self.only_answer_output, sample_steps=sample_steps, reduce_steps_ratio=self.reduce_steps_ratio, reduce_steps_equally_spaced=self.reduce_steps_equally_spaced), batched=True, load_from_cache_file=False)

            print("Original train_dataset size: ", len(train_dataset))
            print("Original eval_dataset size: ", len(eval_dataset))
            print("Original predict_dataset size: ", len(predict_dataset))

            # Downsample the dataset if needed
            if self.downsample_rate < 1.0:
                rng = np.random.default_rng(self.downsample_seed)
                if "length" in train_dataset.column_names and len(self.train_lengths) > 1:
                    sampled_train_splits = []
                    for train_length in self.train_lengths:
                        length_subset = train_dataset.filter(lambda x, train_length=train_length: x["length"] == train_length)
                        if len(length_subset) == 0:
                            continue
                        permutations = rng.permutation(len(length_subset))
                        min_sample = max(int(self.minimum_sample), int(self.downsample_rate * len(length_subset)))
                        length_subset = length_subset.select(permutations[:min_sample])
                        sampled_train_splits.append(length_subset)
                    if len(sampled_train_splits) > 0:
                        train_dataset = concatenate_datasets(sampled_train_splits)
                else:
                    permutations = rng.permutation(len(train_dataset))
                    min_sample = max(int(self.minimum_sample), int(self.downsample_rate*len(train_dataset)))
                    train_dataset = train_dataset.select(permutations[:min_sample])

            if self.downsample_rate < 1.0:
                rng = np.random.default_rng(self.downsample_seed)
                if "length" in eval_dataset.column_names and len(self.train_lengths) > 1:
                    sampled_eval_splits = []
                    for train_length in self.train_lengths:
                        length_subset = eval_dataset.filter(lambda x, train_length=train_length: x["length"] == train_length)
                        if len(length_subset) == 0:
                            continue
                        permutations = rng.permutation(len(length_subset))
                        min_sample = max(int(self.minimum_sample_validation), int(self.downsample_rate * len(length_subset)))
                        length_subset = length_subset.select(permutations[:min_sample])
                        sampled_eval_splits.append(length_subset)
                    if len(sampled_eval_splits) > 0:
                        eval_dataset = concatenate_datasets(sampled_eval_splits)
                else:
                    permutations = rng.permutation(len(eval_dataset))
                    min_sample = max(int(self.minimum_sample_validation), int(self.downsample_rate*len(eval_dataset)))
                    eval_dataset = eval_dataset.select(permutations[:min_sample])

            if self.downsample_rate < 1.0:
                rng = np.random.default_rng(self.downsample_seed)
                # When multiple test lengths are requested, sample each length bucket
                # independently so each test subset gets the configured sample floor.
                if "length" in predict_dataset.column_names and len(self.test_lengths) > 1:
                    sampled_predict_splits = []
                    for test_length in self.test_lengths:
                        length_subset = predict_dataset.filter(lambda x, test_length=test_length: x["length"] == test_length)
                        if len(length_subset) == 0:
                            continue
                        permutations = rng.permutation(len(length_subset))
                        min_sample = max(int(self.minimum_sample_validation), int(self.downsample_rate * len(length_subset)))
                        length_subset = length_subset.select(permutations[:min_sample])
                        sampled_predict_splits.append(length_subset)
                    if len(sampled_predict_splits) > 0:
                        predict_dataset = concatenate_datasets(sampled_predict_splits)
                else:
                    permutations = rng.permutation(len(predict_dataset))
                    min_sample = max(int(self.minimum_sample_validation), int(self.downsample_rate*len(predict_dataset)))
                    predict_dataset = predict_dataset.select(permutations[:min_sample])
            
            # Add sample_idx to each sample BEFORE storing in task_to_train_datasets
            # This ensures each original data point gets a unique index (0, 1, 2, ..., N-1) per task
            if "sample_idx" not in train_dataset.column_names:
                def _add_sample_idx(example, idx):
                    example["sample_idx"] = int(idx)
                    return example
                train_dataset = train_dataset.map(_add_sample_idx, with_indices=True)
            
            extended_task_name = task_name
            print("Task: {} train dataset size: {} validation dataset size: {} test dataset size: {}".format(extended_task_name, len(train_dataset), len(eval_dataset), len(predict_dataset)))
            self.task_to_train_datasets[extended_task_name] = train_dataset
            self.task_to_valid_datasets[extended_task_name] = eval_dataset
            self.task_to_test_datasets[extended_task_name] = predict_dataset
            self.task_to_collators[extended_task_name] = CasualLMInstructionCollator(self.tokenizer, padding="max_length", 
                                                    max_source_length=self.max_input_length, max_target_length=self.max_output_length)

        self.multitask_train_dataset = MultitaskDataset(self.task_to_train_datasets)
        self.multitask_valid_dataset = MultitaskDataset(self.task_to_valid_datasets)
        self.multitask_test_dataset = MultitaskDataset(self.task_to_test_datasets)
        self.multitask_collator = MultitaskCollator(self.task_to_collators)
        self.multitask_train_sampler = MultitaskBatchSampler(sampler=np.arange(sum([len(dataset) for dataset in self.task_to_train_datasets.values()])), 
                                                                batch_size=self.batch_size, drop_last=False, task_to_datasets=self.task_to_train_datasets, shuffle=self.shuffle_train)
            # self.task_to_train_datasets, self.batch_size, shuffle=True)
        self.multitask_valid_sampler = MultitaskBatchSampler(sampler=np.arange(sum([len(dataset) for dataset in self.task_to_valid_datasets.values()])), 
                                                                batch_size=self.inference_batch_size, drop_last=False, task_to_datasets=self.task_to_valid_datasets, shuffle=False)
            # self.task_to_valid_datasets, self.inference_batch_size, shuffle=False)

        self.multitask_test_sampler = MultitaskBatchSampler(sampler=np.arange(sum([len(dataset) for dataset in self.task_to_test_datasets.values()])), 
                                                                batch_size=self.inference_batch_size, drop_last=False, task_to_datasets=self.task_to_test_datasets, shuffle=False)

        if hasattr(self, "residuals") and hasattr(self, "weights"):
            cur_len = 0
            for extended_task_name, train_dataset in self.task_to_train_datasets.items():
                self.task_to_train_datasets[extended_task_name] = train_dataset.add_column("weights", self.weights[cur_len: cur_len+len(train_dataset)]) # add weights to train dataset
                cur_len += len(train_dataset)

            cur_len = 0
            for extended_task_name, train_dataset in self.task_to_train_datasets.items():
                self.task_to_train_datasets[extended_task_name] = train_dataset.add_column("residuals", self.residuals[cur_len: cur_len+len(train_dataset)])
                cur_len += len(train_dataset)

            print("Weights and residuals loaded!", "Weights mean: ", self.weights.mean(), "Residuals mean: ", self.residuals.mean())

    def train_dataloader(self):
        return DataLoader(
            self.multitask_train_dataset,
            batch_sampler=self.multitask_train_sampler,
            collate_fn=self.multitask_collator,
            num_workers=15
        )

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.multitask_valid_dataset,
            batch_sampler=self.multitask_valid_sampler,
            collate_fn=self.multitask_collator,
            num_workers=15
        )

        if self.eval_test_during_fit:
            test_loader = DataLoader(
                self.multitask_test_dataset,
                batch_sampler=self.multitask_test_sampler,
                collate_fn=self.multitask_collator,
                num_workers=15
            )
            return [valid_loader, test_loader]

        return valid_loader

    def test_dataloader(self):
        return DataLoader(
            self.multitask_test_dataset,
            batch_sampler=self.multitask_test_sampler,
            collate_fn=self.multitask_collator,
            num_workers=15
        )
