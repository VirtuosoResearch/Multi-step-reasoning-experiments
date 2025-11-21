import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader, SequentialSampler, IterableDataset
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import *
from torch.utils.data import BatchSampler
from src.utils.multitask_dataset import MultitaskDataset, MultitaskBatchSampler, MultitaskCollator
from datasets import load_dataset

import torch
import numpy as np


class convert_format:

    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    def __init__(self, only_answer_output=False):
        self.only_answer_output = only_answer_output

    def __call__(self, examples):
        examples["input"] = [self.problem_prompt.format(instruction=item) for item in examples['question']]
        examples["only_answer"] = [answer.split('#### ')[1].replace(',', '') if '#### ' in answer else answer for answer in examples["answer"]]
        if self.only_answer_output:
            examples["output"] = [f"The answer is: {answer}" for answer in examples["only_answer"]]
        else:
            examples["output"] = [item + " The answer is: {}".format(examples["only_answer"][i]) for i, item in enumerate(examples["answer"])]
        return examples


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

        converted_batch = batch

        # prepare input sources
        sources = []; source_lengths = []
        for instance in converted_batch:
            source = instance["input"]
            source = source.replace("\n", " ")
            source = " ".join(source.split())
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))
            source_lengths.append(min(len(tokenized_source), self.max_source_length))

        labels = []; label_lengths = []
        for instance in converted_batch:
            label = instance["output"]
            label = label.replace("\n", " ")
            label = " ".join(label.split())
            tokenized_label = self.tokenizer(label)["input_ids"]
            if len(tokenized_label) <= self.max_target_length:
                labels.append(label)
            else:
                labels.append(self.tokenizer.decode(tokenized_label[:self.max_target_length], skip_special_tokens=True))
            label_lengths.append(min(len(tokenized_label), self.max_target_length))

        inputs = [source + " " + label for source, label in zip(sources, labels)]

        model_inputs = self.tokenizer(
                text = inputs, 
                max_length=self.max_source_length + self.max_target_length, 
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True)
        
        # prepare labels
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        label_mask = model_inputs["attention_mask"].clone().bool()
        model_inputs["labels"] = model_inputs["labels"].masked_fill(~label_mask, self.label_pad_token_id)
        for i, length in enumerate(source_lengths):
            model_inputs["labels"][i, :length] = self.label_pad_token_id            

        if "weights" in converted_batch[0]:
            model_inputs["weights"] = torch.Tensor([instance["weights"] for instance in converted_batch])

        if "residuals" in converted_batch[0]:
            model_inputs["residuals"] = torch.Tensor([instance["residuals"] for instance in converted_batch])
        
        only_answer = self.tokenizer(
            [instance["only_answer"] for instance in converted_batch],
            max_length=32,
            padding=self.padding,
            return_tensors=self.return_tensors,
            truncation=True
        )
        model_inputs["only_answer"] = only_answer["input_ids"]

        return model_inputs

class GSM8KDataModule(pl.LightningDataModule):
    
    def __init__(
        self,
        tokenizer,
        batch_size=8,
        inference_batch_size=32,
        max_input_length=512,
        max_output_length=64,
        shuffle_train=True,
        eval_split=0.1,
        downsample_ratio=1.0, # ratio of downsampling
        minimum_samples=100,
        minimum_samples_validation=100,
        downsample_seed=0,
        only_answer_output=False
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.batch_size = batch_size
        if inference_batch_size is None:
            self.inference_batch_size = batch_size
        else:
            self.inference_batch_size = inference_batch_size
        self.shuffle_train = shuffle_train

        self.eval_split = eval_split # the ratio of the validation set
        self.downsample_rate = downsample_ratio
        self.downsample_seed = downsample_seed
        self.minimum_sample = minimum_samples
        self.minimum_sample_validation = minimum_samples_validation
        self.only_answer_output = only_answer_output

    def setup(self, stage=None):
        self.task_to_train_datasets = {}
        self.task_to_valid_datasets = {}
        self.task_to_test_datasets = {}
        self.task_to_collators = {}
        self.task_to_templates = {}

        dataset = load_dataset("openai/gsm8k", "main")

        train_dataset = dataset['train']
        predict_dataset = dataset['test']

        train_dataset = train_dataset.map(convert_format(only_answer_output=self.only_answer_output), batched=True, load_from_cache_file=False)
        predict_dataset = predict_dataset.map(convert_format(only_answer_output=self.only_answer_output), batched=True, load_from_cache_file=False)

        if self.eval_split != 0:
            tmp_datasets = train_dataset.train_test_split(test_size=self.eval_split, seed=42)
            train_dataset = tmp_datasets['train']
            eval_dataset = tmp_datasets['test']
        else:
            eval_dataset = predict_dataset

        # Downsample the dataset if needed
        if self.downsample_rate < 1.0:
            rng = np.random.default_rng(self.downsample_seed)
            permutations = rng.permutation(len(train_dataset))
            min_sample = max(int(self.minimum_sample), int(self.downsample_rate*len(train_dataset)))
            train_dataset = train_dataset.select(permutations[:min_sample])

        if self.downsample_rate < 1.0:
            rng = np.random.default_rng(self.downsample_seed)
            permutations = rng.permutation(len(eval_dataset))
            min_sample = max(int(self.minimum_sample_validation), int(self.downsample_rate*len(eval_dataset)))
            eval_dataset = eval_dataset.select(permutations[:min_sample])

        if self.downsample_rate < 1.0:
            rng = np.random.default_rng(self.downsample_seed)
            permutations = rng.permutation(len(predict_dataset))
            min_sample = max(int(self.minimum_sample_validation), int(self.downsample_rate*len(predict_dataset)))
            predict_dataset = predict_dataset.select(permutations[:min_sample])

        # self.collator = CasualLMInstructionCollator(self.tokenizer, padding="max_length", 
        #                     max_source_length=self.max_input_length, max_target_length=self.max_output_length)

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

        print("Task: GSM8K train dataset size: {} validation dataset size: {} test dataset size: {}".format(len(train_dataset), len(eval_dataset), len(predict_dataset)))

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.predict_dataset = predict_dataset

        self.task_to_train_datasets["gsm8k"] = train_dataset
        self.task_to_valid_datasets["gsm8k"] = eval_dataset
        self.task_to_test_datasets["gsm8k"] = predict_dataset
        self.task_to_collators["gsm8k"] = CasualLMInstructionCollator(self.tokenizer, padding="max_length", 
                            max_source_length=self.max_input_length, max_target_length=self.max_output_length)

        self.multitask_train_dataset = MultitaskDataset(self.task_to_train_datasets)
        self.multitask_valid_dataset = MultitaskDataset(self.task_to_valid_datasets)
        self.multitask_test_dataset = MultitaskDataset(self.task_to_test_datasets)
        self.multitask_collator = MultitaskCollator(self.task_to_collators)
        self.multitask_train_sampler = MultitaskBatchSampler(sampler=np.arange(sum([len(dataset) for dataset in self.task_to_train_datasets.values()])), 
                                                                batch_size=self.batch_size, drop_last=False, task_to_datasets=self.task_to_train_datasets, shuffle=self.shuffle_train)
        self.multitask_valid_sampler = MultitaskBatchSampler(sampler=np.arange(sum([len(dataset) for dataset in self.task_to_valid_datasets.values()])), 
                                                                batch_size=self.inference_batch_size, drop_last=False, task_to_datasets=self.task_to_valid_datasets, shuffle=False)
        self.multitask_test_sampler = MultitaskBatchSampler(sampler=np.arange(sum([len(dataset) for dataset in self.task_to_test_datasets.values()])), 
                                                                batch_size=self.inference_batch_size, drop_last=False, task_to_datasets=self.task_to_test_datasets, shuffle=False)


    def train_dataloader(self):
        return DataLoader(
            self.multitask_train_dataset,
            batch_sampler=self.multitask_train_sampler,
            collate_fn=self.multitask_collator,
            num_workers=15
        )

    def val_dataloader(self):
        return DataLoader(
            self.multitask_valid_dataset,
            batch_sampler=self.multitask_valid_sampler,
            collate_fn=self.multitask_collator,
            num_workers=15
        )

    def test_dataloader(self):
        return DataLoader(
            self.multitask_test_dataset,
            batch_sampler=self.multitask_test_sampler,
            collate_fn=self.multitask_collator,
            num_workers=15
        )
        