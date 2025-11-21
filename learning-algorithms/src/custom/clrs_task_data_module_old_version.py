import pytorch_lightning as pl
import torch
import os
import numpy as np
import pickle
import torch
import pandas as pd
from torch.utils.data import DataLoader, SequentialSampler, IterableDataset
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import *
from torch.utils.data import BatchSampler

import glob
import tqdm
import random

from src.utils.multitask_dataset import MultitaskDataset, MultitaskBatchSampler, MultitaskCollator
from datasets import load_dataset

@dataclass
class Seq2SeqInstructionCollator:
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
        sources = []
        for instance in converted_batch:
            source = instance["input"]
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))
        model_inputs = self.tokenizer(
                sources, 
                max_length=self.max_source_length, 
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of)
        
        # prepare labels
        labels = [instance["output"] for instance in converted_batch]
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                labels,
                max_length=self.max_target_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of
            )
        label_mask = labels["attention_mask"].bool()
        model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)

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
                max_length=self.max_source_length, 
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
        
        return model_inputs

class convert_format:

    def __call__(self, examples):
        examples["input"] = examples["question"][:]
        examples["output"] = examples["answer"][:]
        return examples

class CLRSDataModule(pl.LightningDataModule):
    
    def __init__(
        self,
        task_names, 
        prompt_styles,
        tokenizer,
        batch_size=8,
        inference_batch_size=32,
        max_input_length=512,
        max_output_length=64,
        shuffle_train=True,
        eval_all=False,
        downsample_ratio=1.0, # ratio of downsampling
        minimum_samples=100,
        minimum_samples_validation=100,
        downsample_seed=0
    ):
        super().__init__()

        self.task_names = task_names # task_name
        self.prompt_styles = prompt_styles # zero_shot, zero_cot, few_shot, few_cot
        # "adjacency" "incident" "friendship" "south_park" "got" "politician"
        # "social_network" "expert" "coauthorship" "random" 

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

        self.downsample_rate = downsample_ratio
        self.downsample_seed = downsample_seed
        self.minimum_sample = minimum_samples
        self.minimum_sample_validation = minimum_samples_validation

    def setup(self, stage=None):
        self.task_to_train_datasets = {}
        self.task_to_valid_datasets = {}
        self.task_to_test_datasets = {}
        self.task_to_collators = {}
        self.task_to_templates = {}
        for i, task_name in enumerate(self.task_names):
            prompt_style = self.prompt_styles[i]

            # Split the dataset into train and validation
            task_file_dir = "./data/clrs_text_tasks/{}_train_{}.json".format(task_name, prompt_style)
            train_dataset = load_dataset("json", data_files=task_file_dir)['train']
            # fileter out the examples by the text encoder
            column_names = train_dataset.column_names
            # convert the input and output format
            train_dataset = train_dataset.map(convert_format(), batched=True, remove_columns=column_names)

            task_file_dir = "./data/clrs_text_tasks/{}_val_zero_shot.json".format(task_name)
            eval_dataset = load_dataset("json", data_files=task_file_dir)['train']
            # fileter out the examples by the text encoder
            column_names = eval_dataset.column_names
            # convert the input and output format
            eval_dataset = eval_dataset.map(convert_format(), batched=True, remove_columns=column_names)
            
            task_file_dir = "./data/clrs_text_tasks/{}_test_zero_shot.json".format(task_name)
            predict_dataset = load_dataset("json", data_files=task_file_dir)['train']
            # fileter out the examples by the text encoder
            column_names = predict_dataset.column_names
            # convert the input and output format
            predict_dataset = predict_dataset.map(convert_format(), batched=True, remove_columns=column_names)

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
            
            extended_task_name = task_name + "_" + prompt_style
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
        )

    def val_dataloader(self):
        return DataLoader(
            self.multitask_valid_dataset,
            batch_sampler=self.multitask_valid_sampler,
            collate_fn=self.multitask_collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.multitask_test_dataset,
            batch_sampler=self.multitask_valid_sampler,
            collate_fn=self.multitask_collator,
        )
        