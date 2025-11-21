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
import networkx as nx
from torch_geometric.data import Data, Batch

def to_torch(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    elif isinstance(value, torch.Tensor):
        return value
    else:
        return torch.tensor(value)
    
def convert_strA_to_adj(str_A):
    A = np.array([list(map(float, row.strip(' []').split())) for row in str_A.split(',') if row])
    return A

def to_node_labels(output_str):
    output = output_str.split("|")[1] if "|" in output_str else output_str
    output = output.strip(" []\n").split(" ")
    output = to_torch(np.array([int(o) for o in output]))
    return output

def to_data(input_str, output_str):
    data_dict = {}
    input_attributes = []
    
    # first get the edge index; create a fully connected graph 
    A_start = input_str.index('A: ')
    A_end = input_str.index(', initial_trace: ')
    str_A = input_str[A_start + 3: A_end] 
    A = convert_strA_to_adj(str_A)
    graph = nx.from_numpy_array(A, create_using=nx.DiGraph())
    edge_list = np.array(list(graph.edges())).T
    data_dict['edge_index'] = torch.tensor(edge_list, dtype=torch.long) # np.concatenate([edge_list, edge_list[::-1]], axis=1)
    num_nodes = A.shape[0]

    # add self loops
    unique_values = np.unique(A)
    is_weighted = unique_values.size != 2 or not np.all(unique_values == np.array([0,1]))
    if is_weighted:
        data_dict["weights"] = (A + np.eye(A.shape[0]))[data_dict["edge_index"][0], data_dict["edge_index"][1]]

    # Parse inputs
    additional_attributes = ['\ns:', ]
    for attribute in additional_attributes:
        if attribute in input_str:
            start = input_str.index(attribute)
            end = input_str.index('A:')
            value = int(input_str[start + len(attribute):end].strip(' ,'))
            feature = torch.zeros(num_nodes); feature[value] = 1
            input_attributes.append(attribute.strip('\n:'))
            data_dict[attribute.strip('\n:')] = feature
    
    data_dict['pos'] = torch.arange(0, 1, 1/num_nodes, dtype=torch.float)
    input_attributes.append('pos')

    # Parse outputs
    data_dict["y"] = to_node_labels(output_str)
    
    data_dict = {k: to_torch(v) for k,v in data_dict.items()}
    data = Data(**data_dict)    
    data.inputs = input_attributes
    return data

@dataclass
class CasualLMInstructionCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None # maximum length of the output
    max_target_length: Optional[int] = None # maximum length of the input
    pad_to_multiple_of: Optional[int] = None 
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    special_token_for_graphs = "<|reserved_special_token_0|>" # id: 128002

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
                return_tensors = self.return_tensors

        converted_batch = batch

        # convert the input to graphs
        batch_graphs = []; graph_sizes = []
        for instance in converted_batch:
            batch_graphs.append(to_data(instance["input"], instance["output"]))
            graph_sizes.append(len(batch_graphs[-1].pos))

        # prepare input sources
        original_sources = []
        for idx, instance in enumerate(converted_batch):
            # right now only use the special token for the nodes positions in the graphs. TODO: we can define some text instruction
            source = instance["input"]
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                original_sources.append(source)
            else:
                original_sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))

        # prepare input sources
        sources = []; source_lengths = []
        for idx, instance in enumerate(converted_batch):
            # right now only use the special token for the nodes positions in the graphs. TODO: we can define some text instruction
            source = instance["input"].split("\n")[0]
            source += "".join([self.special_token_for_graphs]*graph_sizes[idx]) 
            source += instance["input"].split("\n")[-2]
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
        
        original_input_ids = self.tokenizer(
                text = inputs, 
                padding="longest",
                return_tensors=self.return_tensors, 
                truncation=True)["input_ids"]
        model_inputs["original_input_ids"] = original_input_ids
        
        # prepare labels
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        label_mask = model_inputs["attention_mask"].clone().bool()
        model_inputs["labels"] = model_inputs["labels"].masked_fill(~label_mask, self.label_pad_token_id)
        for i, length in enumerate(source_lengths):
            model_inputs["labels"][i, :length] = self.label_pad_token_id    

        # Add the graph data
        batch_graphs = Batch.from_data_list(batch_graphs)
        batch_graphs.inputs = batch_graphs.inputs[0]
        model_inputs["graph_data"] = batch_graphs        

        if "weights" in converted_batch[0]:
            model_inputs["weights"] = torch.Tensor([instance["weights"] for instance in converted_batch])

        if "residuals" in converted_batch[0]:
            model_inputs["residuals"] = torch.Tensor([instance["residuals"] for instance in converted_batch])
        
        return model_inputs

def get_length(question):
    start_index = question.find('A:')
    end_index = question.find(', initial_trace:')
    question = question[start_index:end_index]
    return len(question.split(','))

class add_length:

    def __call__(self, examples):
        examples["length"] = [get_length(q) for q in examples['question']]
        return examples

class convert_format:

    def __init__(self, load_only_last_output=False):
        self.load_only_last_output = load_only_last_output

    def __call__(self, examples):
        examples["input"] = examples["question"][:]
        examples["output"] = examples["answer"][:]
        if self.load_only_last_output:
            examples["output"] = [output.split("|")[-1] for output in examples["output"]]
            examples["output"] = [output.strip() for output in examples["output"]]
        return examples
    
class convert_few_shot_format:
    
    def __init__(self, train_dataset, k=5, seed=42):
        self.train_dataset = train_dataset
        self.k = k
        self.rng = np.random.default_rng(seed)

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
        return examples

class TextGraphCLRSDataModule(pl.LightningDataModule):
    
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
        load_only_last_output=False
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
        self.load_only_last_output = load_only_last_output

    def setup(self, stage=None):
        self.task_to_train_datasets = {}
        self.task_to_valid_datasets = {}
        self.task_to_test_datasets = {}
        self.task_to_collators = {}
        self.task_to_templates = {}
        for i, task_name in enumerate(self.task_names):

            # Split the dataset into train and validation
            train_dataset = load_dataset("tomg-group-umd/CLRS-Text-train")['train']
            train_dataset = train_dataset.filter(lambda x: x['algo_name'] == task_name)
            train_dataset = train_dataset.map(add_length(), batched=True)
            train_dataset = train_dataset.filter(lambda x: x['length'] in self.train_lengths)
            # fileter out the examples by the text encoder
            column_names = train_dataset.column_names
            # convert the input and output format
            train_dataset = train_dataset.map(convert_format(load_only_last_output=self.load_only_last_output), batched=True) # remove_columns=column_names
            # split dataset
            tmp_datasets = train_dataset.train_test_split(test_size=self.eval_split, seed=42)
            train_dataset = tmp_datasets['train']
            eval_dataset = tmp_datasets['test']
            
            predict_dataset = load_dataset("tomg-group-umd/CLRS-Text-test")['test_1']
            predict_dataset = predict_dataset.filter(lambda x: x['algo_name'] == task_name)
            predict_dataset = predict_dataset.filter(lambda x: x['length'] in self.test_lengths)
            # fileter out the examples by the text encoder
            column_names = predict_dataset.column_names
            # convert the input and output format
            if self.use_few_shot:
                predict_dataset = predict_dataset.map(convert_few_shot_format(train_dataset, k=self.few_shot_k), batched=True)
            else:
                predict_dataset = predict_dataset.map(convert_format(load_only_last_output=self.load_only_last_output), batched=True)

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
        