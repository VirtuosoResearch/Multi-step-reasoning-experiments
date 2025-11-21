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

import json
import re
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, degree
import networkx as nx

import glob
import tqdm
import random

from src.utils.multitask_dataset import MultitaskDataset, MultitaskBatchSampler, MultitaskCollator
from datasets import load_dataset

def create_sample_graph():
    """
    Create a toy undirected graph with 10 nodes.
    (For a non-attributed graph we assign each node a constant feature.)
    """
    # Define some edges (this example graph is small and may have cycles)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 1, 2]], dtype=torch.long)
    num_nodes = 10
    x = torch.ones((num_nodes, 1))  # constant feature for each node
    data = Data(x=x, edge_index=edge_index)
    return data

def compute_degree_labels(data):
    """
    Compute each node's degree as a long tensor.
    (Used as a classification target. The number of classes is max_degree+1.)
    """
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
    return deg

def compute_cycle_labels(data):
    """
    Use networkx to mark nodes that lie in any cycle.
    Label is 1 if the node is in a cycle, 0 otherwise.
    """
    edge_index_np = data.edge_index.numpy()
    G = nx.Graph()
    num_nodes = data.num_nodes
    G.add_nodes_from(range(num_nodes))
    edges = list(zip(edge_index_np[0], edge_index_np[1]))
    G.add_edges_from(edges)
    
    cycles = nx.cycle_basis(G)
    cycle_nodes = set()
    for cycle in cycles:
        cycle_nodes.update(cycle)
        
    labels = torch.zeros(num_nodes, dtype=torch.long)
    for node in cycle_nodes:
        labels[node] = 1
    return labels

def compute_component_size_labels(data):
    """
    For each node, compute the size of its connected component.
    (Used as a regression target.)
    """
    edge_index_np = data.edge_index.numpy()
    G = nx.Graph()
    num_nodes = data.num_nodes
    G.add_nodes_from(range(num_nodes))
    edges = list(zip(edge_index_np[0], edge_index_np[1]))
    G.add_edges_from(edges)
    
    labels = torch.zeros(num_nodes, dtype=torch.float)
    for comp in nx.connected_components(G):
        comp_size = len(comp)
        for node in comp:
            labels[node] = comp_size
    return labels

def compute_graph_edge_count(data):
    """
    Compute the number of unique edges in the graph.
    (Assumes the graph is undirected.)
    """
    # Get unique undirected edges.
    pos_edge_index = data.edge_index.t().unique(dim=0)
    # For an undirected graph, you might count each edge once.
    # (Here, we assume the input edge_index already contains both directions.)
    return torch.tensor([pos_edge_index.size(0)], dtype=torch.float)

def generate_link_prediction_data(data, neg_ratio=1.0):
    """
    Generate positive and negative examples for edge (link) prediction.
    Returns:
      - edge_pairs: tensor of shape [num_samples, 2] containing node index pairs.
      - labels: tensor containing 1 for a positive edge, 0 for a negative sample.
    """
    # Positive edges (unique pairs; include both directions)
    pos_edge_index = data.edge_index.t().unique(dim=0)
    pos_set = set([ (u.item(), v.item()) for u, v in pos_edge_index ])
    # Ensure both (u,v) and (v,u) are in the set for an undirected graph.
    pos_set |= set([(v, u) for u, v in pos_set])
    
    num_nodes = data.num_nodes
    num_neg_samples = int(neg_ratio * len(pos_set))
    neg_edges = []
    while len(neg_edges) < num_neg_samples:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        if u == v:
            continue  # skip self-loops
        if (u, v) in pos_set:
            continue
        neg_edges.append((u, v))
    
    pos_edge_pairs = torch.tensor(list(pos_set), dtype=torch.long)
    neg_edge_pairs = torch.tensor(neg_edges, dtype=torch.long)
    
    pos_labels = torch.ones(pos_edge_pairs.size(0), dtype=torch.long)
    neg_labels = torch.zeros(neg_edge_pairs.size(0), dtype=torch.long)
    
    edge_pairs = torch.cat([pos_edge_pairs, neg_edge_pairs], dim=0)
    labels = torch.cat([pos_labels, neg_labels], dim=0)
    return edge_pairs, labels

def seq2graph2(seq, feature_dim=32):
    nodes_match = re.search(r"among nodes\s*(.*?)\.", seq)
    if nodes_match:
        nodes_str = nodes_match.group(1)
        # Extract all numbers from the captured string.
        nodes = list(map(int, re.findall(r'\d+', nodes_str)))
        # For PyG, we set num_nodes to be the maximum node index + 1.
        num_nodes = max(nodes) + 1
    else:
        raise ValueError("Nodes information not found in the input string.")

    # --- Extract the edges ---
    # Find all occurrences of edge tuples in the form (u, v)
    edge_tuples = re.findall(r"\((\d+),\s*(\d+)\)", seq)

    # Build the list of edges.
    # Since the graph is undirected, add both directions (u,v) and (v,u).
    edge_list = []
    for u, v in edge_tuples:
        u, v = int(u), int(v)
        edge_list.append((u, v))
        edge_list.append((v, u))

    # Convert the list to a tensor with shape [2, num_edges]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # --- Assign default weights ---
    # Give every edge a weight of 1.0.
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)

    # --- Add self-loops ---
    # For each node i, add an edge (i, i) with weight 1.0.
    edge_index, edge_weight = add_self_loops(
        edge_index, fill_value=1.0, num_nodes=num_nodes
    )
    #print(edge_index)
    # --- Create the PyG Data object ---
    x = torch.ones(num_nodes, feature_dim)  # constant feature for each node
    data = Data(num_nodes=num_nodes, edge_index=edge_index, edge_weight=edge_weight, x=x)
    # --- Compute node-level labels ---
    degree_labels = compute_degree_labels(data)    # Node degrees (long tensor)
    #cycle_labels = compute_cycle_labels(data)        # 0/1 labels for cycle membership
    #comp_labels = compute_component_size_labels(data)  # Connected component sizes (float tensor)
    
    # For degree prediction head, determine number of classes.
    num_degree_classes = int(degree_labels.max().item()) + 1
    
    # --- Compute graph-level labels ---
    #graph_node_count = torch.tensor([data.num_nodes], dtype=torch.float)
    #graph_edge_count = compute_graph_edge_count(data)  # a tensor of shape [1]
    
    # --- Generate edge (link prediction) data ---
    #edge_pairs, edge_labels = generate_link_prediction_data(data, neg_ratio=1.0)

    data['node_count'] = degree_labels
    #data['node_cycle'] = cycle_labels
    #data['node_component'] = comp_labels
    data['num_degree_classes'] = num_degree_classes
    #data['graph_node_count'] = graph_node_count
    #data['graph_edge_count'] = graph_edge_count
    #print("Graph Data:")
    #print(data.pos)
    #print(data)
    return data

def seq2graph(seq, feature_dim=32):
    nodes_match = re.search(r"among nodes\s*(.*?)\.", seq)
    if nodes_match:
        nodes_str = nodes_match.group(1)
        # Extract all numbers from the captured string.
        nodes = list(map(int, re.findall(r'\d+', nodes_str)))
        # For PyG, we set num_nodes to be the maximum node index + 1.
        num_nodes = max(nodes) + 1
    else:
        raise ValueError("Nodes information not found in the input string.")
    

    # --- Extract the edges ---
    # Find all occurrences of edge tuples in the form (u, v)
    edge_tuples = re.findall(r"\((\d+),\s*(\d+)\)", seq)
    edge_tuples = [(int(u), int(v)) for u, v in edge_tuples]

    # Build the list of edges.
    # Since the graph is undirected, add both directions (u,v) and (v,u).
    edge_list = []
    G = nx.Graph()
    G.add_edges_from(edge_tuples)
    for u, v in edge_tuples:
        u, v = int(u), int(v)
        edge_list.append((u, v))
        edge_list.append((v, u))

    # Convert the list to a tensor with shape [2, num_edges]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # --- Assign default weights ---
    # Give every edge a weight of 1.0.
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)

    # --- Add self-loops ---
    # For each node i, add an edge (i, i) with weight 1.0.
    edge_index, edge_weight = add_self_loops(
        edge_index, fill_value=1.0, num_nodes=num_nodes
    )

    # --- Create the PyG Data object ---
    x = torch.ones(num_nodes, feature_dim)  # constant feature for each node
    
    # Task specific graph-level labels:
    graph_node_count = torch.tensor([num_nodes], dtype=torch.long)
    graph_edge_count = torch.tensor([G.number_of_edges()], dtype=torch.long)
    
    # Node-level label: degree for each node
    node_degrees = [d for _, d in G.degree()]
    node_degree = torch.tensor(node_degrees, dtype=torch.long)
    
    # Cycle check: use cycle_basis (each cycle has length >= 3)
    cycles = nx.cycle_basis(G)
    cycle_check = torch.tensor([1 if len(cycles) > 0 else 0], dtype=torch.long)
    
    # Triangle count: each triangle is counted at each vertex, so divide sum by 3.
    triangles_dict = nx.triangles(G)
    triangle_count = int(sum(triangles_dict.values()) / 3)
    triangle_count = torch.tensor([triangle_count], dtype=torch.float)
    
    # Edge-level tasks: Compute for all unordered pairs (u,v) with u < v.
    edge_existence = {}
    connectivity = {}
    shortest_path = {}
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            u, v = nodes[i], nodes[j]
            # Edge existence: 1 if the edge exists, 0 otherwise.
            edge_existence[(u, v)] = 1 if G.has_edge(u, v) else 0
            # Connectivity and shortest path:
            if nx.has_path(G, u, v):
                connectivity[(u, v)] = 1
                sp_length = nx.shortest_path_length(G, source=u, target=v)
                shortest_path[(u, v)] = sp_length
            else:
                connectivity[(u, v)] = 0
                shortest_path[(u, v)] = 0

    # Convert edge-level dictionaries to tensors.
    if len(edge_existence) > 0:
        pairs = torch.tensor(list(edge_existence.keys()), dtype=torch.long)
        edge_existence_tensor = torch.tensor(list(edge_existence.values()), dtype=torch.long)
        connectivity_tensor = torch.tensor(list(connectivity.values()), dtype=torch.long)
        shortest_path_tensor = torch.tensor(list(shortest_path.values()), dtype=torch.long)
    else:
        pairs = torch.empty((0, 2), dtype=torch.long)
        edge_existence_tensor = torch.empty((0,), dtype=torch.long)
        connectivity_tensor = torch.empty((0,), dtype=torch.long)
        shortest_path_tensor = torch.empty((0,), dtype=torch.long)
        
    edge_pair_tasks = {
        "pairs": pairs,
        "edge_existence": edge_existence_tensor,
        "connectivity": connectivity_tensor,
        "shortest_path": shortest_path_tensor
    }
    
    # Create the PyG Data object and attach the labels.
    data = Data(x=x, edge_index=edge_index)
    data.node_count = graph_node_count
    data.edge_count = graph_edge_count
    data.node_degree = node_degree
    data.cycle_check = cycle_check
    data.triangle_count = triangle_count
    data.edge_pair_tasks = edge_pair_tasks
    data.edge_existence = edge_existence_tensor
    data.connectivity = connectivity_tensor
    data.shortest_path = shortest_path_tensor

    data.pos = torch.arange(0, 1, 1/num_nodes, dtype=torch.float)
    return data

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
    special_token_for_graphs = "<|reserved_special_token_0|>" # id: 128002

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
                return_tensors = self.return_tensors

        converted_batch = []
        batch_graphs = []; graph_sizes = []
        for instance in batch:
            if 'The edges in G are' in instance['input']:
                converted_batch.append(instance)
                batch_graphs.append(seq2graph(instance["input"]))
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
            source = "Given a graph: " # instance["input"].split("\n")[0]
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
        #batch_graphs.inputs = batch_graphs.inputs[0]
        model_inputs["graph_data"] = batch_graphs        

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

class AlgorithmGraphDataModule(pl.LightningDataModule):
    
    def __init__(
        self,
        task_names, 
        prompt_styles,
        text_encoders,
        node_range,
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
        self.text_encoders = text_encoders 
        self.min_nodes = node_range[0]
        self.max_nodes = node_range[1]
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
            text_encoder = self.text_encoders[i]

            # Split the dataset into train and validation
            task_file_dir = "data/tasks/nodes_{}_{}/{}_{}_er_train.json".format(self.min_nodes, self.max_nodes, task_name, prompt_style)
            train_dataset = load_dataset("json", data_files=task_file_dir)['train']
            
            # fileter out the examples by the text encoder
            column_names = train_dataset.column_names
            train_dataset = train_dataset.filter(lambda x: x["text_encoding"] == text_encoder)
            # convert the input and output format
            train_dataset = train_dataset.map(convert_format(), batched=True, remove_columns=column_names)

            task_file_dir = "data/tasks/nodes_{}_{}/{}_{}_er_valid.json".format(self.min_nodes, self.max_nodes, task_name, prompt_style)
            eval_dataset = load_dataset("json", data_files=task_file_dir)['train']
            # fileter out the examples by the text encoder
            column_names = eval_dataset.column_names
            eval_dataset = eval_dataset.filter(lambda x: x["text_encoding"] == text_encoder)
            # convert the input and output format
            eval_dataset = eval_dataset.map(convert_format(), batched=True, remove_columns=column_names)
            
            task_file_dir = "data/tasks/nodes_{}_{}/{}_{}_er_test.json".format(self.min_nodes, self.max_nodes, task_name, prompt_style)
            predict_dataset = load_dataset("json", data_files=task_file_dir)['train']
            # fileter out the examples by the text encoder
            column_names = predict_dataset.column_names
            predict_dataset = predict_dataset.filter(lambda x: x["text_encoding"] == text_encoder)
            # convert the input and output format
            predict_dataset = predict_dataset.map(convert_format(), batched=True, remove_columns=column_names)

            ''' Old Split '''
            # rng = np.random.default_rng(42)
            # permutations = rng.permutation(len(dataset))
            # train_size, eval_size, test_size = int(0.6*len(dataset)), int(0.2*len(dataset)), int(0.2*len(dataset)) 
            # train_dataset = dataset.select(permutations[:train_size])
            # eval_dataset = dataset.select(permutations[train_size:train_size+eval_size]) if not self.eval_all else dataset
            # predict_dataset = dataset.select(permutations[train_size+eval_size:])

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
        