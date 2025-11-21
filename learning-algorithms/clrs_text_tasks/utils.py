import os 
import os.path as osp
import torch
import clrs
import numpy as np
import networkx as nx

import lightning.pytorch as pl
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix

class CLRSData(Data):
    """A data object for CLRS data."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

def to_torch(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    elif isinstance(value, torch.Tensor):
        return value
    else:
        return torch.tensor(value)
    
def infer_type(dp_type, data):
    return data.astype(np.float32) # convert to float32

def pointer_to_one_hot(pointer, n):
    """Convert a pointer to a one-hot vector."""
    return (np.arange(n) == pointer.reshape(-1, 1)).astype(float)

def to_data(inputs, hints, outputs, use_hints=True, use_complete_graph=True):
    data_dict = {}
    input_attributes = []
    hint_attributes = []
    output_attributes = []
    data_dict['length'] = hints[0].data.shape[0]
    
    # first get the edge index; create a fully connected graph 
    input_keywords = [dp.name for dp in inputs]
    if "adj" in input_keywords and (not use_complete_graph):
        graph = nx.from_numpy_array(inputs[input_keywords.index("adj")].data[0], create_using=nx.DiGraph())
    else:
        graph = nx.complete_graph(hints[0].data.shape[-1], create_using=nx.DiGraph())
    edge_list = np.array(list(graph.edges())).T
    data_dict['edge_index'] = torch.tensor(edge_list, dtype=torch.long) # np.concatenate([edge_list, edge_list[::-1]], axis=1)

    # Parse inputs
    for dp in inputs:
        if not use_complete_graph:
            if dp.name == "adj":
                continue
            elif dp.name == "A":
                # add self loops
                unique_values = np.unique(dp.data[0])
                is_weighted = unique_values.size != 2 or not np.all(unique_values == np.array([0,1]))
                if is_weighted:
                    data_dict["weights"] = infer_type("A", (dp.data[0] + np.eye(dp.data[0].shape[0]))[data_dict["edge_index"][0], data_dict["edge_index"][1]])
            elif dp.location == clrs.Location.EDGE:
                data_dict[dp.name] = infer_type(dp.type_, dp.data[0][data_dict["edge_index"][0], data_dict["edge_index"][1]])
                input_attributes.append(dp.name)
            elif dp.location == clrs.Location.NODE:
                if dp.type_ == clrs.Type.POINTER:
                    # Convert pointers to one-hot edge masks
                    n = dp.data[0].shape[0]
                    pointer_matrix = pointer_to_one_hot(dp.data[0], n)
                    data_dict[dp.name] = pointer_matrix[data_dict["edge_index"][0], data_dict["edge_index"][1]]
                else:
                    data_dict[dp.name] = infer_type(dp.type_, dp.data[0])
                input_attributes.append(dp.name)
            else: # Graph
                data_dict[dp.name] = infer_type(dp.type_, dp.data[0])
        else:
            if dp.location == clrs.Location.EDGE:
                data_dict[dp.name] = infer_type(dp.type_, dp.data[0][data_dict["edge_index"][0], data_dict["edge_index"][1]])
                input_attributes.append(dp.name)
            elif dp.location == clrs.Location.NODE:
                if dp.type_ == clrs.Type.POINTER:
                    # Convert pointers to one-hot edge masks
                    n = dp.data[0].shape[0]
                    pointer_matrix = pointer_to_one_hot(dp.data[0], n)
                    data_dict[dp.name] = pointer_matrix[data_dict["edge_index"][0], data_dict["edge_index"][1]]
                else:
                    data_dict[dp.name] = infer_type(dp.type_, dp.data[0])
                input_attributes.append(dp.name)
            else: # Graph
                data_dict[dp.name] = infer_type(dp.type_, dp.data[0])
    # Parse outputs
    for dp in outputs:
        output_attributes.append(dp.name)
        if dp.location == clrs.Location.EDGE:
            data_dict[dp.name] = infer_type(dp.type_, dp.data[0][data_dict["edge_index"][0], data_dict["edge_index"][1]])
        elif dp.location == clrs.Location.NODE:
            if dp.type_ == clrs.Type.POINTER:
                # Convert pointers to one-hot edge masks
                n = dp.data[0].shape[0]
                pointer_matrix = pointer_to_one_hot(dp.data[0], n)
                data_dict[dp.name] = pointer_matrix[data_dict["edge_index"][0], data_dict["edge_index"][1]]
            else:
                data_dict[dp.name] = infer_type(dp.type_, dp.data[0])
        else: # Graph
            data_dict[dp.name] = infer_type(dp.type_, dp.data[0])
    if use_hints:
        # Parse hints
        for dp in hints:
            hint_attributes.append(dp.name)
            if dp.location == clrs.Location.EDGE or (dp.location == clrs.Location.NODE and dp.type_ == clrs.Type.POINTER):
                arr = dp.data.squeeze(1) # Hints, N, N, D (...)
                if dp.location == clrs.Location.NODE:
                    # arr is Hints, N, D (...)
                    # Convert pointers to one-hot edge masks
                    stages = []
                    for hd in range(arr.shape[0]):
                        n = arr.shape[1]
                        pointer_matrix = pointer_to_one_hot(arr[hd], n)
                        stages.append(pointer_matrix)
                    
                    arr = np.stack(stages, axis=0)
                
                # Determine the number of dimensions of the array
                num_dims = arr.ndim
                transpose_indices = tuple(range(num_dims))
                transpose_indices = (1, 2, 0) + transpose_indices[3:]
                data_dict[dp.name] = infer_type(dp.type_, arr.transpose(*transpose_indices)[data_dict["edge_index"][0], data_dict["edge_index"][1]])
            elif dp.location == clrs.Location.NODE and not dp.type_ == clrs.Type.POINTER:
                arr = dp.data.squeeze(1) # Hints, N, D (...)
                # Determine the number of dimensions of the array
                num_dims = arr.ndim
                # Create a tuple of indices to swap the first two dimensions
                transpose_indices = tuple(range(num_dims))
                transpose_indices = (1, 0) + transpose_indices[2:]
                data_dict[dp.name] = infer_type(dp.type_, arr.transpose(*transpose_indices))
            else:
                data_dict[dp.name] = infer_type(dp.type_, dp.data.squeeze(1)[np.newaxis, ...])

        
    data_dict = {k: to_torch(v) for k,v in data_dict.items()}
    data = CLRSData(**data_dict)    
    data.hints = hint_attributes
    data.inputs = input_attributes
    data.outputs = output_attributes
    return data

class CLRSDataset(Dataset):
    def __init__(self, root="./data/CLRS/", algorithm="insertion_sort", split="train",
                 num_samples = 1000, hints=True, ignore_all_hints=False, nickname=None, use_complete_graph=False,
                 **kwargs):
        """ Dataset for CLRS problems.

        Args:
            root (str): Root directory where the dataset should be saved.
            split (str): Split of the dataset to use. One of ['train', 'val', 'test'].
            algorithm (str): Algorithm to use. Check salsa-clrs.ALGORITHMS for a list of available algorithms.
            num_samples (int): Number of samples to collect.
            hints (bool): Whether to use hints or not (hints are still loaded but not returned in the data dict)
            ignore_all_hints (bool): Whether to ignore all hints or not. If True, hints are not even generated, might be beneficial for memory.
            nickname (str): Optional nickname for the dataset (mainly intended for logging purposes).
            graph_generator (str): Name of the graph generator to use. 
            graph_generator_kwargs (dict): Keyword arguments to pass to the graph generator.
            max_cores (int): Maximum number of cores to use for multiprocessing. If -1, it is serial. If None, it is the number of cores on the machine (default: -1)
            **kwargs: Keyword arguments to pass to the algorithm sampler.
        """
        self.data_dir = osp.join(root, "clrs_dataset", f"{algorithm}_{split}", "1.0.0")
        if not os.path.exists(self.data_dir):
            raise NotImplementedError(f"Data not found at {self.data_dir}. Please download the data.")
        self.root = root
        self.algorithm = algorithm
        self.split = split
        self.num_samples = num_samples

        self.specs = {}

        self.hints = hints
        self.ignore_all_hints = ignore_all_hints
        self.nickname = nickname
        self.use_complete_graph = use_complete_graph
        
        super().__init__(root, None, None, None)
        
        _, _, specs = clrs.create_dataset(
            folder=self.data_dir, algorithm=self.algorithm,
            split=self.split, batch_size=1)
        self.specs = specs
        self._update_specs()

    @property
    def raw_file_names(self):
        return [osp.join(self.data_dir, f"clrs_dataset-{self.split}.tfrecord-00000-of-00001")]

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(self.num_samples)]
    
    @property
    def raw_dir(self) -> str:
        return self.data_dir

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f'processed_{self.split}', f"{self.algorithm}" + ("_complete" if self.use_complete_graph else ""))

    def _update_specs(self):
        # get a batch
        batch = self.get(0)
        specs = {}
        for key, data in self.specs.items():
            if key not in batch:
                continue
            stage, location, type_ = data
            if type_ == clrs.Type.CATEGORICAL:
                specs[key] = (stage, location, clrs.Type.CATEGORICAL, batch[key].shape[-1])
            else:
                specs[key] = (stage, location, type_, None)
        self.specs = specs

    def process(self):
        train_ds, num_samples, specs = clrs.create_dataset(
            folder=self.data_dir, algorithm=self.algorithm,
            split=self.split, batch_size=1)
        self.specs = specs
        
        root_dir = f"./data/CLRS/processed_{self.split}"
        processed_dir = osp.join(root_dir, f"{self.algorithm}" + ("_complete" if self.use_complete_graph else ""))
        if not osp.exists(processed_dir):
            os.makedirs(processed_dir)

        for i, feedback in enumerate(train_ds.as_numpy_iterator()):
            if i >= num_samples:
                break
            features = feedback.features
            outputs = feedback.outputs
            inputs = features.inputs # inputs "key" and "pos"
            hints = features.hints
            lengths = features.lengths
            
            data = to_data(inputs, hints, outputs, use_complete_graph = self.use_complete_graph)
            torch.save(data, osp.join(processed_dir, f'data_{i}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        if not self.hints and not self.ignore_all_hints:
            for hint in data.hints:
                delattr(data, hint)
            del data.hints
        return data


class CLRSCollater(object):
    """Special Collater that can handle hints. """
    def __init__(self, follow_batch=None, exclude_keys=None):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def normalise_length(self, batch):
        """Normalise the length of the batch by padding with zeros."""
        max_len = max([data.length for data in batch])
        for data in batch:
            if data.length < max_len:
                # pad all hints
                for hint in data.hints:
                    data[hint] = torch.cat([data[hint], torch.zeros(*data[hint].shape[:1], max_len - data[hint].shape[1], *data[hint].shape[2:])], dim=1)
                # pad randomness
                if "randomness" in data.inputs:
                    data["randomness"] = torch.cat([data["randomness"], torch.zeros(*data["randomness"].shape[:1], max_len - data["randomness"].shape[1], *data["randomness"].shape[2:])], dim=1)
        return batch
    
    def collate(self, batch):
        if "hints" in batch[0].keys() or "randomness" in batch[0].inputs:
            batch = self.normalise_length(batch)
        batch = Batch.from_data_list(batch, self.follow_batch,
                                        self.exclude_keys)
        batch.hints = batch.hints[0]
        batch.inputs = batch.inputs[0]
        batch.outputs = batch.outputs[0]
        return batch

    def __call__(self, batch):
        return self.collate(batch)