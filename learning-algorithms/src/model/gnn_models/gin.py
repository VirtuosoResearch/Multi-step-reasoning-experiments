import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data
from torch_geometric.utils import degree
import networkx as nx
import numpy as np
import scipy.sparse as sp

TASK_LIST = [
    "node_count", "edge_count", "edge_existence", "node_degree",
    "connectivity", "cycle_check", "shortest_path", "triangle_count"
]

TASK_MAP = {
    'edge_existence': 'edge_existence',
    'node_degree': 'node_degree',
    'node_count': 'node_count',
    'edge_count': 'edge_count',
    'connected_nodes': 'connectivity',
    'cycle_check': 'cycle_check',
    'disconnected_nodes': 'connectivity',
    'reachability': 'connectivity',
    'shortest_path': 'shortest_path',
    'maximum_flow': 'connectivity',
    'triangle_counting': 'triangle_count',
    'node_classification': 'node_degree'
}

    


class GINEncoder(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers, task_names):
        super(GINEncoder, self).__init__()
        # Use ModuleDicts so that each task branch is registered.
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            conv_layers = nn.ModuleDict()
            bn_layers = nn.ModuleDict()
            for task in task_names:
                if i == 0:
                    mlp = nn.Sequential(
                        nn.Linear(num_features, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
                else:
                    mlp = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
                conv_layers[task] = GINConv(mlp)
                bn_layers[task] = nn.BatchNorm1d(hidden_dim)
            self.convs.append(conv_layers)
            self.bns.append(bn_layers)
        print(self.convs)

    def forward(self, x, edge_index, task_name='node_count'):
        x = x.to(torch.bfloat16)
        for conv, bn in zip(self.convs, self.bns):
            
            x = conv[task_name](x, edge_index)
            x = bn[task_name](x)
            x = F.relu(x)
        return x

# =============================================================================
# MultiTaskHeads: A single module that holds heads for every task.
# In our final multi-task model, each task will have its own branch (keyed in a dict).
# =============================================================================
class MultiTaskHeads(nn.Module):
    def __init__(self, hidden_dim, node_count_class=40):
        super(MultiTaskHeads, self).__init__()
        self.node_count_class = node_count_class
        self.edge_count_class = node_count_class **2
        self.branches = nn.ModuleDict({
            # Graph-level tasks:
            #"node_count": nn.Linear(hidden_dim, 1),
            #"edge_count": nn.Linear(hidden_dim, 1),
            "node_count": nn.Linear(hidden_dim, self.node_count_class),
            "edge_count": nn.Linear(hidden_dim, self.edge_count_class),
            "cycle_check": nn.Linear(hidden_dim, 2),
            "triangle_count": nn.Linear(hidden_dim, 1),
            # Node-level:
            #"node_degree": nn.Linear(hidden_dim, 1),
            "node_degree": nn.Linear(hidden_dim, self.node_count_class),
            # Edge-level (all these heads take concatenated embeddings of two nodes):
            "edge_existence": nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)
            ),
            "connectivity": nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)
            ),
            "shortest_path": nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.node_count_class)
            )
        })

    def forward(self, node_emb, batch_index, edge_pair_tasks=None):
        outputs = {}
        # Graph-level predictions: pool the node embeddings.
        pooled = global_add_pool(node_emb, batch_index)  # [batch_size, hidden_dim]
        outputs["node_count"] = self.branches["node_count"](pooled).squeeze(-1)
        outputs["edge_count"] = self.branches["edge_count"](pooled).squeeze(-1)
        outputs["cycle_check"] = self.branches["cycle_check"](pooled)
        outputs["triangle_count"] = self.branches["triangle_count"](pooled).squeeze(-1)
        # Node-level prediction:
        outputs["node_degree"] = self.branches["node_degree"](node_emb).squeeze(-1)
        # Edge-level predictions:
        if edge_pair_tasks is not None and edge_pair_tasks["pairs"].size(0) > 0:
            pairs = edge_pair_tasks["pairs"]  # [num_pairs, 2]
            u = node_emb[pairs[:, 0]]
            v = node_emb[pairs[:, 1]]
            pair_emb = torch.cat([u, v], dim=1)
            outputs["edge_existence"] = self.branches["edge_existence"](pair_emb).squeeze(-1)
            outputs["connectivity"] = self.branches["connectivity"](pair_emb).squeeze(-1)
            outputs["shortest_path"] = self.branches["shortest_path"](pair_emb).squeeze(-1)
        else:
            outputs["edge_existence"] = None
            outputs["connectivity"] = None
            outputs["shortest_path"] = None
        return outputs

# =============================================================================
# Full MultiTask Model: It holds an encoder and a heads module.
# In our final grouped models, each branch of the heads will be used.
# =============================================================================
class MultiTaskGIN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers, task_names):
        super(MultiTaskGIN, self).__init__()
        self.task_names = task_names
        self.encoder = GINEncoder(num_features, hidden_dim, num_layers, task_names)
        self.heads = MultiTaskHeads(hidden_dim)

    def forward(self, data, task_name, is_pretrain=False):
        if not is_pretrain:
            task_name = TASK_MAP[task_name]
        # task_branch is a string that indicates which encoder branch to use.
        x, edge_index = data.x, data.edge_index
        # If using batching, data.batch indicates the graph id for each node.
        batch_index = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long)
        # Use the specified branch for encoding.
        node_emb = self.encoder(x, edge_index, task_name)
        outputs = self.heads(node_emb, batch_index, 
                             edge_pair_tasks=data.edge_pair_tasks if hasattr(data, 'edge_pair_tasks') else None)
        return outputs

    def encode(self, data, task_name):
        task_name = TASK_MAP[task_name]
        return self.encoder(data.x, data.edge_index, task_name)