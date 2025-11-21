import re
import torch
import numpy as np

def extract_target_node(text):
    """
    Extract the target node ID from a question like 'Q: What is the degree of node 12?'
    Returns an integer node ID if found, otherwise raises ValueError.
    """
    match = re.search(r'What\sis\sthe\sdegree\sof\snode\s+(\d+)', text)
    if match:
        return int(match.group(1))
    else:
        return -1


def extract_edges(text):
    # Extract edge tuples from the format "(i, j)"
    edge_tuples = re.findall(r'\((\d+),\s*(\d+)\)', text)
    edges = [(int(i), int(j)) for i, j in edge_tuples]
    return edges

def get_one_hop_subgraph_spans(text, target_node, include_prefix=True):
    edges = extract_edges(text)
    neighbors = {j if i == target_node else i for i, j in edges if target_node in (i, j)}
    relevant_nodes = neighbors.union({target_node})

    subgraph_spans = []

    # Include edge spans involving the target node
    for match in re.finditer(r'\((\d+),\s*(\d+)\)', text):
        i, j = int(match.group(1)), int(match.group(2))
        if i == target_node or j == target_node:
            subgraph_spans.append((match.start(), match.end()))

    # Include individual node mentions
    for node in relevant_nodes:
        for match in re.finditer(rf',\s\b{node}\b,', text):
            subgraph_spans.append((match.start()+2, match.end()-1))

    # Include the introductory explanation sentence
    if include_prefix:
        prefix = "In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge. G describes a graph among nodes"
        prefix_start = text.find(prefix)
        if prefix_start != -1:
            subgraph_spans.append((prefix_start, prefix_start + len(prefix)))

    return subgraph_spans

def create_token_mask_node_degree(text, tokenizer, target_node, max_length=512):
    subgraph_spans = get_one_hop_subgraph_spans(text, target_node)

    # Tokenize with offset mapping to map tokens to character spans
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False, max_length=max_length, truncation=True, padding='max_length')
    offsets = encoding["offset_mapping"]
    token_mask = []

    for start, end in offsets:
        # Check if this token span overlaps any subgraph span
        overlaps = any(not (end <= s or start >= e) for s, e in subgraph_spans)
        token_mask.append(1 if overlaps else 0)

    return np.array(token_mask, dtype=np.int64)
# tokenizer.convert_ids_to_tokens(encoding["input_ids"])
