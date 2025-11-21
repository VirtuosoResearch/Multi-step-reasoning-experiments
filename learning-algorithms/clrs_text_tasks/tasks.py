
import random
import networkx as nx
import numpy as np
from clrs_text_tasks import graph_text_encoder


class GraphTask:
	"""The parent class for all the graph tasks."""
	answer_prompt = "Answer: {answer}"
	chain_of_thought_promt = "Steps: \n{steps}"

	def __init__(self):
		self.name = 'default'
		self.problem_description = 'default'
		self.algorithm_description = 'default'
		self.question_prompt = 'default'
		self.answer_prompt = 'default'
		self.chain_of_thought_promt = 'default'

	def prepare_examples(
		self,
		dataset,
		encoding_method,
		add_description,
		add_cot,
	):
		raise NotImplementedError()
	
	def prepare_few_shot_examples(
		self, dataset, encoding_method="incident", num_samples=5, add_cot=False, random_seed=42): 
		"""Create a recordio file with few-shot examples for the task."""
		examples = self.prepare_examples(dataset, encoding_method, add_description=False, add_cot=add_cot)
		no_cot_examples = self.prepare_examples(dataset, encoding_method, add_description=False, add_cot=False)
		few_shot_example_list = [example['question'] + example['answer'] for example in examples]
		random.seed(random_seed)

		for i, example in enumerate(examples):
			# sample few shot string
			few_shots_str = 'Example:\n'
			for _ in range(num_samples):
				few_shots_str += random.choice(few_shot_example_list) + '\n'
			
			# Revise the example question
			if add_cot:
				examples[i]['answer'] = examples[i]['question'][examples[i]['question'].index("Steps:"): ] + examples[i]['answer']
			examples[i]['question'] = self.problem_description + "\n" + self.algorithm_description + "\n" + \
				few_shots_str + (no_cot_examples[i]['question'][:-9] if add_cot else no_cot_examples[i]['question'])
			
		return examples

class BellmanFordTask(GraphTask):
	"""The Bellman-Ford task."""

	def __init__(self):
		self.name = 'bellman_ford'
		self.problem_description = "In a single-source shortest-paths problem, we are given a weighted directed graph and a source node. We aim to find the shortest path starting from source node to every other node. For each node, each node has a pointer to its predecessor along the optimal path from the source. By convention, the source node points to itself. "
		self.algorithm_description = "The Bellman-Ford algorithm starts by initializing the distance to the source node as 0 and all other vertices as infinity. It then iteratively relaxes each edge in the graph V-1 times where V is the number of vertices. Relaxing an edge involves checking whether the distance to the destination node can be reduced by taking the edge and, if so, updating the distance and the predecessor. After V-1 iterations, the algorithm performs an additional pass to check for negative-weight cycles. If any edge can still be relaxed, it indicates the presence of a negative-weight cycle, which is reported. Otherwise, the shortest paths from the source to all vertices are finalized."

		self.question_prompt = "Question: Run the algorithm and return the predecessor node of all nodes in the shortest path to node {source} in alphabetical order."
		self.answer_prompt = "{answer}"
		self.chain_of_thought_promt = "Steps: \n{steps}"
	
	def get_answer(self, data):
		answer = list(data.edge_index.T[data.pi==1])
		answer = sorted(answer, key=lambda x: x[0])
		answer = [str(x[1].item()) for x in answer]
		answer = ", ".join(answer)
		return answer

	def get_intermediate_results(self, data):
		hints = data.pi_h
		intermediate_results = []
		for i in range(hints.shape[1]):
			answer = list(data.edge_index.T[data.pi_h[:, i]==1])
			if len(answer) < data.num_nodes: break
			answer = sorted(answer, key=lambda x: x[0])
			answer = [str(x[1].item()) for x in answer]
			answer = ", ".join(answer)
			intermediate_results.append(answer)
		return intermediate_results
	
	def prepare_examples(
		self,
		dataset,
		encoding_method="incident",
		add_description=True,
		add_cot=False,
	):
		examples = []
		name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
		for ind, data in enumerate(dataset):
			graph = nx.from_edgelist(data.edge_index.T.numpy())
			edge_weights = {tuple(edge): weight for edge, weight in zip(data.edge_index.T.numpy(), data.weights.numpy())}
			nx.set_edge_attributes(graph, values = edge_weights, name = 'weight')
			
			graph_encoding = graph_text_encoder.encode_graph(graph, encoding_method, add_weights=True)
			source = data.s.nonzero().item()
			source = name_dict[source]
			question = (self.problem_description + "\n" + self.algorithm_description + "\n" if add_description else "")
			question += graph_encoding + self.question_prompt.format(source=source) 
			if add_cot:
				question += "\n" + self.chain_of_thought_promt.format(steps="\n".join(self.get_intermediate_results(data)))
			question += "\nAnswer: "
			answer = self.get_answer(data)
			examples.append({
				'id': str(ind),
				'question': question,
				'answer': answer,
			})
		return examples

class BFSTask(GraphTask):
	"""The breadth-first search task."""

	def __init__(self):
		self.name = 'bfs'
		self.problem_description = "In the graph search problem, we are given a undirected graph and a source node. We aim to explore the edges of the graph to discover every node that is reachable from the source node. For each node, each node has a pointer to its predecessor along the search procedure from the source. By convention, the source node points to itself."
		self.algorithm_description = "The breadth-first search algorithm explores a graph level by level, starting from a specified source node. It begins by marking the source node as visited and enqueues it. The algorithm then repeatedly dequeues a node, processes it, and enqueues all its unvisited neighbors, marking them as visited to avoid reprocessing. This ensures that nodes are visited in increasing order of their distance from the source."

		self.question_prompt = "Question: Run the algorithm and return the predecessor node of all nodes of the breadth-first traversal starting from node {source} in alphabetical order."
		self.answer_prompt = "{answer}"
		self.chain_of_thought_promt = "Steps: \n{steps}"
	
	def get_answer(self, data):
		answer = list(data.edge_index.T[data.pi==1])
		answer = sorted(answer, key=lambda x: x[0])
		answer = [str(x[1].item()) for x in answer]
		answer = ", ".join(answer)
		return answer

	def get_intermediate_results(self, data):
		hints = data.pi_h
		intermediate_results = []
		for i in range(hints.shape[1]):
			answer = list(data.edge_index.T[data.pi_h[:, i]==1])
			if len(answer) < data.num_nodes: break
			answer = sorted(answer, key=lambda x: x[0])
			answer = [str(x[1].item()) for x in answer]
			answer = ", ".join(answer)
			intermediate_results.append(answer)
		return intermediate_results
	
	def prepare_examples(
		self,
		dataset,
		encoding_method="incident",
		add_description=True,
		add_cot=False,
	):
		examples = []
		name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
		for ind, data in enumerate(dataset):
			graph = nx.from_edgelist(data.edge_index.T.numpy())
			graph_encoding = graph_text_encoder.encode_graph(graph, encoding_method, add_weights=False)

			source = data.s.nonzero().item()
			source = name_dict[source]
			question = (self.problem_description + "\n" + self.algorithm_description + "\n" if add_description else "")
			question += graph_encoding + self.question_prompt.format(source=source) 
			if add_cot:
				question += "\n" + self.chain_of_thought_promt.format(steps="\n".join(self.get_intermediate_results(data)))
			question += "\nAnswer: "
			answer = self.get_answer(data)
			examples.append({
				'id': str(ind),
				'question': question,
				'answer': answer,
			})
		return examples
	
class DFSTask(GraphTask):
	"""The breadth-first search task."""

	def __init__(self):
		self.name = 'dfs'
		self.problem_description = "In the graph search problem, we are given a directed graph and a source node. We aim to explore the edges of the graph to discover every node that is reachable from the source node. For each node, each node has a pointer to its predecessor along the search procedure from the source. By convention, the source node points to itself."
		self.algorithm_description = "The depth-first search algorithm searches deeper in the graph whenever possible. Depth-first search explores edges out of the most recently discovered node v that still has unexplored edges leaving it. Once all of v’s edges have been explored, the search backtracks to explore edges leaving the node from which was discovered. This process continues until we have discovered all the nodes that are reachable from the original source node. If any undiscovered nodes remain, then depth-first search selects one of them as a new source, and it repeats the search from that source. The algorithm repeats this entire process until it has discovered every node."

		self.question_prompt = "Question: Run the algorithm and return the predecessor node of all nodes of the depth-first traversal starting from node {source} in alphabetical order."
		self.answer_prompt = "{answer}"
		self.chain_of_thought_promt = "Steps: \n{steps}"
	
	def get_answer(self, data):
		answer = list(data.edge_index.T[data.pi==1])
		answer = sorted(answer, key=lambda x: x[0])
		answer = [str(x[1].item()) for x in answer]
		answer = ", ".join(answer)
		return answer

	def get_intermediate_results(self, data):
		hints = data.pi_h
		intermediate_results = []
		for i in range(hints.shape[1]):
			answer = list(data.edge_index.T[data.pi_h[:, i]==1])
			if len(answer) < data.num_nodes: break
			answer = sorted(answer, key=lambda x: x[0])
			answer = [str(x[1].item()) for x in answer]
			answer = ", ".join(answer)
			intermediate_results.append(answer)
		return intermediate_results
	
	def prepare_examples(
		self,
		dataset,
		encoding_method="incident",
		add_description=True,
		add_cot=False,
	):
		examples = []
		name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
		for ind, data in enumerate(dataset):
			graph = nx.from_edgelist(data.edge_index.T.numpy())
			graph_encoding = graph_text_encoder.encode_graph(graph, encoding_method, add_weights=False)

			question = (self.problem_description + "\n" + self.algorithm_description + "\n" if add_description else "")
			question += graph_encoding + self.question_prompt
			if add_cot:
				question += "\n" + self.chain_of_thought_promt.format(steps="\n".join(self.get_intermediate_results(data)))
			question += "\nAnswer: "
			answer = self.get_answer(data)
			examples.append({
				'id': str(ind),
				'question': question,
				'answer': answer,
			})
		return examples 
	
class TopologicalSortTask(GraphTask):
	"""The topological sort task."""

	def __init__(self):
		self.name = 'topological_sort'
		self.problem_description = "In the topological sort problem, we are given a directed acyclic graph G. We aim to find a linear ordering of all its nodes such that if G contains an edge (u, v) then u appears before v in the ordering. For each node, each node has a pointer to its predecessor along the ordering."
		self.algorithm_description = "The topological sort algorithm typically uses the depth-first search algorithm. In the depth-first search, nodes are recursively explored, and once all descendants of a node are visited, the node is added to a stack or list, ensuring a reverse post-order. Once all nodes have been visited, the stack or list contains the topological ordering in reverse order, which can be output directly."

		self.question_prompt = "Question: Run the algorithm and return the predecessor node of all nodes of the topological sort ordering in alphabetical order."
		self.answer_prompt = "{answer}"
		self.chain_of_thought_promt = "Steps: \n{steps}"
	
	def get_answer(self, data):
		answer = list(data.edge_index.T[data.topo==1])
		answer = sorted(answer, key=lambda x: x[0])
		answer = [str(x[1].item()) for x in answer]
		answer = ", ".join(answer)
		return answer

	def get_intermediate_results(self, data):
		hints = data.topo_h
		intermediate_results = []
		for i in range(hints.shape[1]):
			answer = list(data.edge_index.T[data.topo_h[:, i]==1])
			if len(answer) <=4: break
			answer = sorted(answer, key=lambda x: x[0])
			answer = [str(x[1].item()) for x in answer]
			answer = ", ".join(answer)
			intermediate_results.append(answer)
		return intermediate_results
	
	def prepare_examples(
		self,
		dataset,
		encoding_method="incident",
		add_description=True,
		add_cot=False,
	):
		examples = []
		name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
		for ind, data in enumerate(dataset):
			graph = nx.from_edgelist(data.edge_index.T.numpy())
			graph_encoding = graph_text_encoder.encode_graph(graph, encoding_method, add_weights=False)

			question = (self.problem_description + "\n" + self.algorithm_description + "\n" if add_description else "")
			question += graph_encoding + self.question_prompt
			if add_cot:
				question += "\n" + self.chain_of_thought_promt.format(steps="\n".join(self.get_intermediate_results(data)))
			question += "\nAnswer: "
			answer = self.get_answer(data)
			examples.append({
				'id': str(ind),
				'question': question,
				'answer': answer,
			})
		return examples 
	
class SCCTask(GraphTask):
	"""The strongly connected component task."""

	def __init__(self):
		self.name = 'strongly_connected_component'
		self.problem_description = "In the strongly connected component problem, we are given a directed graph G. We aim to decompose the graph into its strongly connected components. A strongly connected component of G is a maximal set of nodes such in which every pair of nodes are connected. Each node has a pointer to indicate the component it belongs to." 
		self.algorithm_description = "The strongly connected components algorithm involves two depth-first search traversals. In the depth-first search (DFS), nodes are recursively explored, and once all descendants of a node are visited, the node is added to a stack or list, ensuring a reverse post-order. First, a DFS is performed on the original graph to compute a finishing order of nodes, stored in a stack. Next, the graph is transposed (all edges are reversed). A second DFS is then performed on the transposed graph, following the vertices in the order of the stack. During this second DFS, each traversal identifies all vertices reachable from a given starting vertex, forming a strongly connected component. Repeating this process for all vertices in the stack ensures that all SCCs are identified."

		self.question_prompt = "Question: Run the algorithm and return the pointer of all nodes to the strongly connected component it belongs to."
		self.answer_prompt = "{answer}"
		self.chain_of_thought_promt = "Steps: \n{steps}"
	
	def get_answer(self, data):
		answer = list(data.edge_index.T[data.scc_id==1])
		answer = sorted(answer, key=lambda x: x[0])
		answer = [str(x[1].item()) for x in answer]
		answer = ", ".join(answer)
		return answer

	def get_intermediate_results(self, data):
		hints = data.scc_id_h
		intermediate_results = []
		for i in range(hints.shape[1]):
			answer = list(data.edge_index.T[data.scc_id_h[:, i]==1])
			if len(answer) < data.num_nodes: break
			answer = sorted(answer, key=lambda x: x[0])
			answer = [str(x[1].item()) for x in answer]
			answer = ", ".join(answer)
			intermediate_results.append(answer)
		return intermediate_results
	
	def prepare_examples(
		self,
		dataset,
		encoding_method="incident",
		add_description=True,
		add_cot=False,
	):
		examples = []
		name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
		for ind, data in enumerate(dataset):
			graph = nx.from_edgelist(data.edge_index.T.numpy())
			graph_encoding = graph_text_encoder.encode_graph(graph, encoding_method, add_weights=False)

			question = (self.problem_description + "\n" + self.algorithm_description + "\n" if add_description else "")
			question += graph_encoding + self.question_prompt
			if add_cot:
				question += "\n" + self.chain_of_thought_promt.format(steps="\n".join(self.get_intermediate_results(data)))
			question += "\nAnswer: "
			answer = self.get_answer(data)
			examples.append({
				'id': str(ind),
				'question': question,
				'answer': answer,
			})
		return examples 