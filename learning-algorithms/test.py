# %%
import graph_task as graph_task
import graph_task_utils as utils


GRAPHS_DIR = "../data/graphs"
TASK_DIR="../data/tasks"
TASK = "edge_existence"
algorithms = ['er']
text_encoders = [
    'adjacency',
    'incident',
    'coauthorship',
    'friendship',
    'south_park',
    'got',
    'social_network',
    'politician',
    'expert',
]


TASK_CLASS = {
    'edge_existence': graph_task.EdgeExistence,
    'node_degree': graph_task.NodeDegree,
    'node_count': graph_task.NodeCount,
    'edge_count': graph_task.EdgeCount,
    'connected_nodes': graph_task.ConnectedNodes,
    'cycle_check': graph_task.CycleCheck,
    'disconnected_nodes': graph_task.DisconnectedNodes,
    'reachability': graph_task.Reachability,
    'shortest_path': graph_task.ShortestPath,
    'maximum_flow': graph_task.MaximumFlow,
    'triangle_counting': graph_task.TriangleCounting,
    'node_classification': graph_task.NodeClassification,
}

# Loading the graphs.
graphs = []
generator_algorithms = []
for algorithm in algorithms:
    loaded_graphs = utils.load_graphs(
        GRAPHS_DIR,
        algorithm,
        'test',
    )
    graphs += loaded_graphs
    generator_algorithms += [algorithm] * len(loaded_graphs)

# Defining a task on the graphs
task = TASK_CLASS[TASK]()

# %%

# change this to not using tensorflow 
def create_example_feature(
    key,
    question,
    answer,
    algorithm,
    encoding_method,
    nnodes,
    nedges,
):
  """Create a tensorflow example from a datapoint."""
  key_feature = str(key)
  question_feature = question
  answer_feature = answer
  algorithm_feature = algorithm
  encoding_method_feature = encoding_method
  nnodes_feature = value=[nnodes]
  nedges_feature = value=[nedges]
  example_feats = {'id': key_feature,
          'question': question_feature,
          'answer': answer_feature,
          'algorithm': algorithm_feature,
          'text_encoding': encoding_method_feature,
          'nnodes': nnodes_feature,
          'nedges': nedges_feature,
      }
  return example_feats


def prepare_examples(
    examples_dict,
    encoding_method,
):
  """Create a list of tf.train.Example from a dict of examples."""
  examples = []
  for key, value in examples_dict.items():
    (
        question,
        answer,
        nnodes,
        nedges,
        algorithm,
    ) = (
        value['question'],
        value['answer'],
        value['nnodes'],
        value['nedges'],
        value['algorithm'],
    )
    examples.append(
        create_example_feature(
            key,
            question,
            answer,
            algorithm,
            encoding_method,
            nnodes,
            nedges,
        )
    )
  return examples


def create_zero_shot_task(
    task,
    graphs,
    generator_algorithms,
    text_encoders,
    cot = False,
):
  """Create a recordio file with zero-shot examples for the task."""
  examples = []
  for encoding_method in text_encoders:
    examples_dict = task.prepare_examples_dict(
        graphs, generator_algorithms, encoding_method
    )
    if cot:
      for key in examples_dict.keys():
        examples_dict[key]['question'] += "Let's think step by step. "
    examples += prepare_examples(examples_dict, encoding_method) # TODO: change this
  return examples

split='test'
cot = False
zero_shot_examples = create_zero_shot_task(
    task, graphs, generator_algorithms, text_encoders, cot=False
)

# %%
import json
file_name = task.name + ('_zero_cot_' if cot else '_zero_shot_')
file_name += split

def write_examples(examples, output_path):
    with open(output_path + ".json", 'w') as file:
        for example in examples:
            json.dump(example, file)

if not os.path.exists(TASK_DIR):
    os.makedirs(TASK_DIR)

write_examples(
    zero_shot_examples,
    os.path.join(TASK_DIR, file_name),
)

# %%
from datasets import load_dataset

dataset = load_dataset("json", data_files="./data/tasks/nodes_80_100/edge_existence_zero_shot_er_test.json")['train']
# for i in range(100):
#     # print(dataset[i]['question'])
#     print(dataset[i]['nnodes'], dataset[i]['nedges'], dataset[i]['answer'])

# %%
from transformers import AutoTokenizer
from src.custom.algorithm_task_data_module import AlgorithmDataModule

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token_id = tokenizer.eos_token_id

lens = []
for i in range(1000):
    lens.append(len(tokenizer(dataset[i]['question'])['input_ids']))
    print(lens[-1])
print(max(lens), min(lens), sum(lens) / len(lens))

# %%
data_module = AlgorithmDataModule(
    task_names=["node_degree"],
    prompt_styles=["zero_shot"],
    text_encoders=["adjacency"],
    tokenizer=tokenizer,
)
data_module.setup()
# %%
data_module.task_to_train_datasets['node_degree_zero_shot'][0]
