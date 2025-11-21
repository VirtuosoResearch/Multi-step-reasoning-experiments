# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -e
set -x

# python3 -m venv graphqa
# source graphqa/bin/activate

# pip3 install -r graphqa/requirements.txt

# Fill in appropriate output path
OUTPUT_PATH="./data/graphs"

echo "The output path is set to: $OUTPUT_PATH"

# for node in 15 20 # 40 60 80
# do
# for algorithm in "ba" "sbm" "sfn" # "complete" "star" "path" #  "er" #
# do
#   echo "Generating test examples for $algorithm"
#   python3 -m graph_tasks.graph_generator \
#                     --algorithm=$algorithm \
#                     --number_of_graphs=10000 \
#                     --split=train \
#                     --output_path=$OUTPUT_PATH \
#                     --min_nodes $node --max_nodes $((node+1))
# done

# for algorithm in "ba" "sbm" "sfn" # "complete" "star" "path" # 
# do
#   echo "Generating test examples for $algorithm"
#   python3 -m graph_tasks.graph_generator \
#                     --algorithm=$algorithm \
#                     --number_of_graphs=500 \
#                     --split=test \
#                     --output_path=$OUTPUT_PATH\
#                     --min_nodes $node --max_nodes $((node+1))
# done


# for algorithm in "ba" "sbm" "sfn" # "complete" "star" "path" # 
# do
#   echo "Generating test examples for $algorithm"
#   python3 -m graph_tasks.graph_generator \
#                     --algorithm=$algorithm \
#                     --number_of_graphs=500 \
#                     --split=valid \
#                     --output_path=$OUTPUT_PATH \
#                     --min_nodes $node --max_nodes $((node+1))
# done
# done


# Fill in appropriate output path
GRAPHS_DIR="./data/graphs"
TASK_DIR="./data/tasks"
# TASKS=("edge_existence" "node_degree" "node_count" "edge_count" "connected_nodes" "cycle_check" "disconnected_nodes" "reachability" "shortest_path" "maximum_flow" "triangle_counting" "node_classification")
TASKS=("node_degree")

# For experimenting with only erdos-reyni graph use `er``.
# For all graph generators, set to `all`.
# ALGORITHM="er" # 'er', 'ba', 'sbm', 'sfn', 'complete', 'star', 'path'

echo "The output path is set to: $TASK_DIR"

for node in 15 20 # 40 60 80
do
for ALGORITHM in "ba" "sbm" "sfn" # "star" "path" "complete"
do
for  task in "${TASKS[@]}"
do
  echo "Generating examples for task $task"
  python3 -m graph_tasks.graph_task_generator \
                --task=$task \
                --algorithm=$ALGORITHM \
                --task_dir=$TASK_DIR \
                --graphs_dir=$GRAPHS_DIR \
                --random_seed=1234 \
                --split=train \
                --min_nodes $node --max_nodes $((node+1))
done
done


for ALGORITHM in "ba" "sbm" "sfn" # "star" "path" "complete"
do
for  task in "${TASKS[@]}"
do
  echo "Generating examples for task $task"
  python3 -m graph_tasks.graph_task_generator \
                --task=$task \
                --algorithm=$ALGORITHM \
                --task_dir=$TASK_DIR \
                --graphs_dir=$GRAPHS_DIR \
                --random_seed=1234 \
                --split=valid \
                --min_nodes $node --max_nodes $((node+1))
done
done

for ALGORITHM in "ba" "sbm" "sfn" # "star" "path" "complete"
do
for  task in "${TASKS[@]}"
do
  echo "Generating examples for task $task"
  python3 -m graph_tasks.graph_task_generator \
                --task=$task \
                --algorithm=$ALGORITHM \
                --task_dir=$TASK_DIR \
                --graphs_dir=$GRAPHS_DIR \
                --random_seed=1234 \
                --split=test \
                --min_nodes $node --max_nodes $((node+1))
done
done
done