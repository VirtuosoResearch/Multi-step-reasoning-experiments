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
# set -e
set -x

# python3 -m venv graphqa
# source graphqa/bin/activate

# pip3 install -r graphqa/requirements.txt

# Fill in appropriate output path
GRAPHS_DIR="./data/graphs"
TASK_DIR="./data/tasks"
TASKS=("edge_existence" "node_degree" "node_count" "edge_count" "connected_nodes" "cycle_check" "disconnected_nodes" "reachability" "shortest_path" "maximum_flow" "triangle_counting" "node_classification")
# TASKS=("maximum_flow")

# For experimenting with only erdos-reyni graph use `er``.
# For all graph generators, set to `all`.
# ALGORITHM="er" # 'er', 'ba', 'sbm', 'sfn', 'complete', 'star', 'path'

echo "The output path is set to: $TASK_DIR"

for ALGORITHM in "er" # "ba" "sbm" "sfn" "star" "path" "complete"
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
                --min_nodes 50 --max_nodes 100
done
done


for ALGORITHM in "er" # "ba" "sbm" "sfn" "star" "path" "complete"
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
                --min_nodes 50 --max_nodes 100
done
done

for ALGORITHM in "er" # "ba" "sbm" "sfn" "star" "path" "complete"
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
                --min_nodes 50 --max_nodes 100
done
done