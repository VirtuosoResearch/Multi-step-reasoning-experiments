
# Overview

Brief guide for the learning-algorithms folder. This project contains training scripts, task generators, and data modules for experimenting with algorithmic and math-language tasks (CLRS-style and graph tasks). 

## Quick start

Prerequisites
- Python 3.8+ (see `environment.yml` / `requirements.txt` for specifics)

Install dependencies (example):

```zsh
pip install -r requirements.txt
# or create the conda env: conda env create -f environment.yml
```

Run a training script (examples):

```zsh
# Train generic model entrypoint
python train.py

# Train CLRS-text-style tasks
python train_clrs_text.py

# Train math tasks / GSM8K-style experiments
python train_math.py
```

Several convenience shell scripts are available under `scripts/` for common training setups and grouped experiments.

Test noise stability:
```bash
bash scripts/noise_stability/test_dfs.sh
```

## Repository layout

- `train.py`, `train_clrs_text.py`, `train_math.py` — top-level Python entrypoints that wire datasets, models, and training loops.
- `clrs_text_tasks/` — CLRS-style text task definitions, encoders, and utilities. Key files:
	- `tasks.py`, `graph_text_encoder.py`, `utils.py`, `name_dictionaries.py`
- `graph_tasks/` — graph task generators and utilities. Contains scriptable generators and helpers used to produce training instances.
- `scripts/` — bash helpers to run grouped experiments, training batches, evaluation runs, and dataset generation. See subfolders for specialized experiments (graph, math, CLRS-text).
- `src/` — core PyTorch/Lightning datamodules and model wiring used by training scripts. Notable files and folders:
	- `src/custom/` — data modules and task-specific data loaders
	- `src/model/` — model wrappers and architecture definitions (GraphLlama, GNNs, etc.)
- `model/` — model definitions and architectures used in experiments.
- `utils/` — assorted utility code used across scripts and training.

## Tasks & data generation

This repo contains both on-the-fly task generators (in `graph_tasks/`) and dataset/task definitions (in `clrs_text_tasks/`). Use the provided generator scripts (shell wrappers in `graph_tasks/` and `scripts/`) to produce datasets for offline evaluation or training.