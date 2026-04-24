# %%
from src.custom.clrs_text_task_data_module import TextCLRSDataModule
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
data_module =TextCLRSDataModule(
    task_names=["mst_prim"],
    tokenizer=tokenizer,
    batch_size=8,
    inference_batch_size=8,
    max_input_length=1024,
    max_output_length=1024,
    eval_all=True,
    eval_split=0.2,
    downsample_ratio=1,
    minimum_samples=10000,
    minimum_samples_validation=10000,
    train_lengths=[12],
    test_lengths=[12],
    use_few_shot=False, 
    few_shot_k=0,
    only_answer_output=False,
    reduce_steps_ratio=1)
data_module.setup(stage="fit")

# %%
# measure input and output lengths
max_input_length = 0 
max_output_length = 0
for i in range(len(data_module.multitask_train_dataset.datasets[0])):
    sample = data_module.multitask_train_dataset.datasets[0][i]
    new_input_length = len(tokenizer(sample['input'])['input_ids'])
    # if new_input_length > 2048:
    #     print(new_input_length)
    new_output_length = len(tokenizer(sample['output'])['input_ids'])
    # if new_output_length > 1024:
    #     print(new_output_length)
    max_input_length = max(max_input_length, len(tokenizer(sample['input'])['input_ids']))
    max_output_length =  max(max_output_length, len(tokenizer(sample['output'])['input_ids']))
for i in range(len(data_module.multitask_valid_dataset.datasets[0])):
    sample = data_module.multitask_valid_dataset.datasets[0][i]
    new_input_length = len(tokenizer(sample['input'])['input_ids'])
    new_output_length = len(tokenizer(sample['output'])['input_ids'])
    max_input_length = max(max_input_length, len(tokenizer(sample['input'])['input_ids']))
    max_output_length =  max(max_output_length, len(tokenizer(sample['output'])['input_ids']))

print(max_input_length, max_output_length)

# mst_kruskal 509 210
# mst_prim 615 403
# bfs 345 259
# dfs 338 609