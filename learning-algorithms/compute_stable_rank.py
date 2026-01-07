# %%
import torch

checkpoint_dir = "./external_lightning_logs/Qwen-Qwen2.5-1.5B_dfs_use_only_answer_output_lora_r_16_clrs_dfs_wo_steps_only_lora_run_0/epoch_epoch=6.pt"
# "./external_lightning_logs/Qwen-Qwen2.5-1.5B_dfs_use_only_answer_output_lora_r_16_clrs_dfs_wo_steps_only_lora_run_1/epoch_epoch=8.pt"
# checkpoint_dir = "./external_lightning_logs/Qwen-Qwen2.5-1.5B_dfs_lora_r_16_clrs_dfs_only_lora_run_0/epoch_epoch=8.pt"

state_dict = torch.load(checkpoint_dir)

# %%
import numpy as np
stable_ranks = []
for key in state_dict.keys():
    if "lora_A" in key:
        # load lora_A and lora_B and restore the weight matrix
        weight_A = state_dict[key]
        weight_B = state_dict[key.replace("lora_A", "lora_B")]
        weight = weight_B @ weight_A
        # convert to float32
        weight = weight.float()
        
        # compute statle rank
        frob_norm = torch.norm(weight)
        spectral_norm = torch.linalg.norm(weight, ord=2)
        stable_ranks.append((frob_norm**2/spectral_norm**2).item())
print(np.sum(stable_ranks))

# %%