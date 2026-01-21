# %%
import torch

checkpoint = torch.load('./checkpoints/nonlinear_attention_model_input10_examples200_lr0.001_bs1000_T100_quad_v1/best_model.pt', map_location='cpu')
state_dict = checkpoint['model_state_dict'] 

# %%
import matplotlib.pyplot as plt
import numpy as np

plt.imshow(state_dict['attention.W_kq.weight'].numpy(), cmap='viridis')

# %%
plt.imshow(state_dict['attention.W_pv.weight'].numpy(), cmap='viridis')


# %%
plt.imshow(state_dict['attention.W_kq_2.weight'].numpy(), cmap='viridis')

# %%
plt.imshow(state_dict['attention.W_kq_3.weight'].numpy(), cmap='viridis')

# %%

# %%
plt.imshow(state_dict['attention.W_v_1.weight'].numpy(), cmap='viridis')

# %%

# %%
plt.imshow(state_dict['attention.W_v_2.weight'].numpy(), cmap='viridis')

# %%

# %%
plt.imshow(state_dict['attention.W_v_3.weight'].numpy(), cmap='viridis')

# %%

# %%
plt.imshow(state_dict['attention.W_pv.weight'].numpy(), cmap='viridis')


# %%

# %%
print(state_dict['attention.W_pv.weight'].numpy()[11:21, 0:10])

# %%
print(state_dict['attention.W_kq.weight'].numpy()[0:11, 11:22])