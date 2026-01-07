# %%
import os
import numpy as np

path = "../outputs/evaluate_compression_bound/train_dfs_length_10_w_steps_v4.log"
# read a file 
training_losses = []; count = 0
with open(path, "r") as file:
    for line in file.readlines():
        # get the number in a line like "train_loss_step=0.0993"
        if "train_loss_step" in line:
            loss = float(line.split("train_loss_step=")[-1][:5])
            training_losses.append(loss)
            count += 1
        if count >= 1250*10:
            break

np.save("./dfs_w_steps_training_losses.npy", np.array(training_losses))
# %%
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns

def smooth(losses, window_size=10):
    return np.convolve(losses, np.ones(window_size)/window_size, mode='valid')

from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed

f, ax2 = plt.subplots(figsize=(6, 4))

training_loss_w_steps = np.load("./dfs_w_steps_training_losses.npy")
training_loss_wo_steps = np.load("./dfs_wo_steps_training_losses.npy")
training_loss_wo_steps = smooth(training_loss_wo_steps, 40)
training_loss_w_steps = smooth(training_loss_w_steps, 40)
# training_loss_wo_steps /= training_loss_wo_steps[0]
# training_loss_w_steps /= training_loss_w_steps[0]
x = np.arange(len(training_loss_wo_steps))
p1 = ax2.plot(x, training_loss_wo_steps, lw = 2, color="royalblue", linestyle="dashed", label=r"$\mathrm{w/o~intermediate~steps}$")
p2 = ax2.plot(x, training_loss_w_steps, lw = 3, color="orange", linestyle="solid", label=r"$\mathrm{w/~intermediate~steps}$")


ax2.set_ylabel(r'$\mathrm{Training~loss}$', fontsize = 32)
ax2.set_xlabel(r'$\mathrm{\#~steps}$', fontsize = 32) 
# plt.xticks(x, [r"$100$", r"$200$", r"$500$", r"$1000$", r"$2000$", r"$5000$"])
# ax2.set_yticks(np.arange(-20, 81, 20))
# ax2.set_xticks([0, 1000, 2000, 3000])
ax2.set_ylim((-0.1, 1.3))
# ax.set_xlim((-2.5, 3.5))

ax2.tick_params(labelsize=32)
ax2.grid(ls=':', lw=0.8)
# plt.title(r"$\mathrm{Evaluated~on~Bellman}$" + "-"  + r"$\mathrm{Ford}$", fontsize=32)
plt.legend(fontsize=24)

plt.tight_layout()
plt.savefig("./figures/plot_training_convergence_dfs_w_and_wo_steps.pdf", format="pdf", dpi=1200)
plt.show()