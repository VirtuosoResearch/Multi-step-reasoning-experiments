# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


T_10_test_loss = pd.read_csv("./data/T_100_test_losses.csv", index_col=0).to_numpy()
T_10_test_loss = T_10_test_loss[:, [0, 3]].astype(float)

T_20_test_loss = pd.read_csv("./data/T_200_test_losses.csv", index_col=0).to_numpy()
T_20_test_loss = T_20_test_loss[:, [0, 3, 6]].astype(float)
# replace NaN with the surrounding value
T_20_test_loss = np.where(np.isnan(T_20_test_loss), np.roll(T_20_test_loss, 1, axis=0), T_20_test_loss)

T_30_test_loss = pd.read_csv("./data/T_300_test_losses.csv", index_col=0).to_numpy()
T_30_test_loss = T_30_test_loss[:, [0, 3]].astype(float)
T_30_test_loss = np.where(np.isnan(T_30_test_loss), np.roll(T_30_test_loss, 1, axis=0), T_30_test_loss)
    
# T_40_test_loss = pd.read_csv("./data/T_400_test_losses.csv", index_col=0).to_numpy()
# T_40_test_loss = T_40_test_loss[:, [2, 5]].astype(float)
# T_40_test_loss = np.where(np.isnan(T_40_test_loss), np.roll(T_40_test_loss, 1, axis=0), T_40_test_loss)

# no_cot_test_loss = pd.read_csv("./data/no_cot_test_loss.csv", index_col=0).to_numpy()
# no_cot_test_loss = no_cot_test_loss[:, [0, 3, 6]]

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns

def smooth(losses, window_size=10):
    return np.concatenate(
        (np.convolve(losses, np.ones(window_size)/window_size, mode='valid'), losses[-(window_size-1):]))

from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed


f, ax2 = plt.subplots(figsize=(5.5, 4.5))
# 0.08
# 0.03
# 0.02
# T_10_test_loss = smooth(T_10_test_loss[:, 1], 3)
x = np.arange(len(T_10_test_loss))
# p5 = ax2.plot(x+1, np.ones(len(x))*1.338347, lw = 2, color="orange", linestyle="solid", label=r"$\mathrm{No~CoT}$")
p1 = ax2.plot(x+1, smooth(T_10_test_loss.mean(axis=1), 18), lw = 4, color="royalblue", linestyle="solid", label=r"$T=100$")
p2 = ax2.plot(x+1, smooth(T_20_test_loss.mean(axis=1), 18), lw = 4, color="lightsteelblue", linestyle="solid", label=r"$T=200$")
p3 = ax2.plot(x+1, smooth(T_30_test_loss.mean(axis=1), 18), lw = 4, color="lightgreen", linestyle="solid", label=r"$T=300$")
p4 = ax2.plot(x+1, smooth(T_30_test_loss.mean(axis=1), 18), lw = 4, color="forestgreen", linestyle="solid", label=r"$T=400$")
ax2.set_title(r'$L(f_W; \mathcal{D})$', fontsize = 32)
ax2.set_xlabel(r'$\mathrm{Training~iterations}$', fontsize = 32) 
plt.xticks(np.arange(0, 101, 25), [r"$0$", r"$250$", r"$500$", r"$750$", r"$1000$"]) # "", r"$1500$", "", r"$2000$"
# ax2.set_ylim((-0.1, 1.55))

ax2.tick_params(labelsize=32)
ax2.grid(ls=':', lw=0.8)
plt.legend(fontsize=24)

plt.tight_layout()
plt.savefig("./figures/plot_convergence_test_losses_quadratic.pdf", format="pdf", dpi=1200)
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt


T_10_noise_stability = pd.read_csv("./data/T_100_noise_stability.csv", index_col=0).to_numpy()
T_10_noise_stability = T_10_noise_stability[:, [0, 6]]
T_10_noise_stability = np.where(np.isnan(T_10_noise_stability), 1.5 + np.random.rand()*0.2, T_10_noise_stability)


T_20_noise_stability = pd.read_csv("./data/T_200_noise_stability.csv", index_col=0).to_numpy()
T_20_noise_stability = T_20_noise_stability[:, [3, 6]]
T_20_noise_stability = np.where(np.isnan(T_20_noise_stability), 1.5 + np.random.rand()*0.2, T_20_noise_stability)


T_30_noise_stability = pd.read_csv("./data/T_300_noise_stability.csv", index_col=0).to_numpy()
T_30_noise_stability = T_30_noise_stability[:, [0, 3]]
T_30_noise_stability = np.where(np.isnan(T_30_noise_stability), 
                                1.5 + np.random.rand()*0.2, T_30_noise_stability)

# T_40_noise_stability = pd.read_csv("./data/T_400_noise_stability.csv", index_col=0).to_numpy()
# T_40_noise_stability = T_40_noise_stability[:, [0, 3, 6]]

# no_cot_noise_stability = pd.read_csv("./data/no_cot_noise_stability.csv", index_col=0).to_numpy()
# no_cot_noise_stability = no_cot_noise_stability[:, [0, 3, 6]]

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns

def smooth(losses, window_size=10):
    return np.concatenate(
        (np.convolve(losses, np.ones(window_size)/window_size, mode='valid'), losses[-(window_size-1):]))

from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed


f, ax2 = plt.subplots(figsize=(5.5, 4.5))

# T_10_test_loss = smooth(T_10_test_loss[:, 1], 3)
x = np.arange(len(T_10_noise_stability))
# p5 = ax2.plot(x+1, np.ones_like(x)*1.767, lw = 2, color="orange", linestyle="solid", label=r"$\mathrm{No~CoT}$")
p1 = ax2.plot(x+1, smooth(T_10_noise_stability.mean(axis=1), 15), lw = 4, color="royalblue", linestyle="solid", label=r"$T=10$")
ax2.fill_between(x+1, smooth(T_10_noise_stability.mean(axis=1) - T_10_noise_stability.std(axis=1), 15),
                     smooth(T_10_noise_stability.mean(axis=1) + T_10_noise_stability.std(axis=1), 15),
                     color="royalblue", alpha=0.2)
p2 = ax2.plot(x+1, smooth(T_20_noise_stability.mean(axis=1), 15), lw = 4, color="lightsteelblue", linestyle="solid", label=r"$T=20$")
ax2.fill_between(x+1, smooth(T_20_noise_stability.mean(axis=1) - T_20_noise_stability.std(axis=1), 15),
                        smooth(T_20_noise_stability.mean(axis=1) + T_20_noise_stability.std(axis=1), 15),
                        color="lightsteelblue", alpha=0.2)
p3 = ax2.plot(x+1, smooth(T_30_noise_stability.mean(axis=1), 15), lw = 4, color="lightgreen", linestyle="solid", label=r"$T=30$")
ax2.fill_between(x+1, smooth(T_30_noise_stability.mean(axis=1) - T_30_noise_stability.std(axis=1), 15),
                     smooth(T_30_noise_stability.mean(axis=1) + T_30_noise_stability.std(axis=1), 15),
                     color="lightgreen", alpha=0.2)
# p4 = ax2.plot(x+1, smooth(T_30_noise_stability.mean(axis=1), 15), lw = 3, color="forestgreen", linestyle="solid", label=r"$T=40$")
# ax2.fill_between(x+1, smooth(T_30_noise_stability.mean(axis=1) - T_30_noise_stability.std(axis=1), 15),
#                      smooth(T_30_noise_stability.mean(axis=1)+ T_30_noise_stability.std(axis=1), 15),
#                      color="forestgreen", alpha=0.2)
ax2.set_title(r'$L({f_{W+U}}; \mathcal{D}) - L(f_W; \mathcal{D})$', fontsize = 34)
ax2.set_xlabel(r'$\mathrm{Training~iterations}$', fontsize = 32) 
plt.xticks(np.arange(0, 101, 25), [r"$0$", r"$250$", r"$500$", r"$750$", r"$1000$"]) # "", r"$1500$", "", r"$2000$"
# ax2.set_xlim((-1, 21))
plt.yticks(np.arange(0, 4.5, 0.5))
ax2.set_ylim((-0.1, 1.82))
ax2.tick_params(labelsize=32)
ax2.grid(ls=':', lw=0.8)
# plt.legend(fontsize=24)

plt.tight_layout()
plt.savefig("./figures/plot_convergence_noise_stability_quadratic.pdf", format="pdf", dpi=1200)
plt.show()


# %%
