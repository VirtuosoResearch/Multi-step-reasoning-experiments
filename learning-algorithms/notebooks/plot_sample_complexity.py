# %%
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns

from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed

f, ax2 = plt.subplots(figsize=(6, 4))


w_steps = 100-np.array([
30.5565,
34.9500,
50.5500,
89.9000,
93.5500,
99.1500,
])
w_steps_std = np.array([
0.1546,
0.5500,
3.9500,
1.1000,
0.2500,
0.1500,
])
wo_steps = 100-np.array([
34.0446,
35.3304,
55.6250,
70.2500,
76.8750,
81.5000,
])
wo_steps_std = np.array([
0.2946,
3.8304,
0.8750,
0.2500,
0.3750,
0.2500,
])
x = np.arange(len(w_steps))
for i in range(len(x)):
    scatter2 = ax2.scatter(x[i], w_steps[i], s=200, marker="s", edgecolors = "none", facecolors='orange')

for i in range(len(x)):
    scatter1 = ax2.scatter(x[i], wo_steps[i], s=200, marker="o", edgecolors = "none", facecolors='royalblue')


p1 = ax2.plot(x, wo_steps, lw = 4, color="royalblue", linestyle="dashed", label=r"$\mathrm{w/o~intermediate~steps}$")
# ax2.fill_between(
#     x, 
#     wo_steps+wo_steps_std,
#     wo_steps-wo_steps_std, 
#     color="royalblue", alpha=0.3
# )


p2 = ax2.plot(x, w_steps, lw = 4, color="orange", linestyle="solid", label=r"$\mathrm{w/~intermediate~steps}$")
# ax2.fill_between(
#     x, 
#     w_steps+w_steps_std,
#     w_steps-w_steps_std, 
#     color="orange", alpha=0.3
# )

ax2.set_ylabel(r'$\mathrm{Error~rate}~(\%)$', fontsize = 32)
ax2.set_xlabel(r'$\mathrm{\#~training~samples}$', fontsize = 32) 
plt.xticks(x, [r"$100$", r"$200$", r"$500$", r"$1000$", r"$2000$", r"$5000$"])
ax2.set_yticks(np.arange(-20, 81, 20))
ax2.set_ylim((-10, 150))
# ax.set_xlim((-2.5, 3.5))

ax2.tick_params(labelsize=32)
ax2.grid(ls=':', lw=0.8)
# plt.title(r"$\mathrm{Evaluated~on~Bellman}$" + "-"  + r"$\mathrm{Ford}$", fontsize=32)
plt.legend(fontsize=24)

plt.tight_layout()
plt.savefig("./figures/plot_sample_complexity_dfs_w_and_wo_steps.pdf", format="pdf", dpi=1200)
plt.show()