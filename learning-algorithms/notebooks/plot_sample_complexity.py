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

f, ax2 = plt.subplots(figsize=(6.5, 5))

# Dijkstra
samples_1 = np.array([
10000,
7000,
4000,
2000,
200,
])

# Bellman-Ford
samples_2 = np.array([
10000,
9000,
7000,
5000,
1000,
])

x = np.arange(len(samples_1))
for i in range(len(x)):
    scatter2 = ax2.scatter(x[i], samples_1[i], s=200, marker="o", edgecolors = "none", facecolors='red')

for i in range(len(x)):
    scatter1 = ax2.scatter(x[i], samples_2[i], s=200, marker="s", edgecolors = "none", facecolors='royalblue')

p1 = ax2.plot(x, samples_2, lw = 4, color="royalblue", linestyle="dashed", label=r"$\mathrm{Bellman~Ford}$")
p2 = ax2.plot(x, samples_1, lw = 4, color="red", linestyle="solid", label=r"$\mathrm{Dijkstra}$")

ax2.set_ylabel(r'$\mathrm{\#~Training~Samples}$', fontsize = 32)
ax2.set_xlabel(r'$T$', fontsize = 32) 
plt.xticks(x, [r"$0$", r"$2$", r"$5$", r"$10$", r"$20$"])

plt.yticks([0, 2000, 4000, 6000, 8000, 10000],)
# set y as scientific notation
ax2.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
# change the font size 
ax2.yaxis.offsetText.set_fontsize(28)

ax2.tick_params(labelsize=32)
ax2.grid(ls=':', lw=0.8)
plt.legend(fontsize=24)

plt.tight_layout()
plt.savefig("./figures/plot_sample_complexity_bellman_ford_vs_dijkstra.pdf", format="pdf", dpi=1200)
plt.show()

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

f, ax2 = plt.subplots(figsize=(6.5, 5))

# Dijkstra
samples_1 = np.array([
5000,
1000,
1000,
500,
500,
])

# Bellman-Ford
samples_2 = np.array([
5000,
5000,
4000,
3000,
1000,
])

x = np.arange(len(samples_1))
for i in range(len(x)):
    scatter2 = ax2.scatter(x[i], samples_1[i], s=200, marker="o", edgecolors = "none", facecolors='red')

for i in range(len(x)):
    scatter1 = ax2.scatter(x[i], samples_2[i], s=200, marker="s", edgecolors = "none", facecolors='royalblue')

p1 = ax2.plot(x, samples_2, lw = 4, color="royalblue", linestyle="dashed", label=r"$\mathrm{Prim's}$")
p2 = ax2.plot(x, samples_1, lw = 4, color="red", linestyle="solid", label=r"$\mathrm{Kruskal's}$")

ax2.set_ylabel(r'$\mathrm{\#~Training~Samples}$', fontsize = 32)
ax2.set_xlabel(r'$T$', fontsize = 32) 
plt.xticks(x, [r"$0$", r"$2$", r"$5$", r"$10$", r"$20$"])

plt.yticks([0, 2000, 4000, 6000, 8000, 10000],)
# set y as scientific notation
ax2.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
# change the font size 
ax2.yaxis.offsetText.set_fontsize(28)

ax2.tick_params(labelsize=32)
ax2.grid(ls=':', lw=0.8)
plt.legend(fontsize=24)

plt.tight_layout()
plt.savefig("./figures/plot_sample_complexity_prim_vs_kruskal.pdf", format="pdf", dpi=1200)
plt.show()