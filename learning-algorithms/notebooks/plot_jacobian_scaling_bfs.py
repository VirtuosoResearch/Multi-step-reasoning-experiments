# %%
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc


T_VALUES = np.array([1, 2, 3, 4, 5, 6])
MEAN_L_T_GAP = np.array([
0.0,
0.0895441,
0.1417674,
0.1814788,
0.339238,
1.104263,
# 1.6247299,
])
bound_values = np.array([
    0,
0.3449774,
0.8109022,
1.2213081,
2.3032357,
6.3612268
])
# MEAN_RECURSIVE_LOG_RHO_T_1 = np.array([1.00000000e+04, 3.62511394e+04, 1.37013767e+05, 4.74817139e+05,
#        1.59053437e+06, 5.81889180e+06]) # 1.68572301e+07

prefix="bfs"  # "bellman_ford" # "bfs" #
output_dir=Path(__file__).resolve().parents[1] / "notebooks" / "figures"

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
mpl.rcParams["savefig.dpi"] = 1200
mpl.rcParams["text.usetex"] = True

output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{prefix}_jacobian_scaling.pdf"

fig, ax_loss = plt.subplots(figsize=(7, 5))
ax_rho = ax_loss.twinx()

loss_color = "royalblue"
rho_color = "forestgreen"

loss_line = ax_loss.plot(
    T_VALUES,
    MEAN_L_T_GAP,
    lw=5,
    color=loss_color,
    linestyle="solid",
    marker="o",
    markersize=12,
    label=r"$\mathrm{Inference~loss}~\mathbf{~(left~axis)}$",
)[0]
rho_line = ax_rho.plot(
    T_VALUES,
    bound_values,
    lw=5,
    color=rho_color,
    linestyle="dashed",
    marker="s",
    markersize=12,
    label=r"$\mathrm{Our~bound}~\mathbf{~(right~axis)}$",
)[0]

ax_loss.set_xlabel(r"$T$", fontsize=32)
# ax_loss.set_ylabel( fontsize=28)
# ax_rho.set_ylabel(r"$\rho_{t,1}$", fontsize=32)

ax_loss.set_xticks(T_VALUES)
ax_loss.tick_params(axis="both", labelsize=32)
ax_rho.tick_params(axis="y", labelsize=32)
ax_loss.tick_params(axis="y")
ax_rho.tick_params(axis="y")
ax_loss.set_ylim(-0.2, 3.5)
ax_loss.set_yticks([0, 2, 4])
ax_rho.set_ylim(-0.4, 7)
ax_rho.set_yticks([0, 4, 8])

ax_loss.grid(ls=":", lw=0.8)
ax_rho.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax_rho.yaxis.get_offset_text().set_fontsize(32)

ax_loss.legend(
    handles=[rho_line, loss_line],
    fontsize=26,
    loc="upper left",
    frameon=True,
)

plt.title(r"$\mathrm{Breadth}$" + "-" + r"$\mathrm{first~search}$", fontsize=32)

plt.tight_layout()
plt.savefig(output_path, format="pdf", dpi=1200)
plt.show()
print(f"Saved {output_path}")


# %%
