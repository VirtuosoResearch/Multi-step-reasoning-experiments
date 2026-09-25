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
0.0508688,
0.2657029,
0.523178,
0.7506365,
2.285090,
# 2.9809759,
])
bound_values = np.array([
0,
0.1481915,
0.9249265,
2.3797756,
7.1525192,
14.223101
])

# MEAN_RECURSIVE_LOG_RHO_T_1 = np.array([1.00000000e+06, 3.40455992e+06, 1.43422838e+07, 5.08439297e+07,
#        1.60730230e+08, 5.28461229e+08]) # 1.46324452e+09

prefix="dijkstra"  # "bellman_ford" # "bfs" #
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
    label=r"$L_t(f_W)-L_t^{\mathrm{CoT}}(f_W)$",
)[0]
rho_line = ax_rho.plot(
    T_VALUES,
    bound_values,
    lw=5,
    color=rho_color,
    linestyle="dashed",
    marker="s",
    markersize=12,
    label=r"$\rho_{t,1}\mathrm{~(right~axis)}$",
)[0]

ax_loss.set_xlabel(r"$T$", fontsize=32)
# ax_loss.set_ylabel( fontsize=28)
# ax_rho.set_ylabel(r"$\rho_{t,1}$", fontsize=32)

ax_loss.set_xticks(T_VALUES)
ax_loss.tick_params(axis="both", labelsize=32)
ax_rho.tick_params(axis="y", labelsize=32)
ax_loss.tick_params(axis="y")
ax_rho.tick_params(axis="y")
ax_loss.set_ylim(-0.2, 4.0)
ax_loss.set_yticks([0, 1, 2, 3, 4])
ax_rho.set_ylim(-0.8, 16)
ax_rho.set_yticks([0, 4, 8, 12, 16])

ax_loss.grid(ls=":", lw=0.8)
# ax_rho.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax_rho.yaxis.get_offset_text().set_fontsize(32)
plt.title(r"$\mathrm{Dijkstra}$" + "" + r"$\mathrm{~algorithm}$", fontsize=32)

# ax_loss.legend(
#     handles=[loss_line, rho_line],
#     fontsize=24,
#     loc="upper left",
#     frameon=True,
# )

plt.tight_layout()
plt.savefig(output_path, format="pdf", dpi=1200)
plt.show()
print(f"Saved {output_path}")


# %%
