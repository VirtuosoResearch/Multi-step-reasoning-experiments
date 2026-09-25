# %%
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

T_VALUES = np.array([1, 2, 3, 4, 5])
MEAN_L_T_GAP = np.array([
0.0,
0.17715469,
0.6021179,
1.41194086,
3.82395117
])
MEAN_RECURSIVE_LOG_RHO_T_1 = np.exp(np.array([
# 0,
2.3444892565,
2.8848397386,
4.2208765713,
5.4935170968,
6.7856938006
]))
bound_values = np.array([
0.000,
6042.7315188,
97939.390018,
562378.00138,
4674023.4132,
])

# np.array([1.00000000e+06, 3.64644204e+06, 1.25179169e+07, 4.26330198e+07, 1.33170437e+08, 4.46009497e+08])

prefix="mst_prim"  # "bellman_ford" # "bfs" #
output_dir=Path(__file__).resolve().parents[1] / "notebooks" / "figures"

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
mpl.rcParams["savefig.dpi"] = 1200
mpl.rcParams["text.usetex"] = True

output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{prefix}_jacobian_scaling_llama.pdf"

fig, ax_loss = plt.subplots(figsize=(7.5, 5))
ax_rho = ax_loss.twinx()

loss_color = "royalblue"
rho_color = "forestgreen"

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

loss_line = ax_loss.plot(
    T_VALUES,
    MEAN_L_T_GAP,
    lw=5,
    color=loss_color,
    linestyle="solid",
    marker="o",
    markersize=12,
    label=r"$\mathrm{Test~loss}~\mathbf{~(left~axis)}$",
)[0]
# bound_line = ax_loss.plot(
#     T_VALUES,
#     bound_values,
#     lw=5,
#     color="orange",
#     linestyle="solid",
#     marker="o",
#     markersize=12,
#     label=r"\mathrm{Our~bound}",
# )[0]
# ax_loss.set_xlabel(r"$T$", fontsize=32)
# ax_loss.set_ylabel( fontsize=28)
# ax_rho.set_ylabel(r"$\rho_{t,1}$", fontsize=32)

ax_loss.set_xticks(T_VALUES, [r"$T=1$", r"$T=2$", r"$T=3$", r"$T=4$", r"$T=5$"], fontsize=32)
ax_loss.tick_params(axis="both", labelsize=32)
ax_rho.tick_params(axis="y", labelsize=32)
ax_loss.tick_params(axis="y", labelsize=32)
ax_rho.tick_params(axis="y", labelsize=32)
ax_loss.set_ylim(-0.15, 5.5)
# ax_loss.set_yticks([0, 2, 4, 6, 8, 10, 12])
# ax_rho.set_ylim(-0.75, 17.5)

ax_loss.grid(ls=":", lw=0.8)
ax_rho.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax_rho.yaxis.get_offset_text().set_fontsize(15)

# ax_loss.legend(
#     handles=[rho_line, loss_line],
#     fontsize=28,
#     loc="upper left",
#     frameon=True,
# )
plt.title(r"$\mathrm{Prim}$" + "'" + r"$\mathrm{s~algorithm, Llama}$" + "-" + r"$\mathrm{1B}$", fontsize=32)
plt.tight_layout()
plt.savefig(output_path, format="pdf", dpi=1200)
plt.show()
print(f"Saved {output_path}")


# %%
