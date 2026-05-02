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
MEAN_RECURSIVE_LOG_RHO_T_1 = np.array([1.00000000e+04, 3.62511394e+04, 1.37013767e+05, 4.74817139e+05,
       1.59053437e+06, 5.81889180e+06]) # 1.68572301e+07

prefix="bfs"  # "bellman_ford" # "bfs" #
output_dir=Path(__file__).resolve().parents[1] / "notebooks" / "figures"

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
mpl.rcParams["savefig.dpi"] = 1200
mpl.rcParams["text.usetex"] = True

output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{prefix}_jacobian_scaling.pdf"

fig, ax_loss = plt.subplots(figsize=(4.5, 4.5))
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
    MEAN_RECURSIVE_LOG_RHO_T_1,
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
ax_loss.tick_params(axis="both", labelsize=28)
ax_rho.tick_params(axis="y", labelsize=28)
ax_loss.tick_params(axis="y")
ax_rho.tick_params(axis="y")
ax_loss.set_ylim(-0.12, 4)
# ax_loss.set_yticks([0, 2, 4, 6, 8])
ax_rho.set_ylim(-2e5, 8e6)
ax_rho.set_yticks([0, 2e6, 4e6, 6e6, 8e6])

ax_loss.grid(ls=":", lw=0.8)
ax_rho.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax_rho.yaxis.get_offset_text().set_fontsize(26)

# ax_loss.legend(
#     handles=[loss_line, rho_line],
#     fontsize=20,
#     loc="upper left",
#     frameon=True,
# )

plt.tight_layout()
plt.savefig(output_path, format="pdf", dpi=1200)
plt.show()
print(f"Saved {output_path}")


# %%
