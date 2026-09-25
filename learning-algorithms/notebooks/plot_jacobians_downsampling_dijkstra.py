# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

# Log (sum of rhos)	Error rate
# 28.96892257	53.35
# 15.74008515	63.05
# 3.538360564	73.1
# 4.286977176	67.25
# 4.818730534	64.85

RATIO_VALUES = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
LOG_SUM_RHOS = np.array([7.06892257, 4.74008515, 2.538360564, 3.286977176, 3.818730534])
ERROR_RATES = np.array([2.3325, 1.8475, 1.345 , 1.6375, 1.7575])
# 100 - np.array([53.35, 63.05, 73.1, 67.25, 64.85])

prefix="dijkstra"  # "bellman_ford" # "bfs" #
output_dir=Path(__file__).resolve().parents[1] / "notebooks" / "figures"

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
mpl.rcParams["savefig.dpi"] = 1200
mpl.rcParams["text.usetex"] = True

output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{prefix}_downsampled_jacobians.pdf"

fig, ax_loss = plt.subplots(figsize=(7, 5))
ax_rho = ax_loss.twinx()

loss_color = "royalblue"
rho_color = "forestgreen"

loss_line = ax_loss.plot(
    RATIO_VALUES,
    ERROR_RATES,
    lw=5,
    color=loss_color,
    linestyle="solid",
    marker="o",
    markersize=12,
    label=r"$\mathrm{Test~loss~\mathbf{(left~axis)}}$",
)[0]
rho_line = ax_rho.plot(
    RATIO_VALUES,
    LOG_SUM_RHOS,
    lw=5,
    color=rho_color,
    linestyle="dashed",
    marker="s",
    markersize=12,
    label=r"$\sum_{i=1}^{T-1} \rho_{T, i}~\mathbf{(right~axis)}$",
)[0]

ax_loss.set_xlabel(r"$T$", fontsize=36)
# ax_loss.set_ylabel(r"$\mathrm{Error~rate}$", fontsize=36)
# ax_rho.set_ylabel(r"$\log(\sum_{i=1}^{T-1} \rho_{T, i})$", fontsize=36)

ax_loss.set_xticks(RATIO_VALUES, [r"$20$", r"$10$", r"$5$", r"$2$", r"$0$"])
ax_loss.tick_params(axis="both", labelsize=36)
ax_rho.tick_params(axis="y", labelsize=36)
ax_loss.tick_params(axis="y")
ax_rho.tick_params(axis="y")
ax_loss.set_ylim(1, 4)
ax_loss.set_yticks([0, 1, 2, 3])
ax_rho.set_ylim(1, 13)
ax_rho.set_yticks(np.arange(1, 11, 3), labels=[r"$2$", r"$10^2$", r"$10^3$", r"$10^4$"]) 

ax_loss.grid(ls=":", lw=0.8)
# ax_rho.ticklabel_format(axis="y", style="plain")

ax_loss.legend(
    handles=[loss_line, rho_line],
    fontsize=26,
    loc="upper left",
    frameon=True,
)

plt.tight_layout()
plt.savefig(output_path, format="pdf", dpi=1200)
plt.show()
print(f"Saved {output_path}")


# %%
