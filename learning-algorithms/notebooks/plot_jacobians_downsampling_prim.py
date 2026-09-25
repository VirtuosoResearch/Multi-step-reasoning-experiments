# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
RATIO_VALUES = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
LOG_SUM_RHOS = np.array([
7.44452523,
4.84039670,
3.525654767,
2.245632213,
3.208773732
])
ERROR_RATES = np.array([2.39  , 1.905 , 1.79  , 1.7   , 1.8855])

# 100 - np.array([
# 56.20,
# 61.90,
# 64.20,
# 66.00,
# 62.29])

prefix="mst_prim"  # "bellman_ford" # "bfs" #
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

# ax_loss.legend(
#     handles=[loss_line, rho_line],
#     fontsize=2,
#     loc="upper left",
#     frameon=True,
# )

plt.tight_layout()
plt.savefig(output_path, format="pdf", dpi=1200)
plt.show()
print(f"Saved {output_path}")


# %%
