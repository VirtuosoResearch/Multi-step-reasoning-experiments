# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
RATIO_VALUES = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
LOG_SUM_RHOS = np.array([
29.44452523,
14.84039670,
7.525654767,
4.245632213,
6.208773732
])
ERROR_RATES = 100 - np.array([
56.20,
61.90,
64.20,
66.00,
62.29])

prefix="mst_prim"  # "bellman_ford" # "bfs" #
output_dir=Path(__file__).resolve().parents[1] / "notebooks" / "figures"

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
mpl.rcParams["savefig.dpi"] = 1200
mpl.rcParams["text.usetex"] = True

output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{prefix}_downsampled_jacobians.pdf"

fig, ax_loss = plt.subplots(figsize=(5.8, 5))
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
    label=r"$\mathrm{Test~error~rate~(\%)}$",
)[0]
rho_line = ax_rho.plot(
    RATIO_VALUES,
    LOG_SUM_RHOS,
    lw=5,
    color=rho_color,
    linestyle="dashed",
    marker="s",
    markersize=12,
    label=r"$\log(\sum_{i=1}^{T-1} \rho_{T, i})~\mathbf{(right~axis)}$",
)[0]

ax_loss.set_xlabel(r"$T$", fontsize=32)
# ax_loss.set_ylabel(r"$\mathrm{Error~rate}$", fontsize=28)
# ax_rho.set_ylabel(r"$\log(\sum_{i=1}^{T-1} \rho_{T, i})$", fontsize=28)

ax_loss.set_xticks(RATIO_VALUES, [r"$20$", r"$10$", r"$5$", r"$2$", r"$0$"])
ax_loss.tick_params(axis="both", labelsize=28)
ax_rho.tick_params(axis="y", labelsize=28)
ax_loss.tick_params(axis="y")
ax_rho.tick_params(axis="y")
ax_loss.set_ylim(20, 60)
# ax_loss.set_yticks([0, 8, 16, 24, 32])
ax_rho.set_ylim(0, 60)
ax_rho.set_yticks(np.arange(0, 61, 15), labels=[r"$10^{1}$", r"$10^{6}$", r"$10^{12}$", r"$10^{18}$", r"$10^{24}$"])

ax_loss.grid(ls=":", lw=0.8)
# ax_rho.ticklabel_format(axis="y", style="plain")

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
