# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

BIT_LABELS = [r"$32$", r"$4$", r"$3$", r"$2$", r"$1$"]
PRIM_VALUES = np.exp(np.array([
3.538360564,
3.116990879,
2.634454808,
2.244766177,
2.234011947
]))

prefix = "dijkstra"
output_dir = Path(__file__).resolve().parents[1] / "notebooks" / "figures"

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
mpl.rcParams["savefig.dpi"] = 1200
mpl.rcParams["text.usetex"] = True

output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{prefix}_bits_bar.pdf"

fig, ax = plt.subplots(figsize=(5.8, 5))

x = np.arange(len(BIT_LABELS))
bar_color = "royalblue"

ax.bar(x, PRIM_VALUES, color=bar_color, width=0.65)

ax.set_xlabel(r"$\mathrm{Bit~precision}$", fontsize=32)
ax.set_ylabel(r"$\sum_{i=1}^{T-1} \rho_{T, i}$", fontsize=28)
ax.set_xticks(x, BIT_LABELS)
# ax.set_yticks(np.arange(0, 4.1, 1), [r"$1$", r"$2$", r"$8$", r"$20$", r"$50$"])
ax.tick_params(axis="both", labelsize=28)
ax.grid(axis="y", ls=":", lw=0.8)

plt.tight_layout()
plt.savefig(output_path, format="pdf", dpi=1200)
plt.show()
print(f"Saved {output_path}")


# %%
