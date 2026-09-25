# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
mpl.rcParams["savefig.dpi"] = 1200
mpl.rcParams["text.usetex"] = True


tasks = ["Prim's", "Dijkstra", "Bellman-ford"]
methods = ["SFT CoT", "Implicit CoT", "Coconut", "Ours"]

# Rows: tasks, Columns: methods

test_loss = np.array(
    [
        [2.908557177, 2.175646871, 1.674367203, 1.131858978],
        [3.401780576, 3.232215506, 3.12263763, 1.276884651],
        [1.839802718, 1.504626811, 1.461021996, 0.8711714745],
    ]
)

error_amplification = np.array(
    [
        [53.5193129, 30.72588793, 13.69779096, 5.179336279],
        [339.6444149, 23.79096727, 17.92498929, 6.047411224],
        [243.1744987, 153.3734751, 24.97575983, 3.032948647],
    ]
)

method_styles = [
    {"color": "lightgrey", "hatch": "", "label": r"$\mathrm{SFT}$" + "-" + r"$\mathrm{CoT}$"},
    {"color": "royalblue", "hatch": "", "label": r"$\mathrm{Implicit~CoT}$"},
    {"color": "forestgreen", "hatch": "", "label": r"$\mathrm{Coconut}$"},
    {"color": "orange", "hatch": "", "label": r"$\mathrm{Alg.~1}$"},
]


def plot_grouped_bars(values, ylabel, out_path, y_ticks=None, y_limit=None):
    n = len(tasks)
    ind = np.arange(n) * 24
    width = 4.0
    shift = 0.8

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    fig, ax = plt.subplots(figsize=(7.5, 5))

    bars = []
    for i, style in enumerate(method_styles):
        bars.append(
            ax.bar(
                ind + (width + shift) * i,
                values[:, i],
                width,
                color=style["color"],
                hatch=style["hatch"],
                ecolor="white",
            )
        )

    ax.set_xticks(ind + (width + shift) * 1.5 )
    ax.set_xticklabels(tasks, fontsize=32)

    if y_ticks is not None:
        plt.yticks(y_ticks)
    if y_limit is not None:
        plt.ylim(y_limit)

    # ax.set_title(ylabel, fontsize=32)
    ax.legend(
        (bars[0][0], bars[1][0], bars[2][0], bars[3][0]),
        (method_styles[0]["label"], method_styles[1]["label"], method_styles[2]["label"], method_styles[3]["label"]),
        loc=2,
        fontsize=26,
        ncol=2,
    )

    ax.yaxis.grid(True, lw=0.4)
    ax.tick_params(axis="both", which="major", labelsize=32)
    ax.tick_params(axis="both", which="minor", labelsize=32)

    plt.tight_layout()
    plt.savefig(out_path, format="pdf", dpi=100)


plot_grouped_bars(
    test_loss,
    r"$\mathrm{Test~Loss}$",
    "./figures/test_loss_comparison.pdf",
    y_ticks=np.arange(0, 4.1, 1.0),
    y_limit=[0, 4.8],
)

# %%

def plot_grouped_bars(values, ylabel, out_path, y_ticks=None, y_limit=None):
    n = len(tasks)
    ind = np.arange(n) * 24
    width = 4.0
    shift = 0.8

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    fig, ax = plt.subplots(figsize=(7.5, 5))

    bars = []
    for i, style in enumerate(method_styles):
        bars.append(
            ax.bar(
                ind + (width + shift) * i,
                values[:, i],
                width,
                color=style["color"],
                hatch=style["hatch"],
                ecolor="white",
            )
        )

    ax.set_xticks(ind + (width + shift) * 1.5 )
    ax.set_xticklabels(tasks, fontsize=32)

    if y_ticks is not None:
        plt.yticks(y_ticks)
    if y_limit is not None:
        plt.ylim(y_limit)

    # ax.set_title(ylabel, fontsize=32)
    # ax.legend(
    #     (bars[0][0], bars[1][0], bars[2][0], bars[3][0]),
    #     (method_styles[0]["label"], method_styles[1]["label"], method_styles[2]["label"], method_styles[3]["label"]),
    #     loc=2,
    #     fontsize=26,
    #     ncol=2,
    # )

    ax.yaxis.grid(True, lw=0.4)
    ax.tick_params(axis="both", which="major", labelsize=32)
    ax.tick_params(axis="both", which="minor", labelsize=32)

    plt.tight_layout()
    plt.savefig(out_path, format="pdf", dpi=100)


plot_grouped_bars(
    error_amplification,
    r"$\mathrm{Error~Amplification~Factor}$",
    "./figures/error_amplification_factor_comparison.pdf",
    y_ticks=np.arange(0, 101, 20),
    y_limit=[0, 90],
)
