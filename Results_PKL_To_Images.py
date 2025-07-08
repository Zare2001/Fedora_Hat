import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle, pandas as pd
from pathlib import Path
from IPython.display import display

# 1. load every pickle “frame” --------------------------------------
def load_all_results_stream(path="results.pkl"):
    blocks = []
    with Path(path).open("rb") as fh:
        while True:
            try:
                blocks.append(pickle.load(fh))   # one dict per combo
            except EOFError:
                break
    return blocks

blocks = load_all_results_stream("results.pkl")

# 2. expand each block into step-wise rows --------------------------
rows = []
for blk in blocks:
    # metadata constant for this experiment
    meta = {
        "timestamp": blk.get("timestamp"),
        "privacy":   blk.get("privacy"),
        "attack":    blk.get("attack"),
        "noise":     blk.get("noise"),
    }
    # how many PDMM iterations were logged
    n_steps = max(len(blk.get("loss", [])),
                  len(blk.get("accuracy", [])),
                  len(blk.get("FAR", [])))

    # create one row per step
    for step in range(n_steps):
        rows.append({
            **meta,
            "step":      step,
            "loss":      blk.get("loss",      [None]*n_steps)[step],
            "accuracy":  blk.get("accuracy",  [None]*n_steps)[step],
            "FAR":       blk.get("FAR",       [None]*n_steps)[step],
            "MDR":       blk.get("MDR",       [None]*n_steps)[step],
            "Error":     blk.get("Error",     [None]*n_steps)[step],
        })

df = pd.DataFrame(rows)

# 3. display in notebook or save to CSV -----------------------------
display(df.head())                 # shows first few rows interactively
df.to_csv("results_longform.csv", index=False)
print("Saved full step-wise table ➜ results_longform.csv")



# ------------------------------------------------------------------
# helper: draw ONE figure given a pivoted table + metric list
# ------------------------------------------------------------------
def _plot_metrics(tbl, metrics, title, fname=None):
    n = len(metrics)
    fig, axs = plt.subplots(1, n, figsize=(6*n, 4), sharex=True)
    if n == 1:
        axs = [axs]

    markers = ['o','s','d','^','v','<','>','p','*','h','+','x']
    noises  = sorted(tbl.index.get_level_values("noise").unique())

    marker_spacing = 34  # Markers appear every

    for idx, noise in enumerate(noises):
        pivot   = tbl.xs(noise, level="noise")
        marker  = markers[idx % len(markers)]
        var_lbl = fr"$\sigma^2={noise**2:g}$"

        offset = (idx *15 )% marker_spacing
        markevery = list(range(offset, len(pivot), marker_spacing))

        for m, ax in zip(metrics, axs):
            if m not in pivot.columns:
                continue
            ax.plot(
                pivot.index,
                pivot[m],
                marker=marker,
                linestyle='-',
                markevery=markevery,
                label=var_lbl,
                markersize=5
            )

    for m, ax in zip(metrics, axs):
        ax.set_ylabel(m.capitalize() if m != "loss" else "Loss")
        ax.grid(True)
        if m == "loss":
            ax.set_yscale("log")

    axs[-1].set_xlabel("Iteration")
    axs[0].set_xlabel("Iteration")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper right", bbox_to_anchor=(1.00, 0.98),
               frameon=True)

    fig.tight_layout(rect=[0, 0, 0.88, 1])

    if fname:
        fig.savefig(fname, dpi=300)
        print("saved ➜", fname)
    plt.show()






# ------------------------------------------------------------------
# main wrapper
# ------------------------------------------------------------------
def plot_noise_sweep(df_long, attack, privacy, save=False):
    subset = (df_long.query("attack == @attack and privacy == @privacy")
                      .set_index(["noise", "step"])
                      .sort_index())

    if subset.empty:
        print(f"No rows for attack={attack}, privacy={privacy}")
        return

    metric_cols = ["FAR", "MDR", "accuracy", "loss"]
    mean_tbl = subset[metric_cols].groupby(
                 level=["noise", "step"]).mean(numeric_only=True)

    base = f"attack{attack}_privacy{privacy}"
    _plot_metrics(mean_tbl,
                  metrics=["FAR", "MDR"],
                  title=f"Detection metrics  |  attack={attack}  privacy={privacy}",
                  fname=(f"detection_{base}.png" if save else None))
    _plot_metrics(mean_tbl,
                  metrics=["accuracy", "loss"],
                  title=f"Performance metrics |  attack={attack}  privacy={privacy}",
                  fname=(f"perf_{base}.png" if save else None))

# ------------------------------------------------------------------
# load CSV & run
# ------------------------------------------------------------------
df_long = pd.read_csv("results_longform.csv")

num_cols = ["attack","privacy","noise","step",
            "FAR","MDR","accuracy","loss"]
df_long[num_cols] = df_long[num_cols].apply(pd.to_numeric, errors="coerce")

plot_noise_sweep(df_long, attack=0, privacy=3, save=True)

