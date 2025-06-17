#!/usr/bin/env python3

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def pick_timestamp(store, ts_arg):
    """Return the chosen timestamp key (validate if user supplied one)."""
    if ts_arg:
        if ts_arg not in store:
            raise KeyError(f"Timestamp '{ts_arg}' not found in file. "
                           f"Available: {list(store)}")
        return ts_arg
    return sorted(store)[-1]


def plot_combo(combo_data, privacy_key, attack_key, out_path):
    """
    combo_data : dict  noise -> {'FAR': [...], 'MDR': [...], 'Error': [...]}
    out_path   : pathlib.Path  (png)
    """
    noise_levels = sorted(combo_data)
    markers = ["o", "s", "d", "^", "v", "<", ">", "p", "*", "h", "+", "x"]
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    base_me = 17  # markevery base

    for idx, noise in enumerate(noise_levels):
        metrics = combo_data[noise]
        label = r"$\sigma^2 = 0$" if noise == 0 else rf"$\sigma^2={noise}$"
        me = ((idx * 50) % base_me, base_me)
        m = markers[idx % len(markers)]

        axs[0].plot(metrics["FAR"], marker=m, markevery=me, label=label, ms=6)
        axs[1].plot(metrics["MDR"], marker=m, markevery=me, ms=6)
        axs[2].plot(metrics["Error"], marker=m, markevery=me, ms=6)

    axs[0].set_title("False Alarm Rate")
    axs[1].set_title("Missed Detection Rate")
    axs[2].set_title("Consensus Error")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("Iteration")

    for ax in axs:
        ax.grid(True)
    axs[0].legend()

    fig.suptitle(f"{privacy_key} • {attack_key}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot PDMM results.")
    parser.add_argument("results_pkl", type=Path,
                        help="Pickle file produced by save_results()")
    parser.add_argument("--timestamp", help="Timestamp key to plot "
                        "(default = latest)")
    parser.add_argument("--outdir", default="plots",
                        help="Output directory for figures (default: plots)")
    args = parser.parse_args(argv)

    with args.results_pkl.open("rb") as fh:
        store = pickle.load(fh)

    ts = pick_timestamp(store, args.timestamp)
    run_data = store[ts]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n_fig = 0
    for privacy_key, attacks in run_data.items():
        for attack_key, noise_dict in attacks.items():
            outfile = outdir / f"{ts}_{privacy_key}_{attack_key}.png"
            plot_combo(noise_dict, privacy_key, attack_key, outfile)
            n_fig += 1

    print(f"Generated {n_fig} figure(s) → {outdir.resolve()}")


if __name__ == "__main__":
    sys.exit(main())
