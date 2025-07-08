#!/usr/bin/env python3
"""
fl_metrics_plot.py – Robust visualisation of FAR / MDR and other metrics
-----------------------------------------------------------------------
This script now copes with **three** storage formats you may have used:

1. **Single pickle** containing a *list* of `result_dict` runs.
2. **Pickle stream** – many pickles concatenated in one file (your first idea).
3. **ND‑JSON text** – one JSON object per line (the `{` at byte‑0 error).

Usage (unchanged):
    python fl_metrics_plot.py --file results.pkl --attack 0 --privacy 3 \
                              --param noise_STD --save

It still produces median‑with‑IQR plots for FAR & MDR and median lines for the
other metrics.
"""

import argparse
import os
import pickle
import json
from pathlib import Path
from datetime import datetime  # kept for completeness, not used directly

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

################################################################################
# 1.  universal loader – handles pickle list, pickle stream, or ND‑JSON
################################################################################

def load_results(path: str | os.PathLike):
    """Return a list[dict] regardless of how *path* was saved.

    Tries, in order:
        1. single pickle (list or dict)
        2. stream of pickles
        3. newline‑delimited JSON (utf‑8)
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # --- first try: load *entire* file as one pickle ------------------------
    data = path.read_bytes()
    try:
        obj = pickle.loads(data)
    except pickle.UnpicklingError:
        obj = None

    if obj is not None:
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]

    # --- second try: iterate through a pickle *stream* ----------------------
    blocks: list[dict] = []
    with path.open("rb") as fh:
        while True:
            try:
                blocks.append(pickle.load(fh))
            except EOFError:
                break
            except pickle.UnpicklingError:
                blocks.clear()  # not a stream after all
                break
    if blocks:
        return blocks

    # --- third try: ND‑JSON --------------------------------------------------
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise RuntimeError(
            "File is not a valid pickle and could not be decoded as UTF‑8 text"
        ) from e

    blocks = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            blocks.append(json.loads(ln))
        except json.JSONDecodeError:
            raise RuntimeError(
                "File looked like JSON but a line failed to parse: " + ln[:80]
            )
    if not blocks:
        raise RuntimeError(
            "File did not match any known format (pickle, stream, ND‑JSON)."
        )
    return blocks

################################################################################
# 2.  tidy DataFrame builder (unchanged except for minor naming)
################################################################################

def _safe_step(lst, i):
    return lst[i] if lst and i < len(lst) else None


def build_long_df(blocks: list[dict]):
    rows: list[dict] = []

    fixed_keys = {
        "timestamp",
        "privacy",
        "attack",
        "FAR",
        "MDR",
        "Error",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "test_loss2",
        "test_accuracy2",
    }

    for blk in blocks:
        meta = {
            "timestamp": blk.get("timestamp"),
            "privacy": blk.get("privacy"),
            "attack": blk.get("attack"),
            **{k: blk.get(k) for k in blk.keys() if k not in fixed_keys},
        }

        n_steps = max(len(blk.get(k, [])) for k in fixed_keys)

        for step in range(n_steps):
            rows.append(
                {
                    **meta,
                    "step": step,
                    "FAR": _safe_step(blk.get("FAR", []), step),
                    "MDR": _safe_step(blk.get("MDR", []), step),
                    "Error": _safe_step(blk.get("Error", []), step),
                    "train_loss": _safe_step(blk.get("train_loss", []), step),
                    "train_accuracy": _safe_step(blk.get("train_accuracy", []), step),
                    "test_loss": _safe_step(blk.get("test_loss", []), step),
                    "test_accuracy": _safe_step(blk.get("test_accuracy", []), step),
                    "test_loss2": _safe_step(blk.get("test_loss2", []), step),
                    "test_accuracy2": _safe_step(
                        blk.get("test_accuracy2", []), step
                    ),
                }
            )
    return pd.DataFrame(rows)

################################################################################
# 3.  summary tables (median, q25, q75) – unchanged
################################################################################

def _summary_tables(subset: pd.DataFrame, metric_cols: list[str], param_name: str):
    grp = subset[metric_cols].groupby(level=[param_name, "step"])
    return (
        grp.median(numeric_only=True),
        grp.quantile(0.25),
        grp.quantile(0.75),
    )

################################################################################
# 4.  plotting helper – unchanged logic
################################################################################

def _plot_metrics(tbl_mid, tbl_q25, tbl_q75, metrics, title, fname=None, param_level="param"):
    n = len(metrics)
    fig, axs = plt.subplots(1, n, figsize=(6 * n, 4), sharex=True)
    if n == 1:
        axs = [axs]

    sweep_vals = sorted(tbl_mid.index.get_level_values(param_level).unique())
    markers = ["o", "s", "d", "^", "v", "<", ">", "p", "*", "h", "+", "x"]
    marker_spacing = 34

    for idx, val in enumerate(sweep_vals):
        mid = tbl_mid.xs(val, level=param_level)
        q25 = tbl_q25.xs(val, level=param_level)
        q75 = tbl_q75.xs(val, level=param_level)

        marker = markers[idx % len(markers)]
        label = f"{param_level}={val}"
        offset = (idx * 15) % marker_spacing
        markevery = list(range(offset, len(mid), marker_spacing))

        for m, ax in zip(metrics, axs):
            if m not in mid.columns:
                continue
            ax.plot(mid.index, mid[m], marker=marker, linestyle="-", markevery=markevery, markersize=5, label=label)
            if m in {"FAR", "MDR"}:
                ax.fill_between(mid.index, q25[m], q75[m], alpha=0.25)

    for m, ax in zip(metrics, axs):
        ax.set_ylabel(m)
        ax.grid(True, alpha=0.3)
    axs[-1].set_xlabel("Iteration"); axs[0].set_xlabel("Iteration")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1, 0.98))
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])

    if fname:
        os.makedirs(Path(fname).parent, exist_ok=True)
        fig.savefig(fname, dpi=300)
        print("saved ➜", fname)
    else:
        plt.show()

################################################################################
# 5.  high‑level plotting orchestrator – unchanged
################################################################################

def plot_sweep(df_long: pd.DataFrame, attack: int, privacy: int, param_name: str, save: bool = False, out_dir: str = "plots"):
    subset = df_long.query("attack == @attack and privacy == @privacy").set_index([param_name, "step"]).sort_index()
    if subset.empty:
        raise ValueError(f"No rows for attack={attack}, privacy={privacy}")

    metric_cols = [
        "FAR", "MDR", "Error", "train_accuracy", "train_loss", "test_accuracy", "test_loss", "test_accuracy2", "test_loss2",
    ]
    mid, q25, q75 = _summary_tables(subset, metric_cols, param_name)

    metric_groups = [
        (["FAR", "MDR"], "Detection metrics"),
        (["train_accuracy", "train_loss"], "Training performance"),
        (["test_accuracy", "test_loss"], "Test1 performance"),
        (["test_accuracy2", "test_loss2"], "Test2 performance"),
        (["Error"], "Error over iterations"),
    ]

    for metrics, desc in metric_groups:
        fname = None
        if save:
            fname = os.path.join(out_dir, f"{desc.replace(' ', '_').lower()}_{param_name}_attack{attack}_privacy{privacy}.png")
        _plot_metrics(mid, q25, q75, metrics, f"{desc} | attack={attack} privacy={privacy} {param_name}", fname, param_name)

################################################################################
# 6.  CLI
################################################################################

def _parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--file", default="results.pkl", help="Metrics file – pickle or ND‑JSON")
    p.add_argument("--attack", type=int, default=0, help="Attack id to filter")
    p.add_argument("--privacy", type=int, default=3, help="Privacy id to filter")
    p.add_argument("--param", default="noise_STD", help="Hyper‑parameter column name")
    p.add_argument("--save", action="store_true", help="Save PNGs instead of interactive display")
    p.add_argument("--out_dir", default="plots", help="Destination folder for PNGs (with --save)")
    return p.parse_args()


def main():
    args = _parse_args()
    blocks = load_results(args.file)
    print(f"Loaded {len(blocks)} runs from '{args.file}'")
    df_long = build_long_df(blocks)
    plot_sweep(df_long, args.attack, args.privacy, args.param, args.save, args.out_dir)


if __name__ == "__main__":
    main()
