#!/usr/bin/env python3
"""
Analyze local variability in the four ETT-small datasets.

Expected files:
    ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv

Main design choices
-------------------
1. Native-resolution analysis: quantifies variability at the original sampling rate.
2. Hourly-aligned analysis: resamples every dataset to hourly frequency, allowing fair
   comparisons that are not confounded by the 15-minute vs hourly sampling interval.
3. Metrics are calculated per feature after robust scaling using statistics fitted on
   the whole series. This makes variables with different physical scales comparable.
4. Windows are defined in physical time (default: 24 h and 168 h), not row counts.
5. Moving-block bootstrap confidence intervals preserve part of the temporal dependence.

Outputs
-------
- local_variability_windows.csv: one row per window, dataset, feature and metric
- local_variability_summary.csv: median/mean/IQR and bootstrap CI by dataset/feature/metric
- local_variability_dataset_summary.csv: aggregate summary across features
- local_variability_pairwise.csv: effect sizes and block-permutation p-values
- local_variability_ranking.csv: dataset ranking by aggregate local variability
- ett_local_variability.png: publication-oriented summary plot
- ett_variability_timeseries.png: temporal evolution of aggregate rolling variability

Example
-------
python ett_local_variability.py --data-dir ./ETT-small --output-dir ./ett_variability_results
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path 
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy.stats import median_abs_deviation
except ImportError as exc:
    raise SystemExit("scipy is required: pip install scipy") from exc

import matplotlib.pyplot as plt

DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
DEFAULT_FEATURES = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
PAIRWISE = (
    ("ETTh1", "ETTh2"),
    ("ETTm1", "ETTm2"),
    ("ETTh1", "ETTm1"),
    ("ETTh2", "ETTm2"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local-variability analysis for ETT-small")
    p.add_argument("--data-dir", type=Path, default=Path("/mnt/468c9f88-157c-4fb7-b61b-c96b4fa793a2/Documents/postDoc/CTTC/IEEE_trans/datasets/etth"),
                   help="Directory containing ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv")
    p.add_argument("--output-dir", type=Path, default=Path("ett_variability_results"))
    p.add_argument("--windows-hours", type=int, nargs="+", default=[24, 168],
                   help="Physical rolling-window durations in hours (default: 24 168)")
    p.add_argument("--stride-hours", type=int, default=6,
                   help="Distance between evaluated rolling windows, in hours")
    p.add_argument("--bootstrap", type=int, default=2000,
                   help="Moving-block bootstrap repetitions")
    p.add_argument("--permutations", type=int, default=2000,
                   help="Block-permutation repetitions for pairwise comparisons")
    p.add_argument("--block-hours", type=int, default=168,
                   help="Bootstrap/permutation block length, in hours")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES))
    return p.parse_args()


def load_ett(path: Path, features: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"{path}: missing 'date' column")
    missing = sorted(set(features) - set(df.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.set_index("date").sort_index()
    if df.index.has_duplicates:
        raise ValueError(f"{path}: duplicate timestamps detected")
    out = df[features].apply(pd.to_numeric, errors="coerce")
    if out.isna().any().any():
        bad = out.columns[out.isna().any()].tolist()
        raise ValueError(f"{path}: missing/non-numeric values in {bad}")
    return out


def infer_step_hours(index: pd.DatetimeIndex) -> float:
    diffs = index.to_series().diff().dropna().dt.total_seconds().to_numpy() / 3600.0
    if len(diffs) == 0:
        raise ValueError("Not enough timestamps")
    return float(np.median(diffs))


def robust_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Median/IQR scaling; falls back to MAD, then 1 for constant channels."""
    med = df.median()
    scale = df.quantile(0.75) - df.quantile(0.25)
    mad = pd.Series(
        [median_abs_deviation(df[c], scale="normal", nan_policy="omit") for c in df.columns],
        index=df.columns,
    )
    scale = scale.mask(scale <= 1e-12, mad).mask(lambda s: s <= 1e-12, 1.0)
    return (df - med) / scale


def window_metrics(x: np.ndarray) -> dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size < 3:
        return {k: np.nan for k in ("local_std", "local_mad", "mean_abs_increment",
                                     "rms_increment", "range_iqr", "turning_rate")}
    dx = np.diff(x)
    # Turning rate: fraction of adjacent increments whose signs differ.
    signs = np.sign(dx)
    turning = np.mean(signs[1:] * signs[:-1] < 0) if len(signs) > 1 else np.nan
    q05, q95 = np.quantile(x, [0.05, 0.95])
    return {
        "local_std": float(np.std(x, ddof=1)),
        "local_mad": float(median_abs_deviation(x, scale="normal")),
        "mean_abs_increment": float(np.mean(np.abs(dx))),
        "rms_increment": float(np.sqrt(np.mean(dx ** 2))),
        "range_iqr": float(q95 - q05),
        "turning_rate": float(turning),
    }


def make_window_records(dataset: str, resolution: str, df: pd.DataFrame,
                        windows_hours: Iterable[int], stride_hours: int) -> list[dict]:
    step_h = infer_step_hours(df.index)
    scaled = robust_scale(df)
    rows: list[dict] = []
    for wh in windows_hours:
        n = max(3, int(round(wh / step_h)))
        stride = max(1, int(round(stride_hours / step_h)))
        if n > len(scaled):
            continue
        for end in range(n, len(scaled) + 1, stride):
            chunk = scaled.iloc[end - n:end]
            timestamp = chunk.index[-1]
            for feature in scaled.columns:
                metrics = window_metrics(chunk[feature].to_numpy())
                for metric, value in metrics.items():
                    rows.append({
                        "dataset": dataset, "resolution": resolution,
                        "window_hours": wh, "timestamp": timestamp,
                        "feature": feature, "metric": metric, "value": value,
                    })
    return rows


def moving_block_bootstrap_ci(values: np.ndarray, reps: int, block_len: int,
                              rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return (np.nan, np.nan)
    block_len = min(max(1, block_len), n)
    starts = np.arange(0, n - block_len + 1)
    stats = np.empty(reps)
    blocks_needed = math.ceil(n / block_len)
    for r in range(reps):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([values[s:s + block_len] for s in chosen])[:n]
        stats[r] = np.median(sample)
    return tuple(np.quantile(stats, [0.025, 0.975]))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta computed without materializing the full pairwise matrix."""
    x = np.sort(x[np.isfinite(x)])
    y = np.sort(y[np.isfinite(y)])
    if not len(x) or not len(y):
        return np.nan
    greater = np.searchsorted(y, x, side="left").sum()
    less = (len(y) - np.searchsorted(y, x, side="right")).sum()
    return float((greater - less) / (len(x) * len(y)))


def block_permutation_pvalue(x: np.ndarray, y: np.ndarray, reps: int,
                             block_len: int, rng: np.random.Generator) -> float:
    """Two-sided permutation test on median difference using contiguous blocks."""
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    obs = abs(np.median(x) - np.median(y))
    def blocks(a: np.ndarray) -> list[np.ndarray]:
        return [a[i:i + block_len] for i in range(0, len(a), block_len)]
    pool = blocks(x) + blocks(y)
    nx_blocks = len(blocks(x))
    exceed = 0
    for _ in range(reps):
        order = rng.permutation(len(pool))
        xp = np.concatenate([pool[i] for i in order[:nx_blocks]])
        yp = np.concatenate([pool[i] for i in order[nx_blocks:]])
        exceed += abs(np.median(xp) - np.median(yp)) >= obs
    return (exceed + 1) / (reps + 1)


def summarize(windows: pd.DataFrame, bootstrap: int, block_hours: int,
              stride_hours: int, rng: np.random.Generator) -> pd.DataFrame:
    keys = ["dataset", "resolution", "window_hours", "feature", "metric"]
    records = []
    block_len = max(1, int(round(block_hours / stride_hours)))
    for key, g in windows.groupby(keys, sort=True):
        values = g.sort_values("timestamp")["value"].to_numpy(float)
        lo, hi = moving_block_bootstrap_ci(values, bootstrap, block_len, rng)
        records.append(dict(zip(keys, key)) | {
            "n_windows": int(np.isfinite(values).sum()),
            "mean": float(np.nanmean(values)),
            "median": float(np.nanmedian(values)),
            "q25": float(np.nanquantile(values, 0.25)),
            "q75": float(np.nanquantile(values, 0.75)),
            "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
        })
    return pd.DataFrame(records)


def aggregate_across_features(windows: pd.DataFrame) -> pd.DataFrame:
    """Median across robustly scaled features for each time window."""
    keys = ["dataset", "resolution", "window_hours", "timestamp", "metric"]
    return (windows.groupby(keys, as_index=False)["value"].median()
            .rename(columns={"value": "aggregate_value"}))


def pairwise_tests(aggregate: pd.DataFrame, permutations: int, block_hours: int,
                   stride_hours: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    block_len = max(1, int(round(block_hours / stride_hours)))
    for (resolution, wh, metric), sub in aggregate.groupby(
            ["resolution", "window_hours", "metric"]):
        for a, b in PAIRWISE:
            xa = sub.loc[sub.dataset == a].sort_values("timestamp")["aggregate_value"].to_numpy()
            xb = sub.loc[sub.dataset == b].sort_values("timestamp")["aggregate_value"].to_numpy()
            if len(xa) == 0 or len(xb) == 0:
                continue
            diff = float(np.nanmedian(xa) - np.nanmedian(xb))
            rows.append({
                "resolution": resolution, "window_hours": wh, "metric": metric,
                "dataset_a": a, "dataset_b": b,
                "median_a": float(np.nanmedian(xa)), "median_b": float(np.nanmedian(xb)),
                "median_difference_a_minus_b": diff,
                "cliffs_delta": cliffs_delta(xa, xb),
                "block_permutation_p": block_permutation_pvalue(
                    xa, xb, permutations, block_len, rng),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        # Benjamini-Hochberg FDR correction across all reported tests.
        p = out["block_permutation_p"].to_numpy()
        order = np.argsort(p)
        ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        q = np.empty_like(ranked)
        q[order] = np.clip(ranked, 0, 1)
        out["fdr_q"] = q
    return out


def build_ranking(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    """Rank datasets using hourly-aligned 24h local_std and RMS increments."""
    chosen = dataset_summary[
        (dataset_summary.resolution == "hourly_aligned")
        & (dataset_summary.window_hours == 24)
        & (dataset_summary.metric.isin(["local_std", "rms_increment"]))
    ].copy()
    if chosen.empty:
        return pd.DataFrame()
    # Rank each metric separately, then average ranks; higher variability = rank 1.
    chosen["metric_rank"] = chosen.groupby("metric")["median"].rank(
        ascending=False, method="average")
    result = (chosen.groupby("dataset", as_index=False)
              .agg(mean_variability_rank=("metric_rank", "mean")))
    result["overall_rank"] = result["mean_variability_rank"].rank(method="dense").astype(int)
    return result.sort_values(["overall_rank", "dataset"])


def make_plots(dataset_summary: pd.DataFrame, aggregate: pd.DataFrame,
               out_dir: Path) -> None:
    plot_metrics = ["local_std", "mean_abs_increment", "rms_increment", "turning_rate"]
    data = dataset_summary[
        (dataset_summary.resolution == "hourly_aligned")
        & (dataset_summary.window_hours == 24)
        & (dataset_summary.metric.isin(plot_metrics))
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    colors = {"ETTh1": "#4C78A8", "ETTh2": "#F58518",
              "ETTm1": "#54A24B", "ETTm2": "#E45756"}
    for ax, metric in zip(axes.flat, plot_metrics):
        g = data[data.metric == metric].set_index("dataset").reindex(DATASETS)
        x = np.arange(len(DATASETS))
        ax.bar(x, g["median"], color=[colors[d] for d in DATASETS])
        ax.errorbar(x, g["median"],
                    yerr=np.vstack([g["median"] - g["bootstrap_ci_low"],
                                    g["bootstrap_ci_high"] - g["median"]]),
                    fmt="none", ecolor="black", capsize=3, linewidth=1)
        ax.set_xticks(x, DATASETS)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Robust-scaled value")
    fig.suptitle("ETT local variability (hourly aligned, 24-hour windows)")
    fig.savefig(out_dir / "ett_local_variability.png", dpi=220)
    plt.close(fig)

    ts = aggregate[(aggregate.resolution == "hourly_aligned")
                   & (aggregate.window_hours == 24)
                   & (aggregate.metric == "local_std")]
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    for dataset in DATASETS:
        g = ts[ts.dataset == dataset].sort_values("timestamp")
        ax.plot(g.timestamp, g.aggregate_value, label=dataset, linewidth=0.9, alpha=0.85)
    ax.set_title("Temporal evolution of aggregate local variability")
    ax.set_ylabel("Median local standard deviation across features")
    ax.set_xlabel("Time")
    ax.legend(ncol=4)
    fig.savefig(out_dir / "ett_variability_timeseries.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    raw = {}
    metadata = {}
    for name in DATASETS:
        path = args.data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        raw[name] = load_ett(path, args.features)
        metadata[name] = {
            "rows": len(raw[name]),
            "start": str(raw[name].index.min()),
            "end": str(raw[name].index.max()),
            "native_step_hours": infer_step_hours(raw[name].index),
        }

    records = []
    for name, df in raw.items():
        records += make_window_records(name, "native", df,
                                       args.windows_hours, args.stride_hours)
        # Resampling is essential: it separates transformer-specific variability from
        # the mechanical effect of observing m-series four times more frequently.
        hourly = df.resample("1h").mean().dropna()
        records += make_window_records(name, "hourly_aligned", hourly,
                                       args.windows_hours, args.stride_hours)

    windows = pd.DataFrame(records)
    windows.to_csv(args.output_dir / "local_variability_windows.csv", index=False)

    summary = summarize(windows, args.bootstrap, args.block_hours,
                        args.stride_hours, rng)
    summary.to_csv(args.output_dir / "local_variability_summary.csv", index=False)

    aggregate = aggregate_across_features(windows)
    dataset_summary = summarize(
        aggregate.rename(columns={"aggregate_value": "value"}).assign(feature="ALL_MEDIAN"),
        args.bootstrap, args.block_hours, args.stride_hours, rng)
    dataset_summary.to_csv(args.output_dir / "local_variability_dataset_summary.csv", index=False)

    pairwise = pairwise_tests(aggregate, args.permutations, args.block_hours,
                              args.stride_hours, rng)
    pairwise.to_csv(args.output_dir / "local_variability_pairwise.csv", index=False)

    ranking = build_ranking(dataset_summary)
    ranking.to_csv(args.output_dir / "local_variability_ranking.csv", index=False)
    make_plots(dataset_summary, aggregate, args.output_dir)

    with open(args.output_dir / "analysis_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"arguments": vars(args) | { 
                      "data_dir": str(args.data_dir), "output_dir": str(args.output_dir)},
                   "datasets": metadata}, f, indent=2)

    print(f"Analysis completed. Results written to: {args.output_dir.resolve()}")
    if not ranking.empty:
        print("\nHourly-aligned variability ranking (24 h windows):")
        print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
