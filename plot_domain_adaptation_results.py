"""Plot few-shot domain adaptation experiment results."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "cnn_output" / "domain_adaptation"
REPORT_IMG_DIR = ROOT / "reports" / "figures"
REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)

METHOD_LABELS = {
    "base": "Base",
    "feature_da": "Feature DA",
    "lwf": "LwF",
}


def load_results() -> dict[int, dict]:
    results = {}
    for shot in (16, 32, 64):
        with (RESULT_DIR / f"fewshot_{shot}_results.json").open("r", encoding="utf-8") as f:
            results[shot] = json.load(f)
    return results


def plot_metric_bars(results: dict[int, dict]) -> Path:
    metrics = ["SBP_MAE", "SBP_STD", "DBP_MAE", "DBP_STD"]
    titles = ["SBP MAE", "SBP STD", "DBP MAE", "DBP STD"]
    shots = [16, 32, 64]
    methods = ["base", "feature_da", "lwf"]
    colors = {"base": "#6c757d", "feature_da": "#1f77b4", "lwf": "#ff7f0e"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric, title in zip(axes.ravel(), metrics, titles):
        x = np.arange(len(shots))
        width = 0.24
        for i, method in enumerate(methods):
            vals = [results[shot][method][metric] for shot in shots]
            ax.bar(x + (i - 1) * width, vals, width, label=METHOD_LABELS[method], color=colors[method])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{shot}-shot" for shot in shots])
        ax.set_ylabel("mmHg")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(loc="best")
    fig.suptitle("Few-shot Domain Adaptation Metrics", fontsize=14)
    fig.tight_layout()
    path = REPORT_IMG_DIR / "fewshot_metric_bars.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_mae_trends(results: dict[int, dict]) -> Path:
    shots = [16, 32, 64]
    methods = ["base", "feature_da", "lwf"]
    colors = {"base": "#6c757d", "feature_da": "#1f77b4", "lwf": "#ff7f0e"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title in zip(axes, ["SBP_MAE", "DBP_MAE"], ["SBP MAE Trend", "DBP MAE Trend"]):
        for method in methods:
            vals = [results[shot][method][metric] for shot in shots]
            ax.plot(shots, vals, marker="o", linewidth=2, label=METHOD_LABELS[method], color=colors[method])
        ax.set_title(title)
        ax.set_xlabel("Target-domain training samples")
        ax.set_ylabel("MAE (mmHg)")
        ax.set_xticks(shots)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best")
    fig.tight_layout()
    path = REPORT_IMG_DIR / "fewshot_mae_trends.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_best_scatter() -> Path:
    data = np.load(RESULT_DIR / "fewshot_64_predictions.npz")
    y_true = data["y_true"]
    preds = {
        "Base": data["base_pred"],
        "Feature DA": data["feature_da_pred"],
        "LwF": data["lwf_pred"],
    }

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for col, (name, y_pred) in enumerate(preds.items()):
        for row, bp_idx in enumerate([0, 1]):
            ax = axes[row, col]
            bp_name = "SBP" if bp_idx == 0 else "DBP"
            yt = y_true[:, bp_idx]
            yp = y_pred[:, bp_idx]
            ax.scatter(yt, yp, s=18, alpha=0.65, color="#1f77b4", edgecolors="none")
            lo = min(float(yt.min()), float(yp.min()))
            hi = max(float(yt.max()), float(yp.max()))
            pad = (hi - lo) * 0.05
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1)
            r = np.corrcoef(yt, yp)[0, 1]
            ax.set_title(f"{name} {bp_name} (r={r:.3f})")
            ax.set_xlabel(f"True {bp_name} (mmHg)")
            ax.set_ylabel(f"Predicted {bp_name} (mmHg)")
            ax.grid(alpha=0.25)
    fig.suptitle("64-shot Predictions on Self-built Test Set", fontsize=14)
    fig.tight_layout()
    path = REPORT_IMG_DIR / "fewshot_64_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_method_metric_bars(results: dict[int, dict], method: str) -> Path:
    metrics = ["SBP_MAE", "SBP_STD", "DBP_MAE", "DBP_STD"]
    titles = ["SBP MAE", "SBP STD", "DBP MAE", "DBP STD"]
    shots = [16, 32, 64]
    colors = {"base": "#6c757d", method: "#1f77b4" if method == "feature_da" else "#ff7f0e"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, metric, title in zip(axes.ravel(), metrics, titles):
        x = np.arange(len(shots))
        width = 0.32
        for i, name in enumerate(["base", method]):
            vals = [results[shot][name][metric] for shot in shots]
            ax.bar(x + (i - 0.5) * width, vals, width, label=METHOD_LABELS[name], color=colors[name])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{shot}-shot" for shot in shots])
        ax.set_ylabel("mmHg")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(loc="best")
    fig.suptitle(f"{METHOD_LABELS[method]} Metrics vs Base", fontsize=14)
    fig.tight_layout()
    path = REPORT_IMG_DIR / f"{method}_metric_bars.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_method_mae_trends(results: dict[int, dict], method: str) -> Path:
    shots = [16, 32, 64]
    colors = {"base": "#6c757d", method: "#1f77b4" if method == "feature_da" else "#ff7f0e"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax, metric, title in zip(axes, ["SBP_MAE", "DBP_MAE"], ["SBP MAE", "DBP MAE"]):
        for name in ["base", method]:
            vals = [results[shot][name][metric] for shot in shots]
            ax.plot(shots, vals, marker="o", linewidth=2, label=METHOD_LABELS[name], color=colors[name])
        ax.set_title(f"{METHOD_LABELS[method]} {title} Trend")
        ax.set_xlabel("Target-domain training samples")
        ax.set_ylabel("MAE (mmHg)")
        ax.set_xticks(shots)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best")
    fig.tight_layout()
    path = REPORT_IMG_DIR / f"{method}_mae_trends.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_method_scatter(method: str) -> Path:
    data = np.load(RESULT_DIR / "fewshot_64_predictions.npz")
    y_true = data["y_true"]
    base_pred = data["base_pred"]
    method_pred = data["feature_da_pred" if method == "feature_da" else "lwf_pred"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for col, (name, y_pred) in enumerate([("Base", base_pred), (METHOD_LABELS[method], method_pred)]):
        for row, bp_idx in enumerate([0, 1]):
            ax = axes[row, col]
            bp_name = "SBP" if bp_idx == 0 else "DBP"
            yt = y_true[:, bp_idx]
            yp = y_pred[:, bp_idx]
            ax.scatter(yt, yp, s=18, alpha=0.65, color=colors_for_method(method), edgecolors="none")
            lo = min(float(yt.min()), float(yp.min()))
            hi = max(float(yt.max()), float(yp.max()))
            pad = (hi - lo) * 0.05
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1)
            r = np.corrcoef(yt, yp)[0, 1]
            ax.set_title(f"{name} {bp_name} (r={r:.3f})")
            ax.set_xlabel(f"True {bp_name} (mmHg)")
            ax.set_ylabel(f"Predicted {bp_name} (mmHg)")
            ax.grid(alpha=0.25)
    fig.suptitle(f"64-shot {METHOD_LABELS[method]} Predictions vs Base", fontsize=14)
    fig.tight_layout()
    path = REPORT_IMG_DIR / f"{method}_64_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def colors_for_method(method: str) -> str:
    return "#1f77b4" if method == "feature_da" else "#ff7f0e"


def main() -> None:
    results = load_results()
    paths = [plot_metric_bars(results), plot_mae_trends(results), plot_best_scatter()]
    for method in ("feature_da", "lwf"):
        paths.extend(
            [
                plot_method_metric_bars(results, method),
                plot_method_mae_trends(results, method),
                plot_method_scatter(method),
            ]
        )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
