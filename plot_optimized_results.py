from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "cnn_output" / "domain_adaptation"
FIG_DIR = ROOT / "reports" / "figures_optimized"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SHOTS = [64, 128, 253]
METHODS = {"feature_da": "Feature DA + Calibration", "lwf": "LwF + Calibration"}
COLORS = {"feature_da": "#1f77b4", "lwf": "#ff7f0e", "base": "#6c757d"}


def load_results():
    out = {}
    for shot in SHOTS:
        with open(RESULT_DIR / f"fewshot_{shot}_calibrated_results.json", "r", encoding="utf-8") as f:
            out[shot] = json.load(f)
    return out


def plot_method(method, results):
    metrics = ["SBP_MAE", "SBP_STD", "DBP_MAE", "DBP_STD"]
    titles = ["SBP MAE", "SBP STD", "DBP MAE", "DBP STD"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, metric, title in zip(axes.ravel(), metrics, titles):
        x = np.arange(len(SHOTS))
        width = 0.32
        for i, name in enumerate(["base", method]):
            vals = [results[s][name][metric] for s in SHOTS]
            label = "Base + Calibration" if name == "base" else METHODS[method]
            ax.bar(x + (i - 0.5) * width, vals, width, label=label, color=COLORS[name])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}-shot" for s in SHOTS])
        ax.set_ylabel("mmHg")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(loc="best")
    fig.suptitle(f"{METHODS[method]} Metrics", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{method}_optimized_bars.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax, metric, title in zip(axes, ["SBP_MAE", "DBP_MAE"], ["SBP MAE", "DBP MAE"]):
        for name in ["base", method]:
            vals = [results[s][name][metric] for s in SHOTS]
            label = "Base + Calibration" if name == "base" else METHODS[method]
            ax.plot(SHOTS, vals, marker="o", linewidth=2, label=label, color=COLORS[name])
        ax.set_title(f"{title} Trend")
        ax.set_xlabel("Target-domain training samples")
        ax.set_ylabel("MAE (mmHg)")
        ax.set_xticks(SHOTS)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{method}_optimized_trend.png", dpi=220)
    plt.close(fig)

    pred_key = "feature_da_pred" if method == "feature_da" else "lwf_pred"
    data = np.load(RESULT_DIR / "fewshot_253_calibrated_predictions.npz")
    y_true = data["y_true"]
    y_pred = data[pred_key]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    for ax, idx, bp in zip(axes, [0, 1], ["SBP", "DBP"]):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        err = yp - yt
        mae = np.mean(np.abs(err))
        std = np.std(err)
        r = np.corrcoef(yt, yp)[0, 1]
        ax.scatter(yt, yp, s=28, alpha=0.72, color=COLORS[method], edgecolors="white", linewidths=0.35)
        lo = min(float(yt.min()), float(yp.min()))
        hi = max(float(yt.max()), float(yp.max()))
        pad = (hi - lo) * 0.08
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1.2, label="Ideal")
        coef = np.polyfit(yt, yp, 1)
        xs = np.linspace(lo, hi, 100)
        ax.plot(xs, np.polyval(coef, xs), color="#d62728", linewidth=1.8, label="Fit")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{bp} Prediction", fontsize=13, weight="bold")
        ax.set_xlabel(f"True {bp} (mmHg)")
        ax.set_ylabel(f"Predicted {bp} (mmHg)")
        ax.text(0.04, 0.96, f"MAE = {mae:.2f} mmHg\nSTD = {std:.2f} mmHg\nr = {r:.3f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92))
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right")
    fig.suptitle(f"{METHODS[method]} on Self-built Test Set (253-shot)", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{method}_optimized_scatter.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    results = load_results()
    for method in METHODS:
        plot_method(method, results)
    for p in sorted(FIG_DIR.glob("*.png")):
        print(p)
