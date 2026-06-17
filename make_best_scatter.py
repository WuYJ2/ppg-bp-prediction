from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
data = np.load(root / "cnn_output" / "domain_adaptation" / "fewshot_64_predictions.npz")
y_true = data["y_true"]
y_pred = data["feature_da_pred"]

out = root / "reports" / "figures" / "best_feature_da_64_scatter_polished.png"
out.parent.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
for ax, idx, name in zip(axes, [0, 1], ["SBP", "DBP"]):
    yt = y_true[:, idx]
    yp = y_pred[:, idx]
    err = yp - yt
    mae = np.mean(np.abs(err))
    std = np.std(err)
    r = np.corrcoef(yt, yp)[0, 1]

    ax.scatter(yt, yp, s=28, alpha=0.72, color="#1f77b4", edgecolors="white", linewidths=0.35)
    lo = min(float(yt.min()), float(yp.min()))
    hi = max(float(yt.max()), float(yp.max()))
    pad = (hi - lo) * 0.08
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#222222", linestyle="--", linewidth=1.2, label="Ideal")
    coef = np.polyfit(yt, yp, 1)
    xs = np.linspace(lo, hi, 100)
    ax.plot(xs, np.polyval(coef, xs), color="#d62728", linewidth=1.8, label="Fit")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{name} Prediction", fontsize=13, weight="bold")
    ax.set_xlabel(f"True {name} (mmHg)")
    ax.set_ylabel(f"Predicted {name} (mmHg)")
    ax.text(
        0.04,
        0.96,
        f"MAE = {mae:.2f} mmHg\nSTD = {std:.2f} mmHg\nr = {r:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=True)

fig.suptitle("Feature-level Domain Adaptation (64-shot) on Self-built Test Set", fontsize=15, weight="bold")
fig.tight_layout()
fig.savefig(out, dpi=220)
plt.close(fig)
print(out)
