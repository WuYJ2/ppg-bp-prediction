"""
绘制 64-shot 特征级域适应最优结果的精修散点图。
单图: 左 SBP, 右 DBP, 含 MAE/STD/r 标注 + 理想对角线 + 拟合线。
"""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
RESULT_DIR = root / "cnn_output" / "domain_adaptation"

# 自动找最大的预测文件 (优先 Feature DA 结果)
pred_files = sorted(RESULT_DIR.glob("fewshot_*_predictions.npz"))
if not pred_files:
    raise FileNotFoundError(f"{RESULT_DIR} 中未找到预测文件, 请先运行 domain_adaptation_experiments.py")

def _shot(p):
    try: return int(p.stem.split("_")[1])
    except: return 0
best_file = max(pred_files, key=_shot)
shot = _shot(best_file)
print(f'使用 {shot}-shot 预测数据: {best_file.name}')

data = np.load(best_file)
y_true = data["y_true"]                     # 真实 SBP/DBP: (N, 2)
y_pred = data["feature_da_pred"]            # Feature DA 预测值

out = root / "reports" / "figures" / f"best_feature_da_{shot}shot_scatter.png"
out.parent.mkdir(parents=True, exist_ok=True)

# ---- 左 SBP + 右 DBP 散点图 ----
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
for ax, idx, name in zip(axes, [0, 1], ["SBP", "DBP"]):
    yt = y_true[:, idx]                     # 真实值
    yp = y_pred[:, idx]                     # 预测值
    err = yp - yt
    mae = np.mean(np.abs(err))              # 平均绝对误差
    std = np.std(err)                       # 误差标准差
    r = np.corrcoef(yt, yp)[0, 1]          # Pearson 相关系数

    # 散点: 蓝色半透明
    ax.scatter(yt, yp, s=28, alpha=0.72, color="#1f77b4", edgecolors="white", linewidths=0.35)
    lo = min(float(yt.min()), float(yp.min()))
    hi = max(float(yt.max()), float(yp.max()))
    pad = (hi - lo) * 0.08
    # 黑色虚线: y=x 理想对角线
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#222222", linestyle="--", linewidth=1.2, label="Ideal")
    # 红色实线: 最小二乘拟合
    coef = np.polyfit(yt, yp, 1)
    xs = np.linspace(lo, hi, 100)
    ax.plot(xs, np.polyval(coef, xs), color="#d62728", linewidth=1.8, label="Fit")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{name} Prediction", fontsize=13, weight="bold")
    ax.set_xlabel(f"True {name} (mmHg)")
    ax.set_ylabel(f"Predicted {name} (mmHg)")
    # 左上角: MAE / STD / r 标注框
    ax.text(
        0.04, 0.96,
        f"MAE = {mae:.2f} mmHg\nSTD = {std:.2f} mmHg\nr = {r:.3f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=True)

fig.suptitle(f"Feature-level Domain Adaptation ({shot}-shot) on Self-built Test Set", fontsize=15, weight="bold")
fig.tight_layout()
fig.savefig(out, dpi=220)
plt.close(fig)
print(out)
