"""
evaluate_model.py — 评估 1D-ResNet 模型在测试集上的血压预测性能
输出: SBP/DBP 预测值, MAE/STD, Bland-Altman 图, 相关性散点图, 对比柱状图

用法: python evaluate_model.py
支持 eval_mode: 'base' | 'finetuned' | 'both'
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import (
    ResNet1D, load_public_dataset,
    preprocess_ppg, apply_normalize_3ch
)

# ============================================================
# 配置
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DATA_PATH = os.path.join(SCRIPT_DIR, '数据集', '1、公开数据集')
MODEL_SAVE_PATH  = os.path.join(SCRIPT_DIR, 'models')
OUTPUT_PATH      = os.path.join(SCRIPT_DIR, 'cnn_output')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 评估模式: 'base' | 'finetuned' | 'both'
EVAL_MODE = 'both'

def compute_metrics(y_true, y_pred, name=''):
    """计算 SBP/DBP 的 MAE 和 STD"""
    err_sbp = y_pred[:, 0] - y_true[:, 0]
    err_dbp = y_pred[:, 1] - y_true[:, 1]
    result = {
        'SBP_MAE': float(np.mean(np.abs(err_sbp))),
        'SBP_STD': float(np.std(err_sbp)),
        'DBP_MAE': float(np.mean(np.abs(err_dbp))),
        'DBP_STD': float(np.std(err_dbp)),
    }
    if name:
        print(f'  [{name}] SBP: MAE={result["SBP_MAE"]:.2f}, STD={result["SBP_STD"]:.2f} mmHg')
        print(f'  [{name}] DBP: MAE={result["DBP_MAE"]:.2f}, STD={result["DBP_STD"]:.2f} mmHg')
    return result, err_sbp, err_dbp

def predict(model, X):
    """批量预测"""
    ds = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    preds = []
    with torch.no_grad():
        for (x_batch,) in loader:
            preds.append(model(x_batch.to(DEVICE)).cpu().numpy())
    return np.vstack(preds)

def plot_evaluation(y_true, y_pred, name, output_path):
    """为 SBP/DBP 生成 Bland-Altman 图和相关性散点图"""
    ds_label = 'Public' if 'public' in name else 'Self-built'
    bp_names = ['SBP', 'DBP']

    # --- Bland-Altman ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for bp_idx, bp_name in enumerate(bp_names):
        ax = axes[bp_idx]
        diff = y_pred[:, bp_idx] - y_true[:, bp_idx]
        mean_val = (y_true[:, bp_idx] + y_pred[:, bp_idx]) / 2
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)

        ax.scatter(mean_val, diff, s=8, alpha=0.4, c='blue', edgecolors='none')
        ax.axhline(mean_diff, color='red', lw=1.5)
        ax.axhline(mean_diff + 1.96 * std_diff, color='red', ls='--', lw=1)
        ax.axhline(mean_diff - 1.96 * std_diff, color='red', ls='--', lw=1)
        ax.set_xlabel('Mean (mmHg)')
        ax.set_ylabel('Difference (mmHg)')
        ax.set_title(f'{bp_name} {ds_label} Bland-Altman')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_path, f'BA_{name}.png'), dpi=150)
    plt.close(fig)

    # --- 相关性散点图 ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for bp_idx, bp_name in enumerate(bp_names):
        ax = axes[bp_idx]
        yt = y_true[:, bp_idx]
        yp = y_pred[:, bp_idx]

        ax.scatter(yt, yp, s=8, alpha=0.4, c='blue', edgecolors='none')

        mn_val = min(yt.min(), yp.min())
        mx_val = max(yt.max(), yp.max())
        rng = mx_val - mn_val
        ax.plot([mn_val - 0.05 * rng, mx_val + 0.05 * rng],
                [mn_val - 0.05 * rng, mx_val + 0.05 * rng], 'k--', lw=1.2)

        coeffs = np.polyfit(yt, yp, 1)
        x_fit = np.linspace(mn_val, mx_val, 100)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), 'r-', lw=1.5)

        r = np.corrcoef(yt, yp)[0, 1]
        ax.set_xlabel(f'True {bp_name} (mmHg)')
        ax.set_ylabel(f'Predicted {bp_name} (mmHg)')
        ax.set_title(f'{bp_name} {ds_label} Correlation (r={r:.3f})')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_path, f'Corr_{name}.png'), dpi=150)
    plt.close(fig)

    # --- 预测值 vs 真值折线图 (前 80 样本) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    n_show = min(80, len(y_true))
    for bp_idx, bp_name in enumerate(bp_names):
        ax = axes[bp_idx]
        ax.plot(range(n_show), y_true[:n_show, bp_idx], 'b-', lw=1, label='True')
        ax.plot(range(n_show), y_pred[:n_show, bp_idx], 'r-', lw=1, label='Predicted')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(f'{bp_name} (mmHg)')
        ax.set_title(f'{bp_name} {ds_label} Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_path, f'Line_{name}.png'), dpi=150)
    plt.close(fig)


FS      = 125
SIG_LEN = 2048

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {DEVICE}, 评估模式: {EVAL_MODE}')

# ============================================================
# 加载归一化参数
# ============================================================
norm_data = np.load(os.path.join(MODEL_SAVE_PATH, 'norm_params.npz'))
ch_mean = norm_data['ch_mean']
ch_std  = norm_data['ch_std']
bp_mean = norm_data['bp_mean']
bp_std  = norm_data['bp_std']
norm_stats = {'ch_mean': ch_mean, 'ch_std': ch_std}

# ============================================================
# 加载模型
# ============================================================
models = {}

if EVAL_MODE in ('base', 'both'):
    path = os.path.join(MODEL_SAVE_PATH, 'base_model_best.pth')
    if not os.path.exists(path):
        path = os.path.join(MODEL_SAVE_PATH, 'base_model_final.pth')
    if os.path.exists(path):
        m = ResNet1D(in_channels=3, num_outputs=2).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        m.eval()
        models['base'] = m
        print('基础模型已加载')
    else:
        print('基础模型未找到，跳过')

if EVAL_MODE in ('finetuned', 'both'):
    path = os.path.join(MODEL_SAVE_PATH, 'finetuned_model_best.pth')
    if not os.path.exists(path):
        path = os.path.join(MODEL_SAVE_PATH, 'finetuned_model_final.pth')
    if os.path.exists(path):
        m = ResNet1D(in_channels=3, num_outputs=2).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        m.eval()
        models['finetuned'] = m
        print('微调模型已加载')
    else:
        print('微调模型未找到，跳过')

if not models:
    print('错误: 没有找到任何模型文件，请先运行 train_base_model.py')
    sys.exit(1)

# ============================================================
# 加载测试数据
# ============================================================
print('=== 加载测试数据 ===')

data_pub = load_public_dataset(PUBLIC_DATA_PATH)
X_pub_raw  = data_pub['test']['ppg']
Y_pub_sbp  = data_pub['test']['sbp']
Y_pub_dbp  = data_pub['test']['dbp']

# 预处理
X_pub_3ch  = preprocess_ppg(X_pub_raw, FS)
X_pub_3ch  = apply_normalize_3ch(X_pub_3ch, norm_stats)

print(f'测试集样本数: {len(X_pub_3ch)}')

# ============================================================
# 评估
# ============================================================

all_results = {}

for model_name, model in models.items():
    print(f'\n========== 评估: {model_name} 模型 ==========')

    y_pred = predict(model, X_pub_3ch)
    y_pred = y_pred * bp_std + bp_mean
    y_true = np.column_stack([Y_pub_sbp, Y_pub_dbp])

    pub_res, _, _ = compute_metrics(y_true, y_pred, '测试集')

    all_results[model_name] = {
        'res': pub_res, 'y_true': y_true, 'y_pred': y_pred,
    }

    plot_evaluation(y_true, y_pred, f'{model_name}_test', OUTPUT_PATH)

    np.savez(os.path.join(OUTPUT_PATH, f'predictions_{model_name}.npz'),
             y_true=y_true, y_pred=y_pred, res=pub_res)

# ============================================================
# 对比汇总 (仅在有两个模型时)
# ============================================================
if len(models) == 2:
    print('\n========== 模型对比 ==========')
    print(f'{"模型":<12} {"SBP_MAE":<10} {"SBP_STD":<10} {"DBP_MAE":<10} {"DBP_STD":<10}')
    print('-' * 52)

    model_names = list(models.keys())

    for mn in model_names:
        r = all_results[mn]['res']
        print(f'{mn:<12} {r["SBP_MAE"]:<10.2f} {r["SBP_STD"]:<10.2f} '
              f'{r["DBP_MAE"]:<10.2f} {r["DBP_STD"]:<10.2f}')

    # 对比柱状图
    metrics_keys = ['SBP_MAE', 'SBP_STD', 'DBP_MAE', 'DBP_STD']
    metric_labels = ['SBP MAE', 'SBP STD', 'DBP MAE', 'DBP STD']

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    colors = ['#2196F3', '#FF5722']

    for k, (mk, ml) in enumerate(zip(metrics_keys, metric_labels)):
        ax = axes[k // 2][k % 2]
        width = 0.35
        for mi, mn in enumerate(model_names):
            r = all_results[mn]['res']
            ax.bar(mi * width, r[mk], width, label=mn, color=colors[mi])
        ax.set_xticks([])
        ax.set_ylabel('mmHg')
        ax.set_title(ml)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Base vs Fine-tuned Model Comparison', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_PATH, 'model_comparison.png'), dpi=150)
    plt.close(fig)

print(f'\n=== 评估完成，结果保存至 {OUTPUT_PATH} ===')


# ============================================================
# 画图函数
# ============================================================

