"""
train_base_model.py — 使用公开数据集训练 1D-ResNet 基础模型
输入: PPG + 一阶导数 + 二阶导数 (3 通道)
输出: SBP / DBP (双输出回归)

依赖: torch, numpy, scipy, model.py
用法: python train_base_model.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

# 导入共享模块
from model import (
    ResNet1D, load_public_dataset,
    preprocess_ppg, normalize_3ch, apply_normalize_3ch, normalize_bp
)

# ============================================================
# 配置
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DATA_PATH = os.path.join(SCRIPT_DIR, '数据集', '1、公开数据集')
MODEL_SAVE_PATH  = os.path.join(SCRIPT_DIR, 'models')
OUTPUT_PATH      = os.path.join(SCRIPT_DIR, 'cnn_output')
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 训练超参数
NUM_EPOCHS          = 80
BATCH_SIZE          = 64
LEARNING_RATE       = 1e-3
LR_STEP_SIZE        = 25
LR_GAMMA            = 0.5
WEIGHT_DECAY        = 1e-4
GRAD_CLIP           = 5.0
VAL_FREQ            = 50   # 每 N 次迭代验证一次

# 信号参数
FS       = 125
SIG_LEN  = 2048
NUM_CH   = 3
NUM_OUT  = 2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {DEVICE}')

# ============================================================
# 加载数据
# ============================================================
print('=== 加载公开数据集 ===')
data = load_public_dataset(PUBLIC_DATA_PATH)

X_train_raw = data['train']['ppg']
Y_train_sbp = data['train']['sbp']
Y_train_dbp = data['train']['dbp']

# --- 应用 70/30 拆分 (只取基础训练部分) ---
split_file = os.path.join(MODEL_SAVE_PATH, 'train_split.npz')
if os.path.exists(split_file):
    split = np.load(split_file)
    base_idx = split['base_idx']
    print(f'加载拆分索引: base={len(base_idx)} 样本')
else:
    # 首次运行: 自动生成拆分
    rng = np.random.RandomState(42)
    n_total = len(Y_train_sbp)
    idx = rng.permutation(n_total)
    n_base = int(n_total * 0.7)
    base_idx = np.sort(idx[:n_base])
    ft_idx = np.sort(idx[n_base:])
    np.savez(split_file, base_idx=base_idx, ft_idx=ft_idx, seed=42, ratio=0.7)
    print(f'自动生成拆分: base={len(base_idx)}, ft={len(ft_idx)}')

X_train_raw = X_train_raw[base_idx]
Y_train_sbp = Y_train_sbp[base_idx]
Y_train_dbp = Y_train_dbp[base_idx]
print(f'基础训练样本数: {len(X_train_raw)}')
X_val_raw   = data['val']['ppg']
Y_val_sbp   = data['val']['sbp']
Y_val_dbp   = data['val']['dbp']
X_test_raw  = data['test']['ppg']
Y_test_sbp  = data['test']['sbp']
Y_test_dbp  = data['test']['dbp']

print(f'训练集: {len(X_train_raw)}, 验证集: {len(X_val_raw)}, 测试集: {len(X_test_raw)}')

# ============================================================
# 预处理
# ============================================================
print('=== 数据预处理 (计算导数 + 归一化) ===')

X_train_3ch = preprocess_ppg(X_train_raw, FS)
X_val_3ch   = preprocess_ppg(X_val_raw, FS)
X_test_3ch  = preprocess_ppg(X_test_raw, FS)

X_train_3ch, norm_stats = normalize_3ch(X_train_3ch)
X_val_3ch   = apply_normalize_3ch(X_val_3ch, norm_stats)
X_test_3ch  = apply_normalize_3ch(X_test_3ch, norm_stats)

Y_train_norm, bp_mean, bp_std = normalize_bp(Y_train_sbp, Y_train_dbp)
Y_val_norm   = (np.column_stack([Y_val_sbp, Y_val_dbp]) - bp_mean) / bp_std
Y_test_norm  = (np.column_stack([Y_test_sbp, Y_test_dbp]) - bp_mean) / bp_std

# 保存归一化参数
np.savez(os.path.join(MODEL_SAVE_PATH, 'norm_params.npz'),
         ch_mean=norm_stats['ch_mean'], ch_std=norm_stats['ch_std'],
         bp_mean=bp_mean, bp_std=bp_std)

print(f'PPG 归一化: 均值范围 [{norm_stats["ch_mean"].min():.4f}, '
      f'{norm_stats["ch_mean"].max():.4f}]')
print(f'SBP: mean={bp_mean[0,0]:.2f}, std={bp_std[0,0]:.2f} | '
      f'DBP: mean={bp_mean[0,1]:.2f}, std={bp_std[0,1]:.2f}')

# ============================================================
# 构建 DataLoader
# ============================================================
train_dataset = TensorDataset(
    torch.from_numpy(X_train_3ch), torch.from_numpy(Y_train_norm))
val_dataset   = TensorDataset(
    torch.from_numpy(X_val_3ch), torch.from_numpy(Y_val_norm))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 完整验证集 (一次性验证用)
X_val_t  = torch.from_numpy(X_val_3ch).to(DEVICE)
Y_val_t  = torch.from_numpy(Y_val_norm).to(DEVICE)

# ============================================================
# 构建模型
# ============================================================
print('=== 构建 1D-ResNet ===')
model = ResNet1D(in_channels=NUM_CH, num_outputs=NUM_OUT).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f'可学习参数: {n_params:,}')

# ============================================================
# 训练
# ============================================================
print('=== 开始训练 ===')

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                              weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                             step_size=LR_STEP_SIZE,
                                             gamma=LR_GAMMA)

train_loss_hist = []
val_loss_hist   = []
best_val_loss   = float('inf')
best_epoch      = 0
iteration       = 0

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    t_start = time.time()

    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        # 前向
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)

        # 反向
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        iteration += 1
        epoch_loss += loss.item()

        # 定期验证
        if iteration % VAL_FREQ == 0:
            model.eval()
            with torch.no_grad():
                y_val_pred = model(X_val_t)
                val_loss = criterion(y_val_pred, Y_val_t).item()
            model.train()

            train_loss_hist.append((iteration, epoch_loss / (batch_idx + 1)))
            val_loss_hist.append((iteration, val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(),
                           os.path.join(MODEL_SAVE_PATH, 'base_model_best.pth'))
                print(f'  [Iter {iteration}] 最佳模型已保存, ValLoss={val_loss:.4f}')

    epoch_loss /= len(train_loader)
    scheduler.step()

    # epoch 末验证
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_t)
        val_loss = criterion(y_val_pred, Y_val_t).item()

    print(f'Epoch {epoch}/{NUM_EPOCHS}, Loss={epoch_loss:.4f}, '
          f'ValLoss={val_loss:.4f}, LR={scheduler.get_last_lr()[0]:.2e}, '
          f'耗时 {time.time() - t_start:.1f}s')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(model.state_dict(),
                   os.path.join(MODEL_SAVE_PATH, 'base_model_best.pth'))

print(f'最佳验证损失: {best_val_loss:.4f} (Epoch {best_epoch})')

# ============================================================
# 保存最终模型
# ============================================================
torch.save(model.state_dict(),
           os.path.join(MODEL_SAVE_PATH, 'base_model_final.pth'))
print(f'模型保存至 {MODEL_SAVE_PATH}')

# ============================================================
# 绘制训练曲线
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if train_loss_hist:
    train_arr = np.array(train_loss_hist)
    val_arr   = np.array(val_loss_hist)

    plt.figure(figsize=(8, 5))
    plt.semilogy(train_arr[:, 0], train_arr[:, 1], 'b-', lw=1, label='Training Loss')
    plt.semilogy(val_arr[:, 0], val_arr[:, 1], 'r-', lw=1, label='Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE)')
    plt.title('1D-ResNet Base Model Training Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_PATH, 'base_training_curve.png'), dpi=150)
    plt.close()
    print(f'训练曲线已保存至 {OUTPUT_PATH}')

print('=== 基础模型训练完成 ===')
