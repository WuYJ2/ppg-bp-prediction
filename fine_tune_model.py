"""
fine_tune_model.py — 蒸馏微调 1D-ResNet 模型
策略: 冻结浅层 + 新旧数据混合批 + 知识蒸馏 (软标签)

依赖: torch, numpy, model.py, train_base_model.py 输出的基础模型
用法: python fine_tune_model.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

from model import (
    ResNet1D, load_public_dataset, load_self_dataset,
    preprocess_ppg, normalize_3ch, apply_normalize_3ch, normalize_bp
)

# ============================================================
# 配置
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DATA_PATH = os.path.join(SCRIPT_DIR, '数据集', '1、公开数据集')
SELF_DATA_PATH   = os.path.join(SCRIPT_DIR, 'dataset')
MODEL_SAVE_PATH  = os.path.join(SCRIPT_DIR, 'models')
OUTPUT_PATH      = os.path.join(SCRIPT_DIR, 'cnn_output')
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 微调超参数
NUM_EPOCHS      = 50
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-5
GRAD_CLIP       = 5.0

# 蒸馏参数
DISTILL_WEIGHT = 0.4   # α: 蒸馏损失权重 (0.2-0.7)
TEMPERATURE    = 3.0   # T: 温度 (2-4)

# 新旧数据混合比
SELF_DATA_RATIO = 0.5  # 每批中自建数据占比

# 冻结层名称 (对应 ResNet1D 的子模块)
FROZEN_MODULES = ['conv1', 'bn1', 'layer1', 'layer2']

# 信号参数
FS      = 125
SIG_LEN = 2048

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {DEVICE}')
print(f'蒸馏权重 α = {DISTILL_WEIGHT}, 温度 T = {TEMPERATURE}')
print(f'冻结模块: {FROZEN_MODULES}')

# ============================================================
# 加载基础模型
# ============================================================
print('=== 加载基础模型 ===')

model = ResNet1D(in_channels=3, num_outputs=2).to(DEVICE)

best_path = os.path.join(MODEL_SAVE_PATH, 'base_model_best.pth')
if not os.path.exists(best_path):
    best_path = os.path.join(MODEL_SAVE_PATH, 'base_model_final.pth')
model.load_state_dict(torch.load(best_path, map_location=DEVICE,
                                  weights_only=True))

# 教师模型 = 冻结的基础模型
teacher = ResNet1D(in_channels=3, num_outputs=2).to(DEVICE)
teacher.load_state_dict(torch.load(best_path, map_location=DEVICE,
                                    weights_only=True))
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

print(f'基础模型已加载: {best_path}')

# 加载归一化参数
norm_data = np.load(os.path.join(MODEL_SAVE_PATH, 'norm_params.npz'))
ch_mean = norm_data['ch_mean']
ch_std  = norm_data['ch_std']
bp_mean = norm_data['bp_mean']
bp_std  = norm_data['bp_std']

norm_stats = {'ch_mean': ch_mean, 'ch_std': ch_std}

# ============================================================
# 加载公开数据 (混合批用)
# ============================================================
print('=== 加载公开数据集 (混合训练用) ===')
data_pub = load_public_dataset(PUBLIC_DATA_PATH)

X_pub_raw  = data_pub['train']['ppg']
Y_pub_sbp  = data_pub['train']['sbp']
Y_pub_dbp  = data_pub['train']['dbp']

X_pub_3ch  = preprocess_ppg(X_pub_raw, FS)
X_pub_3ch  = apply_normalize_3ch(X_pub_3ch, norm_stats)
Y_pub_norm = (np.column_stack([Y_pub_sbp, Y_pub_dbp]) - bp_mean) / bp_std

print(f'公开训练数据: {len(X_pub_3ch)} 样本')

# ============================================================
# 加载自建数据 (cell 格式)
# ============================================================
print('=== 加载自建数据集 ===')
data_self = load_self_dataset(SELF_DATA_PATH, sig_len=SIG_LEN)

X_self_raw  = data_self['train']['ppg']
Y_self_sbp  = data_self['train']['sbp']
Y_self_dbp  = data_self['train']['dbp']

X_self_3ch  = preprocess_ppg(X_self_raw, FS)
X_self_3ch  = apply_normalize_3ch(X_self_3ch, norm_stats)
Y_self_norm = (np.column_stack([Y_self_sbp, Y_self_dbp]) - bp_mean) / bp_std

print(f'自建训练数据: {len(X_self_3ch)} 样本')

# ============================================================
# 冻结浅层
# ============================================================
print('=== 冻结浅层权重 ===')

for name, param in model.named_parameters():
    should_freeze = any(name.startswith(prefix + '.') or name.startswith(prefix)
                        for prefix in FROZEN_MODULES)
    param.requires_grad = not should_freeze

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
frozen    = total - trainable
print(f'可训练参数: {trainable:,} / {total:,} (冻结: {frozen:,})')

# ============================================================
# 构建混合 DataLoader
# ============================================================
# 转为 tensor
X_pub_t  = torch.from_numpy(X_pub_3ch)
Y_pub_t  = torch.from_numpy(Y_pub_norm)
X_self_t = torch.from_numpy(X_self_3ch)
Y_self_t = torch.from_numpy(Y_self_norm)

num_pub  = len(X_pub_t)
num_self = len(X_self_t)

batch_pub  = int(BATCH_SIZE * (1 - SELF_DATA_RATIO))
batch_self = BATCH_SIZE - batch_pub

num_batches = max(num_pub // batch_pub, num_self // batch_self) + 1

print(f'混合批: 公开 {batch_pub} + 自建 {batch_self} = {BATCH_SIZE} / 批')

# ============================================================
# 微调训练
# ============================================================
print('=== 开始微调训练 (蒸馏 + 混合批) ===')

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

train_loss_hist = []
best_loss       = float('inf')

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    t_start = time.time()

    # 打乱索引
    idx_pub  = torch.randperm(num_pub)
    idx_self = torch.randperm(num_self)

    for b in range(num_batches):
        # 采样公开数据
        pub_start = (b * batch_pub) % max(1, num_pub - batch_pub + 1)
        pub_end   = min(pub_start + batch_pub, num_pub)
        pub_idx   = idx_pub[pub_start:pub_end]

        # 采样自建数据
        self_start = (b * batch_self) % max(1, num_self - batch_self + 1)
        self_end   = min(self_start + batch_self, num_self)
        self_idx   = idx_self[self_start:self_end]

        x_pub  = X_pub_t[pub_idx].to(DEVICE)
        y_pub  = Y_pub_t[pub_idx].to(DEVICE)
        x_self = X_self_t[self_idx].to(DEVICE)
        y_self = Y_self_t[self_idx].to(DEVICE)

        # === 前向传播 ===
        # 自建数据: 仅任务损失
        y_self_pred = model(x_self)
        loss_self = criterion(y_self_pred, y_self)

        # 公开数据: 任务损失 + 蒸馏损失
        y_pub_pred = model(x_pub)
        task_pub = criterion(y_pub_pred, y_pub)

        # 教师预测 (软标签, 无梯度)
        with torch.no_grad():
            y_teacher = teacher(x_pub)

        # 蒸馏损失: T^2 * MSE(pred/T, teacher/T)
        distill_pub = criterion(y_pub_pred / TEMPERATURE,
                                 y_teacher / TEMPERATURE) * (TEMPERATURE ** 2)

        # 综合损失
        loss = loss_self + (1 - DISTILL_WEIGHT) * task_pub + DISTILL_WEIGHT * distill_pub

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= max(1, num_batches)
    train_loss_hist.append((epoch, epoch_loss))

    print(f'Epoch {epoch}/{NUM_EPOCHS}, Loss={epoch_loss:.4f}, '
          f'耗时 {time.time() - t_start:.1f}s')

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(),
                   os.path.join(MODEL_SAVE_PATH, 'finetuned_model_best.pth'))
        print(f'  最佳模型已保存')

# ============================================================
# 保存
# ============================================================
torch.save(model.state_dict(),
           os.path.join(MODEL_SAVE_PATH, 'finetuned_model_final.pth'))
print(f'模型保存至 {MODEL_SAVE_PATH}')

# 绘制微调曲线
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

train_arr = np.array(train_loss_hist)
plt.figure(figsize=(7, 4))
plt.plot(train_arr[:, 0], train_arr[:, 1], 'b-', lw=1.2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Fine-tuning Loss Curve')
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_PATH, 'finetune_curve.png'), dpi=150)
plt.close()

print('=== 微调完成 ===')
