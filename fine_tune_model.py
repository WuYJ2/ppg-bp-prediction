"""
fine_tune_model.py — 公开→自建跨域微调 1D-ResNet 模型

训练策略:
  源域 (旧数据): 公开数据集全量训练集 (4745 样本), 用于保持基础能力
  目标域 (新数据): 自建数据集训练集 (774 样本), 用于适应新域

微调机制 (三种互补技巧):
  1. 参数冻结: 冻结浅层 (conv1/bn1/layer1/layer2), 只更新深层 (layer3+FC)
     浅层学的是"波形基本形态"(跨域通用), 深层需要在新域重新校准
  2. 混合批采样: 每批 16 公开 + 16 自建, 同时在两个域上注入梯度信号
  3. 知识蒸馏 (可选): 教师模型约束学生模型, 防止灾难性遗忘

消融实验: USE_DISTILL 开关控制蒸馏的有无

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
    preprocess_ppg, normalize_3ch, apply_normalize_3ch, normalize_bp,
    safe_torch_save, safe_torch_load
)

# ============================================================
# 配置
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DATA_PATH = os.path.join(SCRIPT_DIR, '数据集', '1、公开数据集')
SELF_DATA_PATH   = os.path.join(SCRIPT_DIR, 'dataset')           # 自建原始 PPG + BP
MODEL_SAVE_PATH  = os.path.join(SCRIPT_DIR, 'models')
OUTPUT_PATH      = os.path.join(SCRIPT_DIR, 'cnn_output')
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 微调超参数
NUM_EPOCHS      = 80
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-5
GRAD_CLIP       = 5.0

# 消融实验开关
USE_DISTILL = True   # True=蒸馏微调, False=消融实验(无蒸馏)

# 蒸馏参数 (仅 USE_DISTILL=True 时生效)
DISTILL_WEIGHT = 0.4   # α: 蒸馏损失权重 (0.2-0.7)
TEMPERATURE    = 3.0   # T: 温度 (2-4)

# 新旧数据混合比
FT_DATA_RATIO = 0.5  # 每批中自建数据 (目标域) 占比

# 冻结层名称 (对应 ResNet1D 的子模块)
FROZEN_MODULES = ['conv1', 'bn1', 'layer1', 'layer2']

# 信号参数
FS      = 125
SIG_LEN = 2048

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {DEVICE}')
print(f'蒸馏: {"启用" if USE_DISTILL else "关闭 (消融实验)"}')
if USE_DISTILL:
    print(f'  蒸馏权重 α = {DISTILL_WEIGHT}, 温度 T = {TEMPERATURE}')
print(f'冻结模块: {FROZEN_MODULES}')
print(f'训练策略: 公开数据集 (源域) → 自建数据集 (目标域) 跨域微调')

# ============================================================
# 加载基础模型
# ============================================================
print('=== 加载基础模型 ===')

model = ResNet1D(in_channels=3, num_outputs=2).to(DEVICE)

best_path = os.path.join(MODEL_SAVE_PATH, 'base_model_best.pth')
if not os.path.exists(best_path):
    best_path = os.path.join(MODEL_SAVE_PATH, 'base_model_final.pth')
model.load_state_dict(safe_torch_load(best_path, map_location=DEVICE,
                                  weights_only=True))

# 教师模型 = 冻结的基础模型 (仅蒸馏模式)
teacher = None
if USE_DISTILL:
    teacher = ResNet1D(in_channels=3, num_outputs=2).to(DEVICE)
    teacher.load_state_dict(safe_torch_load(best_path, map_location=DEVICE,
                                        weights_only=True))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

print(f'基础模型已加载: {best_path}')

# 加载归一化参数 (由 train_base_model.py 在公开训练集全量上计算)
norm_data = np.load(os.path.join(MODEL_SAVE_PATH, 'norm_params.npz'))
ch_mean = norm_data['ch_mean']
ch_std  = norm_data['ch_std']
bp_mean = norm_data['bp_mean']
bp_std  = norm_data['bp_std']

norm_stats = {'ch_mean': ch_mean, 'ch_std': ch_std}

# ============================================================
# 加载源域数据: 公开数据集 (旧数据, 用于保持基础能力)
# ============================================================
print('=== 加载源域数据: 公开训练集 (全量) ===')
data_pub = load_public_dataset(PUBLIC_DATA_PATH)

X_pub_raw = data_pub['train']['ppg']       # 4745 样本
Y_pub_sbp = data_pub['train']['sbp']
Y_pub_dbp = data_pub['train']['dbp']

print(f'源域 (公开) 训练样本数: {len(X_pub_raw)}')

X_pub_3ch  = preprocess_ppg(X_pub_raw, FS)
X_pub_3ch  = apply_normalize_3ch(X_pub_3ch, norm_stats)
Y_pub_norm = (np.column_stack([Y_pub_sbp, Y_pub_dbp]) - bp_mean) / bp_std

# ============================================================
# 加载目标域数据: 自建数据集 (新数据, 用于适应新域)
# ============================================================
print('=== 加载目标域数据: 自建训练集 ===')
data_self = load_self_dataset(SELF_DATA_PATH, SIG_LEN)

X_self_raw = data_self['train']['ppg']     # 774 样本
Y_self_sbp = data_self['train']['sbp']
Y_self_dbp = data_self['train']['dbp']

print(f'目标域 (自建) 训练样本数: {len(X_self_raw)}')

X_self_3ch  = preprocess_ppg(X_self_raw, FS)
X_self_3ch  = apply_normalize_3ch(X_self_3ch, norm_stats)
Y_self_norm = (np.column_stack([Y_self_sbp, Y_self_dbp]) - bp_mean) / bp_std

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
X_pub_t   = torch.from_numpy(X_pub_3ch)
Y_pub_t   = torch.from_numpy(Y_pub_norm)
X_self_t  = torch.from_numpy(X_self_3ch)
Y_self_t  = torch.from_numpy(Y_self_norm)

num_pub   = len(X_pub_t)
num_self  = len(X_self_t)

batch_pub   = int(BATCH_SIZE * (1 - FT_DATA_RATIO))   # 16
batch_self  = BATCH_SIZE - batch_pub                    # 16

num_batches = max(num_pub // batch_pub, num_self // batch_self) + 1

print(f'混合批: 源域 (公开) {batch_pub} + 目标域 (自建) {batch_self} = {BATCH_SIZE} / 批')

# ============================================================
# 微调训练
# ============================================================
mode_str = '蒸馏 + 混合批' if USE_DISTILL else '消融实验 (无蒸馏, 仅混合批)'
print(f'=== 开始公开→自建跨域微调: {mode_str} ===')

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

train_loss_hist = []
best_loss       = float('inf')

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    t_start = time.time()

    # 每轮独立打乱索引
    idx_pub   = torch.randperm(num_pub)
    idx_self  = torch.randperm(num_self)

    for b in range(num_batches):
        # 采样源域数据 (公开)
        pub_start = (b * batch_pub) % max(1, num_pub - batch_pub + 1)
        pub_end   = min(pub_start + batch_pub, num_pub)
        pub_idx   = idx_pub[pub_start:pub_end]

        # 采样目标域数据 (自建)
        self_start = (b * batch_self) % max(1, num_self - batch_self + 1)
        self_end   = min(self_start + batch_self, num_self)
        self_idx   = idx_self[self_start:self_end]

        x_pub   = X_pub_t[pub_idx].to(DEVICE)
        y_pub   = Y_pub_t[pub_idx].to(DEVICE)
        x_self  = X_self_t[self_idx].to(DEVICE)
        y_self  = Y_self_t[self_idx].to(DEVICE)

        # === 前向传播 ===
        # 目标域 (自建): 任务损失
        y_self_pred = model(x_self)
        loss_self = criterion(y_self_pred, y_self)

        # 源域 (公开): 任务损失 (+ 蒸馏损失)
        y_pub_pred = model(x_pub)
        task_pub = criterion(y_pub_pred, y_pub)

        if USE_DISTILL:
            with torch.no_grad():
                y_teacher = teacher(x_pub)
            distill_pub = criterion(y_pub_pred / TEMPERATURE,
                                     y_teacher / TEMPERATURE) * (TEMPERATURE ** 2)
            loss = loss_self + (1 - DISTILL_WEIGHT) * task_pub + DISTILL_WEIGHT * distill_pub
        else:
            loss = loss_self + task_pub

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
        safe_torch_save(model.state_dict(),
                   os.path.join(MODEL_SAVE_PATH, 'finetuned_model_best.pth'))
        print(f'  最佳模型已保存')

# ============================================================
# 保存
# ============================================================
safe_torch_save(model.state_dict(),
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
plt.title('Fine-tuning Loss Curve (Public → Self-built Cross-Domain)')
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_PATH, 'finetune_curve.png'), dpi=150)
plt.close()

print('=== 公开→自建跨域微调完成 ===')
