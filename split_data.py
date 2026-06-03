"""
split_data.py — 将公开训练集按 7:3 拆分为基础训练集和微调集
只拆分 Train (4745 samples), Val 和 Test 保持不变

用法: python split_data.py
输出: models/train_split.npz (包含 base_idx 和 ft_idx)
"""

import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DATA_PATH = os.path.join(SCRIPT_DIR, '数据集', '1、公开数据集')
MODEL_SAVE_PATH  = os.path.join(SCRIPT_DIR, 'models')
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

SPLIT_RATIO = 0.7   # 70% base, 30% fine-tune
SEED = 42

print(f'=== 拆分公开训练集 ({SPLIT_RATIO:.0%} base / {(1-SPLIT_RATIO):.0%} fine-tune) ===')

# 加载训练集以获取样本数
from scipy.io import loadmat
train_ppg = loadmat(os.path.join(PUBLIC_DATA_PATH, 'TrainPPG.mat'))
n_total = train_ppg['TrainPPG'].shape[0]
print(f'训练集总样本数: {n_total}')

# 随机打乱索引
rng = np.random.RandomState(SEED)
indices = rng.permutation(n_total)
n_base = int(n_total * SPLIT_RATIO)

base_idx = np.sort(indices[:n_base])       # 前 70%
ft_idx   = np.sort(indices[n_base:])       # 后 30%

print(f'基础训练集: {len(base_idx)} 样本')
print(f'微调训练集: {len(ft_idx)} 样本')
print(f'随机种子: {SEED}')

# 保存拆分索引
np.savez(os.path.join(MODEL_SAVE_PATH, 'train_split.npz'),
         base_idx=base_idx, ft_idx=ft_idx, seed=SEED, ratio=SPLIT_RATIO)
print(f'拆分索引已保存至 {MODEL_SAVE_PATH}/train_split.npz')
