"""
split_data.py — [已弃用] 公开数据集 7:3 内部拆分工具

状态: 该文件不再被当前流水线使用。
      旧方案: 将公开训练集拆分为 base (70%) 和 fine-tune (30%)，在同分布下模拟迁移。
      新方案: train_base_model.py 使用公开全量训练 → fine_tune_model.py 在自建数据集上做真实跨域微调。

保留原因: 历史参考，复现 v1.x 内部拆分实验时使用。
"""

import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DATA_PATH = os.path.join(SCRIPT_DIR, '数据集', '1、公开数据集')
MODEL_SAVE_PATH  = os.path.join(SCRIPT_DIR, 'models')
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

SPLIT_RATIO = 0.7
SEED = 42

print('[已弃用] 此拆分仅在复现 v1.x 内部拆分实验时使用。')
print('当前流水线: train_base_model.py (公开全量) → fine_tune_model.py (公开→自建跨域)')
print()

print(f'=== 拆分公开训练集 ({SPLIT_RATIO:.0%} base / {(1-SPLIT_RATIO):.0%} fine-tune) ===')

from scipy.io import loadmat
train_ppg = loadmat(os.path.join(PUBLIC_DATA_PATH, 'TrainPPG.mat'))
n_total = train_ppg['TrainPPG'].shape[0]
print(f'训练集总样本数: {n_total}')

rng = np.random.RandomState(SEED)
indices = rng.permutation(n_total)
n_base = int(n_total * SPLIT_RATIO)

base_idx = np.sort(indices[:n_base])
ft_idx   = np.sort(indices[n_base:])

print(f'基础训练集: {len(base_idx)} 样本')
print(f'微调训练集: {len(ft_idx)} 样本')
print(f'随机种子: {SEED}')

np.savez(os.path.join(MODEL_SAVE_PATH, 'train_split.npz'),
         base_idx=base_idx, ft_idx=ft_idx, seed=SEED, ratio=SPLIT_RATIO)
print(f'拆分索引已保存至 {MODEL_SAVE_PATH}/train_split.npz')
