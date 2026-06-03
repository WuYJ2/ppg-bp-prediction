"""
model.py — 1D-ResNet 模型定义 + 数据处理工具
供 train_base_model.py / fine_tune_model.py / evaluate_model.py 共用
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
import h5py
import os


# ============================================================
# 1D-ResNet 模型
# ============================================================

class BasicBlock1D(nn.Module):
    """1D 残差块: Conv1d(3×1) → BN → ReLU → Conv1d(3×1) → BN → +skip → ReLU"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                           stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet1D(nn.Module):
    """1D ResNet: 输入 (B, 3, 2048) → 输出 (B, 2) SBP/DBP"""

    def __init__(self, in_channels=3, num_outputs=2):
        super().__init__()

        # 初始卷积: 7×1, stride 2
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7,
                                stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 三个残差块
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)   # → (64, 512)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)  # → (128, 256)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2) # → (256, 128)

        self.gap = nn.AdaptiveAvgPool1d(1)  # → (256, 1)
        self.fc = nn.Linear(256, num_outputs)

        self._initialize_weights()

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock1D(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, 3, 2048)
        x = F.relu(self.bn1(self.conv1(x)))    # (B, 64, 1024)
        x = self.maxpool(x)                     # (B, 64, 512)
        x = self.layer1(x)                      # (B, 64, 512)
        x = self.layer2(x)                      # (B, 128, 256)
        x = self.layer3(x)                      # (B, 256, 128)
        x = self.gap(x)                         # (B, 256, 1)
        x = x.view(x.size(0), -1)               # (B, 256)
        x = self.fc(x)                          # (B, 2)
        return x


# ============================================================
# 数据加载工具
# ============================================================

def load_mat_v73(filepath):
    """加载 v7.3 格式 .mat 文件 (HDF5)，返回 {var_name: np.array}"""
    data = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            if key != '#refs#':
                arr = np.array(f[key][:])
                # h5py 存储为 (cols, rows)，需要转置
                if arr.ndim == 2:
                    arr = arr.T
                data[key] = np.squeeze(arr)
    return data


def load_public_dataset(data_path):
    """加载公开数据集 (scipy 兼容 .mat)，返回 (X_train, Y_train, X_val, Y_val, X_test, Y_test)"""
    train_ppg = loadmat(os.path.join(data_path, 'TrainPPG.mat'))
    train_sbp = loadmat(os.path.join(data_path, 'TrainSBP.mat'))
    train_dbp = loadmat(os.path.join(data_path, 'TrainDBP.mat'))
    val_ppg = loadmat(os.path.join(data_path, 'ValPPG.mat'))
    val_sbp = loadmat(os.path.join(data_path, 'ValSBP.mat'))
    val_dbp = loadmat(os.path.join(data_path, 'ValDBP.mat'))
    test_ppg = loadmat(os.path.join(data_path, 'TestPPG.mat'))
    test_sbp = loadmat(os.path.join(data_path, 'TestSBP.mat'))
    test_dbp = loadmat(os.path.join(data_path, 'TestDBP.mat'))

    return {
        'train': {
            'ppg': train_ppg['TrainPPG'].astype(np.float32),
            'sbp': train_sbp['TrainSBP'].ravel().astype(np.float32),
            'dbp': train_dbp['TrainDBP'].ravel().astype(np.float32),
        },
        'val': {
            'ppg': val_ppg['ValPPG'].astype(np.float32),
            'sbp': val_sbp['ValSBP'].ravel().astype(np.float32),
            'dbp': val_dbp['ValDBP'].ravel().astype(np.float32),
        },
        'test': {
            'ppg': test_ppg['TestPPG'].astype(np.float32),
            'sbp': test_sbp['TestSBP'].ravel().astype(np.float32),
            'dbp': test_dbp['TestDBP'].ravel().astype(np.float32),
        },
    }


def load_self_dataset(data_path, sig_len=2048):
    """加载自建数据集 (v7.3 格式 cell 数组), 返回 {train: {ppg, sbp, dbp}, test: {...}}"""

    def load_cell_ppg(filepath, sig_len):
        """从 v7.3 cell 数组 .mat 文件中提取 PPG 信号矩阵
        数据存储为 (sig_len, N) 的 HDF5 引用数组, 每列为一个信号"""
        signals = []
        with h5py.File(filepath, 'r') as f:
            var_name = [k for k in f.keys() if k != '#refs#'][0]
            refs_ds = f[var_name]  # shape (sig_len, N), dtype=object (references)
            n_cells = refs_ds.shape[1]
            print(f'  正在加载 {n_cells} 条 PPG 信号...')

            for j in range(n_cells):
                col_refs = refs_ds[:, j]  # 一次读入整列 (sig_len,) 引用
                sig = np.empty(sig_len, dtype=np.float32)
                for k in range(sig_len):
                    deref = f[col_refs[k]]
                    sig[k] = float(np.array(deref[:]).flat[0])
                signals.append(sig)

                if (j + 1) % 50 == 0:
                    print(f'    {j + 1}/{n_cells}...')

            print(f'  加载完成: {len(signals)} 条')

        return np.array(signals, dtype=np.float32)

    def load_v73_var(filepath):
        """从 v7.3 .mat 文件中加载简单数值变量 (非 cell)"""
        with h5py.File(filepath, 'r') as f:
            var_names = [k for k in f.keys() if k != '#refs#']
            if not var_names:
                raise ValueError(f'{filepath} 中未找到数据变量')
            arr = np.array(f[var_names[0]][:], dtype=np.float32)
            # h5py 存储为转置: MATLAB (1,N) → h5py (N,1)
            return arr.T.flatten() if arr.ndim == 2 else arr.flatten()

    X_train = load_cell_ppg(os.path.join(data_path, 'PPG', 'TrainPPG_cell.mat'), sig_len)
    X_test  = load_cell_ppg(os.path.join(data_path, 'PPG', 'TestPPG_cell.mat'), sig_len)

    Y_train_sbp = load_v73_var(os.path.join(data_path, 'BP', 'TrainSBP.mat'))
    Y_train_dbp = load_v73_var(os.path.join(data_path, 'BP', 'TrainDBP.mat'))
    Y_test_sbp  = load_v73_var(os.path.join(data_path, 'BP', 'TestSBP.mat'))
    Y_test_dbp  = load_v73_var(os.path.join(data_path, 'BP', 'TestDBP.mat'))

    # 确保长度一致
    n_train = min(len(X_train), len(Y_train_sbp), len(Y_train_dbp))
    n_test  = min(len(X_test), len(Y_test_sbp), len(Y_test_dbp))

    print(f'  自建训练集: PPG {X_train.shape}, SBP {Y_train_sbp.shape}, DBP {Y_train_dbp.shape}')
    print(f'  自建测试集: PPG {X_test.shape}, SBP {Y_test_sbp.shape}, DBP {Y_test_dbp.shape}')

    return {
        'train': {
            'ppg': X_train[:n_train],
            'sbp': Y_train_sbp[:n_train],
            'dbp': Y_train_dbp[:n_train],
        },
        'test': {
            'ppg': X_test[:n_test],
            'sbp': Y_test_sbp[:n_test],
            'dbp': Y_test_dbp[:n_test],
        },
    }


# ============================================================
# 预处理
# ============================================================

def preprocess_ppg(ppg_raw, Fs=125):
    """计算 PPG + 一阶导 + 二阶导, 堆叠为 (N, 3, L)"""
    dt = 1.0 / Fs
    N = ppg_raw.shape[0]
    L = ppg_raw.shape[1]
    X_3ch = np.zeros((N, 3, L), dtype=np.float32)

    for i in range(N):
        ppg = ppg_raw[i].astype(np.float64)
        d1 = np.gradient(ppg, dt)
        d2 = np.gradient(d1, dt)
        X_3ch[i, 0, :] = ppg.astype(np.float32)
        X_3ch[i, 1, :] = d1.astype(np.float32)
        X_3ch[i, 2, :] = d2.astype(np.float32)

    return X_3ch


def normalize_3ch(X):
    """对每通道做 z-score 归一化, 返回 (X_norm, stats_dict)"""
    stats = {'ch_mean': np.zeros(3, dtype=np.float32),
             'ch_std': np.zeros(3, dtype=np.float32)}
    X_norm = np.zeros_like(X)

    for c in range(3):
        ch_data = X[:, c, :].ravel()
        stats['ch_mean'][c] = ch_data.mean()
        stats['ch_std'][c] = ch_data.std()
        X_norm[:, c, :] = (X[:, c, :] - stats['ch_mean'][c]) / stats['ch_std'][c]

    return X_norm, stats


def apply_normalize_3ch(X, stats):
    """使用已有统计量做 z-score 归一化"""
    X_norm = np.zeros_like(X)
    for c in range(3):
        X_norm[:, c, :] = (X[:, c, :] - stats['ch_mean'][c]) / stats['ch_std'][c]
    return X_norm


def normalize_bp(sbp, dbp):
    """对 BP 标签做 z-score 归一化"""
    Y = np.column_stack([sbp, dbp]).astype(np.float32)
    bp_mean = Y.mean(axis=0, keepdims=True)
    bp_std = Y.std(axis=0, keepdims=True)
    Y_norm = (Y - bp_mean) / bp_std
    return Y_norm, bp_mean, bp_std
