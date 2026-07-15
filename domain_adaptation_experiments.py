"""Few-shot domain adaptation experiments for the PPG BP project.
小样本域适应实验: 在自建 PPG 目标域上对比两种迁移策略。

The script compares two target-domain reconstruction strategies:
1. Feature-level domain adaptation with CORAL loss.
   → 特征级域适应: 用 CORAL 对齐源域与目标域的 GAP 特征分布
2. Learning without Forgetting (LwF) using the base model as teacher.
   → LwF: 冻结 teacher，student 学习目标标签的同时保持 teacher 在源域的预测

流程: Base 模型 → 少量目标域样本微调 → 自建测试集评估 → JSON/NPZ 输出
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import ResNet1D, apply_normalize_3ch, load_public_dataset, normalize_bp, preprocess_ppg, safe_torch_save, safe_torch_load


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PUBLIC_DIR = SCRIPT_DIR / "数据集" / "1、公开数据集"
DEFAULT_SELF_DIR = SCRIPT_DIR / "dataset"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "models"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "cnn_output" / "domain_adaptation"


def is_lfs_pointer(path: os.PathLike[str] | str) -> bool:
    """Return True when a file is a Git LFS pointer instead of real data."""
    p = Path(path)
    if not p.exists() or p.stat().st_size > 512:
        return False
    try:
        with p.open("rb") as f:
            head = f.read(128)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算 SBP/DBP 的 MAE 和 STD (mmHg)"""
    err = y_pred - y_true
    return {
        "SBP_MAE": float(np.mean(np.abs(err[:, 0]))),
        "SBP_STD": float(np.std(err[:, 0])),
        "DBP_MAE": float(np.mean(np.abs(err[:, 1]))),
        "DBP_STD": float(np.std(err[:, 1])),
    }


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CORAL (CORrelation ALignment) 损失: 对齐源域和目标域特征的均值和协方差。
    L = ||μ_s - μ_t||² + ||C_s - C_t||²_F
    其中 C = (X^T X) / (n-1) 为协方差矩阵。"""
    if source.shape[0] < 2 or target.shape[0] < 2:
        return source.new_tensor(0.0)
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    source_cov = source_centered.t().matmul(source_centered) / (source.shape[0] - 1)
    target_cov = target_centered.t().matmul(target_centered) / (target.shape[0] - 1)
    mean_loss = F.mse_loss(source.mean(dim=0), target.mean(dim=0))
    cov_loss = F.mse_loss(source_cov, target_cov)
    return mean_loss + cov_loss


@dataclass
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


class ResNetWithFeatures(ResNet1D):
    """扩展 ResNet1D: 额外暴露 forward_features() 返回 GAP 层 256 维特征，
    供 CORAL 域适应使用。forward() 行为不变。"""

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取 GAP 后的 256 维特征 (不经过 FC 头)"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.forward_features(x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def assert_real_files(paths: Iterable[Path]) -> None:
    pointers = [str(p) for p in paths if is_lfs_pointer(p)]
    missing = [str(p) for p in paths if not p.exists()]
    if missing or pointers:
        parts = []
        if missing:
            parts.append("Missing files:\n  " + "\n  ".join(missing))
        if pointers:
            parts.append("Git LFS pointer files, not real data:\n  " + "\n  ".join(pointers))
        parts.append("Fetch the LFS data first, then rerun this script.")
        raise FileNotFoundError("\n".join(parts))


def load_v73_vector(path: Path) -> np.ndarray:
    """从 MATLAB v7.3 .mat 文件中读取向量 (如 BP 标签), 自动处理 h5py 转置"""
    with h5py.File(path, "r") as f:
        names = [k for k in f.keys() if k != "#refs#"]
        arr = np.array(f[names[0]][:], dtype=np.float32)
    return arr.T.reshape(-1) if arr.ndim == 2 else arr.reshape(-1)


def load_v73_cell_ppg(path: Path, sig_len: int = 2048) -> np.ndarray:
    """从 MATLAB v7.3 cell 数组 .mat 文件中提取 PPG 信号矩阵。
    每个 cell 的 2048 个元素以 HDF5 标量引用形式存储, 逐列解引用拼成一条信号。"""
    signals = []
    with h5py.File(path, "r") as f:
        names = [k for k in f.keys() if k != "#refs#"]
        refs = f[names[0]]
        n_cells = refs.shape[1]
        for j in range(n_cells):
            sig = np.empty(sig_len, dtype=np.float32)
            for k, ref in enumerate(refs[:, j]):
                sig[k] = float(np.array(f[ref][:]).flat[0])
            signals.append(sig)
    return np.asarray(signals, dtype=np.float32)


def load_self_dataset(data_dir: Path, sig_len: int = 2048) -> DatasetBundle:
    paths = [
        data_dir / "PPG" / "TrainPPG_cell.mat",
        data_dir / "PPG" / "TestPPG_cell.mat",
        data_dir / "BP" / "TrainSBP.mat",
        data_dir / "BP" / "TrainDBP.mat",
        data_dir / "BP" / "TestSBP.mat",
        data_dir / "BP" / "TestDBP.mat",
    ]
    assert_real_files(paths)
    x_train = load_v73_cell_ppg(paths[0], sig_len)
    x_test = load_v73_cell_ppg(paths[1], sig_len)
    y_train = np.column_stack([load_v73_vector(paths[2]), load_v73_vector(paths[3])]).astype(np.float32)
    y_test = np.column_stack([load_v73_vector(paths[4]), load_v73_vector(paths[5])]).astype(np.float32)
    n_train = min(len(x_train), len(y_train))
    n_test = min(len(x_test), len(y_test))
    return DatasetBundle(x_train[:n_train], y_train[:n_train], x_test[:n_test], y_test[:n_test])


def load_norm_stats(model_dir: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    path = model_dir / "norm_params.npz"
    data = np.load(path)
    norm_stats = {"ch_mean": data["ch_mean"], "ch_std": data["ch_std"]}
    return norm_stats, data["bp_mean"], data["bp_std"]


def load_model(model_dir: Path, filename: str, device: torch.device) -> ResNetWithFeatures:
    model = ResNetWithFeatures(in_channels=3, num_outputs=2).to(device)
    state = safe_torch_load(str(model_dir / filename), map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


def predict(model: nn.Module, x: np.ndarray, bp_mean: np.ndarray, bp_std: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=64, shuffle=False)
    preds = []
    with torch.no_grad():
        for (batch,) in loader:
            preds.append(model(batch.to(device)).cpu().numpy())
    return np.vstack(preds) * bp_std + bp_mean


def fit_affine_calibrator(y_pred_train: np.ndarray, y_true_train: np.ndarray, ridge: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """用目标域小样本拟合仿射校准层: y_true ≈ y_pred @ coef + intercept。
    岭回归 (ridge) 防止小样本过拟合。校准用于纠正跨域系统偏移。"""
    x = np.column_stack([y_pred_train, np.ones(len(y_pred_train), dtype=y_pred_train.dtype)])
    penalty = np.eye(x.shape[1], dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ y_true_train)
    return weights[:-1, :], weights[-1, :]


def apply_affine_calibrator(y_pred: np.ndarray, coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return y_pred @ coef + intercept


def predict_with_optional_calibration(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    bp_mean: np.ndarray,
    bp_std: np.ndarray,
    device: torch.device,
    calibrate: bool,
    ridge: float,
) -> np.ndarray:
    test_pred = predict(model, test_x, bp_mean, bp_std, device)
    if not calibrate:
        return test_pred
    train_pred = predict(model, train_x, bp_mean, bp_std, device)
    coef, intercept = fit_affine_calibrator(train_pred, train_y, ridge=ridge)
    return apply_affine_calibrator(test_pred, coef, intercept)


# ============================================================
# 可视化函数
# ============================================================

def plot_scatter_comparison(
    y_true: np.ndarray,
    preds: Dict[str, np.ndarray],
    title: str,
    save_path: Path,
) -> None:
    """多方法 SBP/DBP 散点图矩阵 (2 行 × N 列, 每列一个方法)。
    y_true: (N, 2) 真实值
    preds: {"method_name": (N, 2) 预测值, ...}
    """
    methods = list(preds.keys())
    n_cols = len(methods)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 9))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for col, (name, y_pred) in enumerate(preds.items()):
        for row, bp_idx in enumerate([0, 1]):
            ax = axes[row, col]
            bp_name = "SBP" if bp_idx == 0 else "DBP"
            yt = y_true[:, bp_idx]
            yp = y_pred[:, bp_idx]
            err = yp - yt
            mae = float(np.mean(np.abs(err)))
            std = float(np.std(err))
            r = float(np.corrcoef(yt, yp)[0, 1])

            ax.scatter(yt, yp, s=20, alpha=0.65, color=colors[col % len(colors)], edgecolors="none")
            lo = min(float(yt.min()), float(yp.min()))
            hi = max(float(yt.max()), float(yp.max()))
            pad = (hi - lo) * 0.08
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1, label="Ideal")
            # 拟合线
            coef = np.polyfit(yt, yp, 1)
            xs = np.linspace(lo, hi, 100)
            ax.plot(xs, np.polyval(coef, xs), color="#d62728", linewidth=1.5, label="Fit")
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{name} {bp_name}", fontsize=11, weight="bold")
            ax.set_xlabel(f"True {bp_name} (mmHg)")
            ax.set_ylabel(f"Predicted {bp_name} (mmHg)")
            ax.text(0.04, 0.95, f"MAE={mae:.2f}\nSTD={std:.2f}\nr={r:.3f}",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.9))
            ax.grid(alpha=0.25)
            ax.legend(loc="lower right", fontsize=7)

    fig.suptitle(title, fontsize=14, weight="bold")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'  [可视化] 散点对比图已保存: {save_path}')


def plot_loss_curves(loss_histories: Dict[str, list], save_path: Path) -> None:
    """绘制训练损失曲线 (多个方法对比)。
    loss_histories: {"method": [epoch_loss, ...], ...}
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"feature_da": "#1f77b4", "lwf": "#ff7f0e"}
    for name, losses in loss_histories.items():
        if losses:
            ax.plot(range(1, len(losses) + 1), losses, marker=".", linewidth=1.5,
                    label=name, color=colors.get(name, "#333333"))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.grid(alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'  [可视化] 损失曲线已保存: {save_path}')


def plot_metric_bars(all_metrics: Dict[str, Dict[str, float]], save_path: Path) -> None:
    """三方法 × 四指标柱状图汇总。
    all_metrics: {"method": {"SBP_MAE": ..., "SBP_STD": ..., "DBP_MAE": ..., "DBP_STD": ...}}
    """
    methods = list(all_metrics.keys())
    metric_keys = ["SBP_MAE", "SBP_STD", "DBP_MAE", "DBP_STD"]
    metric_labels = ["SBP MAE", "SBP STD", "DBP MAE", "DBP STD"]
    colors = ["#6c757d", "#1f77b4", "#ff7f0e"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, mk, ml in zip(axes.ravel(), metric_keys, metric_labels):
        vals = [all_metrics[m][mk] for m in methods]
        bars = ax.bar(methods, vals, color=colors[:len(methods)])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.2f}", ha="center", fontsize=9)
        ax.set_title(ml)
        ax.set_ylabel("mmHg")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Few-shot Domain Adaptation Results", fontsize=14, weight="bold")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'  [可视化] 指标柱状图已保存: {save_path}')

# ============================================================

def few_shot_subset(bundle: DatasetBundle, n_shot: int, seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    n = min(n_shot, len(bundle.x_train))
    idx = np.sort(rng.choice(len(bundle.x_train), size=n, replace=False))
    return DatasetBundle(bundle.x_train[idx], bundle.y_train[idx], bundle.x_test, bundle.y_test)


def freeze_backbone_prefixes(model: nn.Module, prefixes: Tuple[str, ...] = ("conv1", "bn1", "layer1")) -> None:
    """冻结骨干浅层 (默认 conv1 + bn1 + layer1), 仅微调深层和 FC 头"""
    for name, param in model.named_parameters():
        param.requires_grad = not any(name.startswith(prefix) for prefix in prefixes)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def train_feature_da(
    model: ResNetWithFeatures,
    source_x: np.ndarray,
    target_x: np.ndarray,
    target_y_norm: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    coral_weight: float,
) -> tuple[ResNetWithFeatures, list]:
    """特征级域适应训练。
    损失 = MSE(目标标签) + coral_weight × CORAL(源域GAP特征, 目标域GAP特征)
    冻结浅层, 仅训练深层 + FC。
    返回 (模型, epoch_loss列表)"""
    freeze_backbone_prefixes(model)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-5)
    target_loader = make_loader(target_x, target_y_norm, batch_size, shuffle=True)
    source_loader = DataLoader(TensorDataset(torch.from_numpy(source_x)), batch_size=batch_size, shuffle=True, drop_last=False)
    criterion = nn.MSELoss()
    loss_history = []
    for epoch in range(epochs):
        model.train()
        source_iter = iter(source_loader)
        epoch_loss = 0.0
        n_batches = 0
        for target_batch, y_batch in target_loader:
            try:
                (source_batch,) = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                (source_batch,) = next(source_iter)
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(target_batch)
            feat_source = model.forward_features(source_batch)
            feat_target = model.forward_features(target_batch)
            loss = criterion(pred, y_batch) + coral_weight * coral_loss(feat_source, feat_target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        loss_history.append(epoch_loss / max(n_batches, 1))
        if (epoch + 1) % 10 == 0:
            print(f'    Feature DA epoch {epoch+1}/{epochs}, loss={loss_history[-1]:.4f}')
    return model, loss_history


def train_lwf(
    model: ResNetWithFeatures,
    teacher: ResNetWithFeatures,
    source_x: np.ndarray,
    target_x: np.ndarray,
    target_y_norm: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    lwf_weight: float,
) -> tuple[ResNetWithFeatures, list]:
    """Learning without Forgetting (LwF) 训练。
    teacher 冻结; student 损失 = MSE(目标标签) + lwf_weight × MSE(student源域预测, teacher源域预测)
    通过保持 teacher 在源域的输出, 防止灾难性遗忘。
    返回 (模型, epoch_loss列表)"""
    freeze_backbone_prefixes(model)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-5)
    target_loader = make_loader(target_x, target_y_norm, batch_size, shuffle=True)
    source_loader = DataLoader(TensorDataset(torch.from_numpy(source_x)), batch_size=batch_size, shuffle=True, drop_last=False)
    criterion = nn.MSELoss()
    loss_history = []
    for epoch in range(epochs):
        model.train()
        source_iter = iter(source_loader)
        epoch_loss = 0.0
        n_batches = 0
        for target_batch, y_batch in target_loader:
            try:
                (source_batch,) = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                (source_batch,) = next(source_iter)
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)
            y_batch = y_batch.to(device)
            with torch.no_grad():
                teacher_source = teacher(source_batch)
            pred_target = model(target_batch)
            pred_source = model(source_batch)
            loss = criterion(pred_target, y_batch) + lwf_weight * criterion(pred_source, teacher_source)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        loss_history.append(epoch_loss / max(n_batches, 1))
        if (epoch + 1) % 10 == 0:
            print(f'    LwF epoch {epoch+1}/{epochs}, loss={loss_history[-1]:.4f}')
    return model, loss_history


def run(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    """主实验流程:
    1. 加载公开数据 (源域) + 自建数据 (目标域)
    2. 预处理 PPG → 3 通道 → 归一化
    3. 评估 Base 模型 (跨域基线)
    4. 训练 Feature DA 模型 → 评估
    5. 训练 LwF 模型 → 评估
    6. 保存预测值 (npz) + 指标 (json)
    """
    os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    public_dir = Path(args.public_dir)
    self_dir = Path(args.self_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert_real_files(
        [
            public_dir / "TrainPPG.mat",
            public_dir / "TrainSBP.mat",
            public_dir / "TrainDBP.mat",
            self_dir / "PPG" / "TrainPPG_cell.mat",
            self_dir / "PPG" / "TestPPG_cell.mat",
        ]
    )

    norm_stats, bp_mean, bp_std = load_norm_stats(model_dir)
    public = load_public_dataset(str(public_dir))
    self_bundle = few_shot_subset(load_self_dataset(self_dir), args.shots, args.seed)

    # ---- 预处理: PPG → 3 通道 → z-score 归一化 ----
    source_x = preprocess_ppg(public["train"]["ppg"], args.fs)
    source_x = apply_normalize_3ch(source_x, norm_stats)
    target_train_x = apply_normalize_3ch(preprocess_ppg(self_bundle.x_train, args.fs), norm_stats)
    target_test_x = apply_normalize_3ch(preprocess_ppg(self_bundle.x_test, args.fs), norm_stats)
    target_y_norm = (self_bundle.y_train - bp_mean) / bp_std

    # ---- 1. Base 模型 (跨域基线, 不做适配) ----
    results = {}
    base = load_model(model_dir, "base_model_best.pth", device)
    base_pred = predict_with_optional_calibration(
        base,
        target_train_x,
        self_bundle.y_train,
        target_test_x,
        bp_mean,
        bp_std,
        device,
        args.calibrate,
        args.calibration_ridge,
    )
    results["base"] = compute_metrics(self_bundle.y_test, base_pred)

    # ---- 可视化: Base 基线预测散点图 ----
    plot_scatter_comparison(
        self_bundle.y_test,
        {"Base (no adapt)": base_pred},
        f"Base Model Cross-domain Baseline ({args.shots}-shot)",
        output_dir / f"scatter_base_{args.shots}shot.png",
    )

    # ---- 2. 特征级域适应 (CORAL) ----
    feature_da = load_model(model_dir, "base_model_best.pth", device)
    feature_da, da_loss = train_feature_da(
        feature_da,
        source_x,
        target_train_x,
        target_y_norm,
        device,
        args.epochs,
        args.batch_size,
        args.lr,
        args.coral_weight,
    )
    da_pred = predict_with_optional_calibration(
        feature_da,
        target_train_x,
        self_bundle.y_train,
        target_test_x,
        bp_mean,
        bp_std,
        device,
        args.calibrate,
        args.calibration_ridge,
    )
    results["feature_da"] = compute_metrics(self_bundle.y_test, da_pred)

    # ---- 可视化: Feature DA 预测散点图 ----
    plot_scatter_comparison(
        self_bundle.y_test,
        {"Base (no adapt)": base_pred, "Feature DA": da_pred},
        f"Feature DA vs Base ({args.shots}-shot)",
        output_dir / f"scatter_feature_da_{args.shots}shot.png",
    )

    # ---- 3. Learning without Forgetting (LwF) ----
    lwf_student = load_model(model_dir, "base_model_best.pth", device)
    teacher = load_model(model_dir, "base_model_best.pth", device)
    lwf_student, lwf_loss = train_lwf(
        lwf_student,
        teacher,
        source_x,
        target_train_x,
        target_y_norm,
        device,
        args.epochs,
        args.batch_size,
        args.lr,
        args.lwf_weight,
    )
    lwf_pred = predict_with_optional_calibration(
        lwf_student,
        target_train_x,
        self_bundle.y_train,
        target_test_x,
        bp_mean,
        bp_std,
        device,
        args.calibrate,
        args.calibration_ridge,
    )
    results["lwf"] = compute_metrics(self_bundle.y_test, lwf_pred)

    # ---- 可视化: LwF 预测散点图 + 三方法汇总对比 ----
    plot_scatter_comparison(
        self_bundle.y_test,
        {"Base (no adapt)": base_pred, "LwF": lwf_pred},
        f"LwF vs Base ({args.shots}-shot)",
        output_dir / f"scatter_lwf_{args.shots}shot.png",
    )

    # 三方法并排对比
    plot_scatter_comparison(
        self_bundle.y_test,
        {"Base": base_pred, "Feature DA": da_pred, "LwF": lwf_pred},
        f"All Methods Comparison ({args.shots}-shot)",
        output_dir / f"scatter_all_{args.shots}shot.png",
    )

    # 训练损失曲线
    plot_loss_curves(
        {"Feature DA": da_loss, "LwF": lwf_loss},
        output_dir / f"loss_curves_{args.shots}shot.png",
    )

    # 指标汇总柱状图
    plot_metric_bars(
        results,
        output_dir / f"metrics_bars_{args.shots}shot.png",
    )

    # ---- 保存: 预测值 (npz) + 指标 (json) ----
    suffix = "_calibrated" if args.calibrate else ""
    np.savez(
        output_dir / f"fewshot_{args.shots}{suffix}_predictions.npz",
        y_true=self_bundle.y_test,
        base_pred=base_pred,
        feature_da_pred=da_pred,
        lwf_pred=lwf_pred,
    )
    with (output_dir / f"fewshot_{args.shots}{suffix}_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def parse_args() -> argparse.Namespace:
    """解析命令行参数, 所有路径均有默认值指向项目本地目录"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", default=str(DEFAULT_PUBLIC_DIR),
                        help="公开数据集目录")
    parser.add_argument("--self-dir", default=str(DEFAULT_SELF_DIR),
                        help="自建数据集目录 (含 PPG cell + BP)")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR),
                        help="预训练模型目录")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="结果输出目录")
    parser.add_argument("--shots", type=int, default=32,
                        help="目标域训练样本数 (few-shot)")
    parser.add_argument("--epochs", type=int, default=40,
                        help="微调训练轮数")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--coral-weight", type=float, default=0.05,
                        help="CORAL 损失权重")
    parser.add_argument("--lwf-weight", type=float, default=0.4,
                        help="LwF 蒸馏损失权重")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fs", type=int, default=125,
                        help="PPG 采样率 (Hz)")
    parser.add_argument("--cpu", action="store_true",
                        help="强制使用 CPU")
    parser.add_argument("--calibrate", action="store_true",
                        help="启用目标域小样本仿射输出校准")
    parser.add_argument("--calibration-ridge", type=float, default=1e-3,
                        help="校准岭回归正则化系数")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        metrics = run(parse_args())
    except FileNotFoundError as exc:
        print(str(exc))
        raise SystemExit(2)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
