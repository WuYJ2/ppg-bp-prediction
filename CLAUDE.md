# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

PPG-based non-invasive blood pressure prediction (SBP / DBP) with four method families across three version generations:

| Version | Language | Method | Branch |
|---------|----------|--------|--------|
| v1.0 | MATLAB | 78 hand-crafted features + ε-SVR | `v1.0` |
| v2.0 DL | Python | 1D-ResNet: public base → self-built fine-tune (real cross-domain) | `v2.0` |
| v2.0 GPR | MATLAB | Gaussian Process Regression with Matérn 5/2 (7 features from v1.0 selection) | `v2.0` |
| v3.0 | Python | Few-shot domain adaptation (CORAL / LwF) | `v3.0` / `master` |

Signal spec: PPG at Fs=125 Hz, 2048 samples per window. Target: SBP + DBP in mmHg.

## Dataset layout

Two datasets in separate directories, both two-level (public + self-built):

- `数据集/1、公开数据集/` — public PPG (4745 train + 1577 val + 1582 test) and pre-extracted 78-d features. Train split 70/30 via `train_split.npz` (seed=42).
- `数据集/2、自建数据集/` — pre-extracted features only (774 train + 273 test). No raw PPG here.
- `dataset/PPG/` — self-built raw PPG waveforms as MATLAB v7.3 cell arrays (`TrainPPG_cell.mat` 253 signals, `TestPPG_cell.mat` 152 signals). Consumed by domain adaptation (v3.0) and cross-domain fine-tuning (v2.0 DL).

The split reason (7:3 on public data) was a temporary workaround: self-built raw PPG waveforms were unavailable during v1.x Python development. Only 78-d features had been pre-extracted by v1.0 MATLAB. The internal split served as a **proof-of-concept** for the retraining paradigm. **As of v2.0 final, the DL pipeline uses real cross-domain: public full training set (4745) → self-built fine-tune set (774).**

## How to run experiments

### Domain adaptation (v3.0, most current)

```bash
# Run all three shot settings (16/32/64):
python domain_adaptation_experiments.py --shots 16
python domain_adaptation_experiments.py --shots 32
python domain_adaptation_experiments.py --shots 64

# Optional: enable output calibration
python domain_adaptation_experiments.py --shots 64 --calibrate

# Visualize results:
python plot_domain_adaptation_results.py
python plot_optimized_results.py      # includes calibration charts
```

Output goes to `cnn_output/domain_adaptation/`: JSON metrics, NPZ predictions, scatter matrices, loss curves, metric bar charts.

### Deep learning (v2.0): public→self-built cross-domain

```bash
python split_data.py              # [已弃用] 仅在复现 v1.x 内部拆分实验时使用
python train_base_model.py        # 公开数据集全量 (4745 样本) 训练基础模型
python fine_tune_model.py         # 自建数据集 (774 样本) 跨域微调 (USE_DISTILL 消融开关)
python evaluate_model.py          # 双数据集评估: 公开测试集 (源域内) + 自建测试集 (跨域)
```

Key ablation: set `USE_DISTILL = False` in `fine_tune_model.py` to disable distillation regularization.
Key change from v1.x: fine-tuning target is now the self-built dataset (real cross-domain), not the internal 7:3 split.

### GPR (MATLAB v2.0)

```matlab
train_gpr         % base model: public features → 7-d selection (from v1.0 SVR)
finetune_gpr      % mixed public + self-built features → 7-d selection
evaluate_gpr      % cross-evaluation on both test sets
```

Feature selection: uses 7 features from v1.0 SVR's `dataSortList = [26,7,32,43,44,23,42]`, covering time (7,23,42), area (43,44), and frequency (26,32) domains. Reduces GP complexity from O(78²) to O(7²) per dimension.

### SVR (MATLAB v1.0)

```matlab
find_all_parameter   % batch feature extraction
train_SBP            % SBP SVR training
train_DBP            % DBP SVR training
find_result          % results summary
```

## Architecture

### Key code dependencies (Python)

`model.py` is the shared infrastructure — every other Python file imports from it:
- `ResNet1D` — the 1D residual network (3-channel input, 256-d GAP bottleneck)
- `load_public_dataset()` — scipy-based loader for standard .mat
- `load_self_dataset()` — h5py-based loader for v7.3 cell arrays (complex HDF5 reference dereferencing)
- `preprocess_ppg()` — computes dPPG + d²PPG, stacks to (N,3,2048)
- `normalize_3ch()` / `normalize_bp()` — per-channel z-score normalization

`domain_adaptation_experiments.py` extends the model:
- `ResNetWithFeatures` inherits `ResNet1D` and exposes `forward_features()` returning the 256-d GAP vector used for CORAL alignment.
- `coral_loss()` computes mean + covariance alignment loss (Frobenius norm on 256×256 matrices).
- `train_feature_da()` and `train_lwf()` implement the two domain-adaptation strategies.

### Data format trap

Self-built PPG is stored as MATLAB v7.3 cell arrays. Each cell holds a 2048×1 signal as HDF5 object references — one reference per sample point. `scipy.io.loadmat` cannot read this format. The `load_self_dataset()` function in `model.py` dereferences each scalar reference individually via h5py. This is the slowest part of the data pipeline.

### Model artifact structure

- `models/base_model_best.pth` — best validation-loss checkpoint
- `models/base_model_final.pth` — final-epoch checkpoint (fallback)
- `models/norm_params.npz` — per-channel means/stds + BP normalization params
- `models/train_split.npz` — [已弃用] 7:3 indices with seed=42 (仅 v1.x 内部拆分实验使用)
- `gpr_models/gpr_base.mat` / `gpr_finetuned.mat` — trained GPR models

### Evaluation convention

All evaluations produce exactly three chart types at 300 DPI:
1. `BA_*.png` — Bland-Altman (mean vs difference, bias line ±1.96SD)
2. `Corr_*.png` — scatter plot with y=x diagonal + linear fit + Pearson r
3. `Line_*.png` — first 80-sample true vs predicted overlay (GPR adds 95% CI band)

Plus `model_comparison.png` for multi-model comparison (4 metrics × N models bar chart).

## Important file-size context

- `find_parameter_amend.m` (861 lines) — the largest single file: 5 sub-functions computing 78 features across 5 feature domains (time 22, area 5, derivative 9, frequency 26, wavelet 17).
- `domain_adaptation_experiments.py` (675 lines) — orchestrates all three DA methods with visualization and JSON/NPZ output.
- `evaluate_gpr.m` (218 lines) — handles 2×2 cross-evaluation (2 models × 2 test sets).

## Domain adaptation method IDs

| Method key | Core idea | Requires source data? |
|---|---|---|
| `base` | No adaptation, direct cross-domain prediction | No |
| `feature_da` | CORAL second-order alignment of GAP features | Yes |
| `lwf` | Teacher soft-label constraint, no source data access | No (teacher model only) |

64-shot Feature DA achieves the best results: SBP MAE 6.21 mmHg, DBP MAE 2.30 mmHg.

## Git LFS

All `.mat` files are tracked via Git LFS. The `is_lfs_pointer()` function in `domain_adaptation_experiments.py` validates data integrity before loading. If LFS files are missing, the script exits early with a clear error.

## Notebooks & PPT

For the academic report and defense PPT, see:
- `reports/项目开发报告/1_final.docx` — final polished DOCX (三线表, 小四 body, template-styled)
- `reports/项目开发报告/PPG无创血压预测项目开发报告.txt` — comprehensive text report
- `reports/项目开发报告/projects/defense_ppt_ppt169_20260622/` — 18-slide defense PPT (1280×720, 1.45× fontSize)
- `reports/项目开发报告/projects/defense_ppt_ppt169_20260622/演讲稿.md` — speaker script
