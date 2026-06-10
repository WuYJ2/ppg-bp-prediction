# PPG 无创血压预测项目

基于光电容积脉搏波 (PPG) 信号预测收缩压 (SBP) 和舒张压 (DBP)，包含三套方案：
- **传统方案**: 手工特征提取 + SVR 回归 (MATLAB)
- **深度学习方案**: 1D-ResNet 端到端预测 (PyTorch)
- **GPR 方案**: 手工特征 + 高斯过程回归 (MATLAB)

## 项目结构

```
.
├── model.py                   # [DL] 1D-ResNet 模型定义 + 数据加载 + 预处理
├── train_base_model.py        # [DL] 基础模型训练 (70% 公开数据)
├── fine_tune_model.py         # [DL] 微调 (冻结浅层 + 新旧混合 + 可选蒸馏)
├── evaluate_model.py          # [DL] 评估 (预测值 / MAE / STD / 图表)
├── split_data.py              # [工具] 公开数据 7:3 拆分
│
├── train_gpr.m                # [GPR] 公开特征 → GPR 基础模型训练
├── finetune_gpr.m             # [GPR] 公开+自建混合 → GPR 微调
├── evaluate_gpr.m             # [GPR] 双数据集评估 (base vs finetuned)
│
├── find_all_parameter.m       # [主程序] 批量特征提取脚本
├── find_parameter_amend.m     # [核心函数] 单条 PPG 信号的 78 维特征计算
├── fparameter_n78.m           # [备选函数] 78 维特征的另一种实现
├── train_DBP.m / train_SBP.m  # SVR 血压预测模型训练
├── find_result.m              # SVR 训练结果汇总分析
│
├── fselect_x.m                # 预处理: 去除波形异常段
├── fdenoise_x.m               # 预处理: 基线漂移 + 高频噪声去除
├── fmoveaverage_x.m           # 预处理: 移动平均滤波
├── fabnormal_x.m              # 预处理: 异常样本剔除
│
├── SVMcgForRegress.m          # SVM 参数网格搜索
├── bin.m / GetPath.py         # 工具: 格式转换 / 批量创建目录
│
├── emd.m / eemd.m             # 经验模态分解 / 集合经验模态分解
├── hhspectrum.m / toimage.m   # Hilbert-Huang 谱计算与可视化
├── extrema.m / tftb.m         # 极值点查找 / 时频分析窗函数
│
├── ParameterVerge.xlsx        # 78 个特征名称表
├── base.mat / lowpass.mat     # 去噪滤波器系数
├── parameter.mat              # 特征参考数据
│
├── dataset/                   # 自建数据集 (PPG cell + BP 标签)
│   ├── PPG/
│   ├── BP/
│   └── Parameter/
│
├── 数据集/
│   ├── 1、公开数据集/         # 公开 PPG + 预提取特征 + BP 标签
│   └── 2、自建数据集/         # 预提取特征 + BP 标签 (无 PPG)
│
├── models/                    # 1D-ResNet 模型 (.pth) + 归一化参数 + 拆分索引
├── cnn_output/                # 1D-ResNet 评估图表 + 预测值
├── gpr_models/                # GPR 模型 (.mat)
├── gpr_output/                # GPR 评估图表 + 预测值
├── result/                    # SVR 模型训练结果 (MATLAB)
└── output/                    # 特征提取结果 (MATLAB)
```

## GPR 高斯过程回归方案 (MATLAB)

### 数据流

```
数据集/1、公开数据集/
├── TrainParameter (4745, 78) → GPR 基础训练 (train_gpr.m)
├── TestParameter  (1582, 78) → 公开测试评估
数据集/2、自建数据集/
├── TrainParameter (774, 78)  ─┐
└── TestParameter  (273, 78)   │ 微调评估
                                │
公开 TrainParameter ────────────┼→ 混合 → GPR 微调 (finetune_gpr.m)
```

### 1. 基础模型训练 (`train_gpr.m`)

直接使用公开数据集的预提取 78 维特征训练 GPR:

1. 加载 `TrainParameter.mat` (4745×78)
2. z-score 归一化
3. 训练 GPR: `fitrgp` + Matérn 5/2 核 + exact 拟合
4. 自评估 MAE/STD
5. 输出: `gpr_models/gpr_base.mat`

### 2. 微调 (`finetune_gpr.m`)

混合公开和自建特征重新训练 GPR:

- 公开旧数据: TrainParameter (4745×78) → 提取特征
- 自建新数据: TrainParameter (774×78) → 直接加载
- 合并标准化 → `fitrgp` (Matérn 5/2, sd 子集近似)
- 输出: `gpr_models/gpr_finetuned.mat`

### 3. 评估 (`evaluate_gpr.m`)

在公开 + 自建两个测试集上分别评估 base 和 finetuned 模型:

- 输出 MAE / STD
- 生成图表: Bland-Altman / 相关性散点图 / 预测折线图 (含 GPR 95% CI)
- 对比柱状图 (双数据集, base vs finetuned)

## gpr_output/ 图表注释

图表命名: `{类型}_{模型}_{数据集}.png`

| 类型 | 说明 |
|------|------|
| `BA_*.png` | Bland-Altman: x=均值, y=差值, 红线=偏差±1.96SD |
| `Corr_*.png` | 相关性散点图: 含 y=x 对角 + 拟合线 + Pearson r |
| `Line_*.png` | 预测折线图: 前 80 样本, 含 GPR 95% 置信区间 |
| `model_comparison.png` | 双数据集 bar chart: base vs finetuned 的 4 指标 |

| 模型 | 数据集 |
|------|--------|
| `base` / `finetuned` | `pub` (公开测试集) / `self` (自建测试集) |

---

## 深度学习方案 (PyTorch)

### 数据流

```
数据集/1、公开数据集/
├── TrainPPG (4745, 2048) ─┬─ 70% → 基础训练 (train_base_model.py)
│                          └─ 30% → 微调 (fine_tune_model.py)
├── ValPPG   (1577, 2048) → 验证 (基础训练)
└── TestPPG  (1582, 2048) → 最终评估 (evaluate_model.py)
```

### 1. 基础模型训练 (`train_base_model.py`)

1. 加载公开训练集, 按 7:3 拆分
2. 预处理: 一阶/二阶导数 → 3 通道 (PPG + dPPG + d²PPG)
3. z-score 归一化
4. 1D-ResNet 架构:

```
Input (3, 2048)
  → Conv1 (7×1, 64, stride 2) → BN → ReLU → MaxPool (3, stride 2)
  → ResBlock1 [3×1, 64] ×2
  → ResBlock2 [3×1, 128] ×2, stride 2, 1×1 proj
  → ResBlock3 [3×1, 256] ×2, stride 2, 1×1 proj
  → GlobalAvgPool → FC (2) → SBP, DBP
```

5. 训练: Adam (lr=1e-3, step decay), MSE, 梯度裁剪
6. 输出: `models/base_model_best.pth`

### 2. 微调 (`fine_tune_model.py`)

| 配置项 | 说明 |
|--------|------|
| `USE_DISTILL` | 消融开关: True=蒸馏, False=仅冻结+混合 |
| `FROZEN_MODULES` | 冻结 conv1+bn1+layer1+layer2 |
| `FT_DATA_RATIO` | 每批中新数据占比 (0.5) |

### 3. 评估 (`evaluate_model.py`)

在公开测试集上评估 base / finetuned 模型, 生成 BA + Corr + Line 图表。

## cnn_output/ 图表注释

| 文件 | 说明 |
|------|------|
| `base_training_curve.png` | 基础模型训练 Loss 曲线 |
| `finetune_curve.png` | 微调 Loss 曲线 |
| `BA_{model}_test.png` | Bland-Altman 一致性分析 |
| `Corr_{model}_test.png` | 相关性散点图 |
| `Line_{model}_test.png` | 预测值折线图 |
| `model_comparison.png` | Base vs Fine-tuned 对比柱状图 |
| `predictions_{model}.npz` | 预测值数据 |

---

## 传统方案 (MATLAB SVR)

### 特征提取
`find_all_parameter.m` 读取 PPG, 调用 `find_parameter_amend()` 计算 78 维特征。

**78 维特征分类:**

| 类别 | 特征编号 | 数量 | 说明 |
|------|---------|------|------|
| 时间域 | 1-8, 12-13, 23-24, 36-42, 45-46 | 22 | 收缩/舒张时间、脉宽、周期、波峰幅值 |
| 面积 | 9-11, 43-44 | 5 | 升支/降支面积及其比值 |
| 一阶微分 | 14-22 | 9 | 变化速率相关特征 |
| 频域 | 25-35, 64-78 | 26 | 基频/谐波、频带能量及能量比 |
| 小波变换 | 47-63 | 17 | 细节系数能量、IMF 能量矩、HHT 边际谱 |

### SVR 训练
`train_SBP.m` / `train_DBP.m`: 特征选择 → 归一化 → ε-SVR → MAE/STD → Bland-Altman + 相关性图。

---

## 分支管理

| 分支 | 说明 |
|------|------|
| `v1.0` | 初始版本: MATLAB SVR + DL, 自建数据微调 |
| `v1.1` | 数据策略改为公开数据集 7:3 内部分割 |
| `v1.2` | 消融实验: 新增 `USE_DISTILL` 开关 |
| `v2.0` | GPR 方案: 公开特征 + 自建特征, 双数据集评估 |
| `master` | 开发主线 (当前 = v2.0) |

---

## 依赖项

### GPR 方案 (MATLAB)
- **MATLAB** R2019b+
- **Statistics and Machine Learning Toolbox** — `fitrgp`, `predict`

### 深度学习方案
- **Python 3.9+**, **PyTorch** 2.x (CUDA 推荐)
- **numpy, scipy, h5py, matplotlib**

### 传统方案 (MATLAB)
- **MATLAB** R2019b+
- **LIBSVM** / **Wavelet Toolbox** / **Signal Processing Toolbox**

---

## 使用方法

### GPR 方案（推荐）
```matlab
train_gpr         % 1. 基础模型 (公开特征)
finetune_gpr      % 2. 微调 (公开+自建混合)
evaluate_gpr      % 3. 评估 (双数据集对比)
```

### 深度学习方案
```bash
python train_base_model.py      # 基础训练
python fine_tune_model.py       # 微调 (消融/蒸馏)
python evaluate_model.py        # 评估
```

### 传统方案 (MATLAB)
```matlab
find_all_parameter              % 特征提取
train_SBP / train_DBP           % SVR 训练
find_result                     % 结果汇总
```
