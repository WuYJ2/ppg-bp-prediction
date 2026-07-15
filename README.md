# PPG 无创血压预测项目

基于光电容积脉搏波 (PPG) 信号预测收缩压 (SBP) 和舒张压 (DBP)，包含四条技术路线：

| 版本 | 方法 | 语言 | 核心特点 |
|------|------|:----:|---------|
| v1.0 | 手工 78 维特征 + ε-SVR | MATLAB | 特征工程基线，奠定特征体系 |
| v2.0 DL | 1D-ResNet 端到端学习 | Python | 公开全量训练→自建跨域微调（真实跨域） |
| v2.0 GPR | 高斯过程回归 (7 特征) | MATLAB | 特征级跨域迁移，预测不确定性量化 |
| v3.0 DA | 小样本域适应（CORAL / LwF） | Python | 目标域仅需 32~64 个标注样本 |

详细技术方案见 [项目总结.md](项目总结.md)。

---

## 项目结构

```
.
├── model.py                        # [DL] 共享基础设施: ResNet1D + 数据加载 + 预处理
├── train_base_model.py             # [DL] 基础训练: 公开全量 4745 → base_model.pth
├── fine_tune_model.py              # [DL] 跨域微调: 公开 4745 + 自建 253 (冻结+混合批+蒸馏)
├── evaluate_model.py               # [DL] 双数据集评估: 公开 + 自建
├── split_data.py                   # [已弃用] 公开数据 7:3 拆分 (仅复现 v1.x 时使用)

├── train_gpr.m                     # [GPR] 基础训练: 公开 4745×7 → Matérn 5/2 GP
├── finetune_gpr.m                  # [GPR] 跨域微调: 公开 4745 + 自建 774 = 5519×7 联合训练
├── evaluate_gpr.m                  # [GPR] 2×2 交叉评估: 双测试集 × 双模型

├── domain_adaptation_experiments.py  # [DA] 小样本域适应实验 (CORAL + LwF)
├── plot_domain_adaptation_results.py # [DA] 域适应结果绘图
├── plot_optimized_results.py          # [DA] 优化版结果绘图 (含校准)
├── make_best_scatter.py               # [DA] 最优结果精修散点图
├── build_word_report.py               # [DA] 实验报告生成 (Word)
├── build_optimized_word_report.py     # [DA] 优化版报告生成

├── find_all_parameter.m          # [v1.0] 批量特征提取脚本
├── find_parameter_amend.m        # [v1.0] 单条 PPG 的 78 维特征计算 (861 行)
├── fparameter_n78.m              # [v1.0] 78 维特征备选实现
├── train_SBP.m / train_DBP.m     # [v1.0] SVR 血压预测训练
├── find_result.m                 # [v1.0] SVR 结果汇总

├── fselect_x.m / fdenoise_x.m    # 预处理: 异常尖峰去除 / 基线漂移+噪声
├── fmoveaverage_x.m             # 预处理: 移动平均滤波
├── fabnormal_x.m                 # 预处理: 异常样本剔除
├── SVMcgForRegress.m             # SVR 超参数网格搜索

├── emd.m / eemd.m                # 经验模态分解 / 集合经验模态分解
├── hhspectrum.m / toimage.m      # Hilbert-Huang 谱
├── extrema.m / tftb.m            # 极值点查找 / 时频分析窗函数

├── ParameterVerge.xlsx           # 78 特征名称表
├── base.mat / lowpass.mat        # 去噪滤波器系数

├── dataset/                      # 自建数据集 (v7.3 cell 原始 PPG)
│   ├── PPG/                      #   TrainPPG_cell.mat (253), TestPPG_cell.mat (152)
│   ├── BP/                       #   Train/Test SBP/DBP 标签
│   └── Parameter/                #   预提取 78 维特征 (Train 774, Test 273)

├── 数据集/
│   ├── 1、公开数据集/            # 公开 PPG + 78 维特征 + BP (Train 4745, Val 1577, Test 1582)
│   └── 2、自建数据集/            # 预提取 78 维特征 + BP (Train 774, Test 273)

├── models/                       # 1D-ResNet 模型 (.pth) + 归一化参数
├── cnn_output/                   # DL 评估图表 + 预测值
│   └── domain_adaptation/        # 域适应实验结果
├── gpr_models/                   # GPR 模型 (.mat)
├── gpr_output/                   # GPR 评估图表 + 预测值
├── reports/                      # 实验报告 (.docx)
├── result/                       # SVR 训练结果
└── output/                       # 特征提取结果
```

---

## 深度学习方案 (v2.0 DL, Python)

### 数据流

```
公开数据集 TrainPPG (4745×2048) ──→ train_base_model.py ──→ base_model.pth
                                                              ↓
公开 TrainPPG (4745) 源域 + 自建 PPG (253) 目标域 ──→ fine_tune_model.py ──→ finetuned_model.pth
                                                              ↓
公开 TestPPG (1582) + 自建 TestPPG (152) ──→ evaluate_model.py ──→ 双数据集评估报告
```

### 1. 基础训练 (`train_base_model.py`)

- 使用公开数据集全量 4745 样本训练，**不再**做内部 7:3 拆分
- 预处理：三通道 (PPG + dPPG + d²PPG) → 独立 z-score 归一化 → BP 联合归一化
- 1D-ResNet (≈500K 参数)：

```
Input (B, 3, 2048)
  → Conv1d(3→64, k=7, s=2, pad=3) → BN → ReLU → MaxPool(k=3, s=2)
  → Layer1: 2×BasicBlock(64→64,  s=1)
  → Layer2: 2×BasicBlock(64→128, s=2)  (1×1 proj)
  → Layer3: 2×BasicBlock(128→256, s=2) (1×1 proj)
  → AdaptiveAvgPool1d(1) → FC(256→2) → [SBP, DBP]
```

- 超参数：Adam(lr=1e-3), 80 epochs, StepLR(step=25, γ=0.5), weight_decay=1e-4, grad_clip=5.0

### 2. 跨域微调 (`fine_tune_model.py`)

三种机制组合：

| 机制 | 说明 |
|------|------|
| **参数冻结** | `requires_grad=False` for conv1/bn1/layer1/layer2 |
| **混合批采样** | 每批 16 源域 + 16 目标域 (1:1) |
| **知识蒸馏** | 教师模型冻结, T=3.0, α=0.4 (可通过 `USE_DISTILL` 开关消融) |

### 3. 评估 (`evaluate_model.py`)

- 双数据集评估：公开测试集 (源域内) + 自建测试集 (跨域核心指标)
- 输出：MAE/STD + Bland-Altman / Correlation / Line 三类图表
- 双模型对比柱状图 (base vs finetuned)

---

## GPR 方案 (v2.0 GPR, MATLAB)

### 特征选择

从 v1.0 的 78 维特征中选取 7 维最优子集（沿用 v1.0 SVR 的 `dataSortList`）：

| 编号 | 名称 | 域 | 生理含义 |
|:---:|------|:---:|---------|
| 7 | width1 | 时域 | 2/3 峰值脉宽 |
| 23 | SV | 时域 | 每搏输出量估算 |
| 26 | f2 | 频域 | FFT 二次谐波 |
| 32 | fs4 | 频域 | 能量比指标 |
| 42 | Z | 时域 | 每搏输出量指数 |
| 43 | SR_DA | 面积 | 舒张面积占比 |
| 44 | SR_SA | 面积 | 收缩面积占比 |

### 数据流

```
公开 TrainParameter (4745×78) → 选取 7 维 (4745×7) → z-score → fitrgp(exact) → gpr_base.mat
                                                                               ↓
公开 4745×7 + 自建 774×7 = 5519×7 → 联合 z-score → fitrgp(sd) → gpr_finetuned.mat
                                                                               ↓
公开 TestParameter (1582×78→7) + 自建 TestParameter (273×78→7) → evaluate_gpr.m → 2×2 评估
```

### 核函数

Matérn 5/2（二阶均方可微，适合生理信号的中等平滑性先验）。78→7 特征降维使 GP 训练从分钟级降至秒级。

### 输出

- 预测值 + **预测标准差**（95% 置信区间）
- Line 图叠加 CI 阴影带（GPR 独有特性）
- 2 测试集 × 2 模型 × 3 类图 + model_comparison 汇总

---

## 域适应方案 (v3.0 DA, Python)

### 数据流

```
公开 PPG (4745, 源域) ──→ Base 1D-ResNet ──→ 跨域基线
自建 PPG (253, 目标域) ──→ Few-shot (16/32/64) ──→ Feature DA / LwF ──→ 评估
```

### 方法

| 方法 | 原理 | 损失函数 |
|------|------|---------|
| **Base** | 无适应 | — |
| **Feature DA** | CORAL 对齐 (均值+协方差) | MSE + λ·CORAL(F_s, F_t) |
| **LwF** | 教师模型软标签约束 | MSE + λ·MSE(student, teacher) |

### 运行

```bash
python domain_adaptation_experiments.py --shots 64
python domain_adaptation_experiments.py --shots 64 --calibrate
python plot_domain_adaptation_results.py
python make_best_scatter.py
```

---

## 关键实验数据

### 1D-ResNet 跨域微调 (公开→自建, 蒸馏)

| 模型 | 测试集 | SBP MAE | SBP STD | DBP MAE | DBP STD |
|------|--------|:---:|:---:|:---:|:---:|
| base | 公开 (1582) | 2.42 | 3.61 | 1.30 | 2.03 |
| base | 自建 (152) | 12.87 | 11.61 | 3.85 | 4.13 |
| finetuned | 自建 (152) | **6.92** | 7.81 | **2.27** | 2.53 |

### GPR 跨域迁移 (78→7 特征)

| 模型 | 测试集 | SBP MAE | SBP STD | DBP MAE | DBP STD |
|------|--------|:---:|:---:|:---:|:---:|
| base | 公开 (1582) | 3.19 | 5.11 | 1.68 | 2.86 |
| base | 自建 (273) | 9.17 | 10.93 | 17.82 | 8.12 |
| finetuned | 自建 (273) | **3.89** | 6.05 | **2.21** | 3.45 |

### 小样本域适应 (64-shot)

| 方法 | SBP MAE | SBP STD | DBP MAE | DBP STD |
|------|:---:|:---:|:---:|:---:|
| Base (无适应) | 11.36 | 12.07 | 7.18 | 4.48 |
| **Feature DA 64-shot** 🏆 | **6.21** | 7.31 | **2.30** | 2.62 |
| LwF 64-shot | 8.02 | 9.07 | 3.31 | 3.07 |

---

## 图表命名规范

### DL (`cnn_output/`)

| 文件名 | 内容 |
|--------|------|
| `BA_{model}_{dataset}.png` | Bland-Altman: x=均值, y=差值, 偏倚±1.96SD |
| `Corr_{model}_{dataset}.png` | 相关性: y=x 对角 + 拟合线 + Pearson r |
| `Line_{model}_{dataset}.png` | 前 80 样本折线对比 (蓝=真, 红=预测) |
| `model_comparison_{public,self-built}.png` | 双数据集 base vs finetuned 柱状图 |
| `predictions_{model}.npz` | 预测值 + 评估指标 |

模型: `base`, `finetuned` | 数据集: `public` (公开), `self` (自建)

### GPR (`gpr_output/`)

| 文件名 | 内容 |
|--------|------|
| `{BA,Corr,Line}_{model}_{dataset}.png` | 同上, Line 图含 95% CI 阴影带 |
| `model_comparison.png` | 双数据集 bar chart |

模型: `base`, `finetuned` | 数据集: `pub` (公开), `self` (自建)

### DA (`cnn_output/domain_adaptation/`)

| 文件名 | 内容 |
|--------|------|
| `scatter_all_{N}shot.png` | 三方法散点矩阵 |
| `loss_curves_{N}shot.png` | Feature DA + LwF 损失曲线 |
| `metrics_bars_{N}shot.png` | 三方法指标对比 |
| `fewshot_{N}_results.json` | 评估指标 JSON |

---

## 运行方法

### DL (Python)

```bash
python train_base_model.py                           # 1. 基础训练
python fine_tune_model.py                            # 2. 微调 (可设 USE_DISTILL=True/False)
python evaluate_model.py                             # 3. 双数据集评估
```

### GPR (MATLAB)

```matlab
train_gpr                                            % 1. 基础训练 (公开 7 特征)
finetune_gpr                                         % 2. 跨域微调 (公开+自建 7 特征)
evaluate_gpr                                         % 3. 2×2 交叉评估
```

### DA (Python)

```bash
python domain_adaptation_experiments.py --shots 32   # 32-shot 实验
python domain_adaptation_experiments.py --shots 64   # 64-shot 实验
python plot_domain_adaptation_results.py             # 结果可视化
```

### SVR (MATLAB)

```matlab
find_all_parameter                                   % 特征提取
train_SBP / train_DBP                                % SVR 训练
find_result                                          % 结果汇总
```

---

## 依赖

| 方案 | 环境 | 关键依赖 |
|------|------|---------|
| DL (Python) | Python 3.9+ | PyTorch 2.x, numpy, scipy, h5py, matplotlib |
| GPR (MATLAB) | R2019b+ | Statistics and Machine Learning Toolbox (`fitrgp`) |
| DA (Python) | Python 3.9+ | PyTorch 2.x, numpy, scipy, h5py, matplotlib |
| SVR (MATLAB) | R2019b+ | LIBSVM, Signal Processing Toolbox, Wavelet Toolbox |

```bash
pip install torch numpy scipy h5py matplotlib
```

---

## 分支管理

| 分支 | 说明 |
|------|------|
| `v1.0` | MATLAB SVR + 78 维特征提取 |
| `v1.1` | 公开数据集 7:3 内部拆分 (已弃用) |
| `v1.2` | 消融实验: `USE_DISTILL` 开关 |
| `v2.0` | GPR 跨域迁移 + 1D-ResNet 公开→自建微调 |
| `v3.0` | 域适应: Feature DA / LwF 小样本迁移 |
| `master` | 开发主线 (当前 = v3.0) |

---

更多技术细节和实验数据见 [项目总结.md](项目总结.md)。
