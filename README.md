# PPG 无创血压预测项目

基于光电容积脉搏波 (PPG) 信号预测收缩压 (SBP) 和舒张压 (DBP)，包含两套方案：

- **传统方案**: 手工特征提取 + SVR 回归
- **深度学习方案**: 1D-ResNet 端到端预测 (基础训练 + 蒸馏微调)

## 项目结构

```
.
├── find_all_parameter.m      # [主程序] 批量特征提取脚本
├── find_parameter_amend.m    # [核心函数] 单条 PPG 信号的 78 维特征计算
├── fparameter_n78.m          # [备选函数] 78 维特征的另一种实现
├── train_DBP.m               # 舒张压 SVR 模型训练脚本
├── train_SBP.m               # 收缩压 SVR 模型训练脚本
├── find_result.m             # 训练结果汇总分析脚本
│
├── fselect_x.m               # 预处理: 去除波形异常段（尖峰检测）
├── fdenoise_x.m              # 预处理: 基线漂移 + 高频噪声去除
├── fmoveaverage_x.m          # 预处理: 移动平均滤波
├── fabnormal_x.m             # 预处理: 异常样本剔除
│
├── SVMcgForRegress.m         # SVM 参数网格搜索（交叉验证）
├── bin.m                     # 工具: .bin → .mat 格式转换
├── GetPath.py                # 工具: 批量创建输出目录
│
├── train_base_model.m        # [DL] 1D-ResNet 基础模型训练 (公开数据集)
├── fine_tune_model.m         # [DL] 蒸馏微调 (冻结浅层 + 新旧混合 + 知识蒸馏)
├── evaluate_model.m          # [DL] 模型评估 (预测值/MAE/STD/Bland-Altman/相关性)
│
├── emd.m / eemd.m            # 经验模态分解 / 集合经验模态分解
├── hhspectrum.m / toimage.m  # Hilbert-Huang 谱计算与可视化
├── extrema.m                 # 极值点查找（EMD 依赖）
├── tftb.m                    # 时频分析窗函数库
│
├── ParameterVerge.xlsx       # 78 个特征名称表
├── base.mat / lowpass.mat    # 去噪滤波器系数
├── parameter.mat             # 特征参考数据
│
├── dataset/
│   ├── PPG/                  # PPG 信号 (cell 格式)，主程序从此读取
│   │   ├── TrainPPG_cell.mat
│   │   └── TestPPG_cell.mat
│   ├── Parameter/            # 提取后的 78 维特征矩阵
│   │   ├── TrainParameter.mat
│   │   └── TestParameter.mat
│   └── BP/                   # 血压标签 (SBP / DBP)
│       ├── TrainSBP.mat / TrainDBP.mat
│       └── TestSBP.mat / TestDBP.mat
│
├── data/                     # [用户准备] 额外待提取特征的原始 .mat 文件
├── output/                   # [自动生成] 特征提取结果
├── result/                   # [自动生成] SVR 模型训练结果与评估图
│   ├── DBP/
│   └── SBP/
├── models/                   # [自动生成] 1D-ResNet 模型文件
├── cnn_output/               # [自动生成] CNN 评估图与预测结果
│
└── 数据集/                   # 深度学习数据集
    ├── 1、公开数据集/        # 基础模型训练数据
    │   ├── TrainPPG.mat / TrainSBP.mat / TrainDBP.mat
    │   ├── ValPPG.mat   / ValSBP.mat   / ValDBP.mat
    │   └── TestPPG.mat  / TestSBP.mat  / TestDBP.mat
    └── 2、自建数据集/        # 微调数据 (特征 + 标签)
```

## 工作流程

### 1. 特征提取

**主程序 `find_all_parameter.m`** 批量读取 `dataset/PPG/` 目录下的 PPG 信号文件，调用 `find_parameter_amend()` 计算 78 维特征，结果保存至 `output/`。

```
find_all_parameter.m
  └── find_parameter_amend(Parameter, data)
        ├── find_time()     → 特征 1-8, 12-13, 23-24, 36-42, 45-46  (22 个时间特征)
        ├── find_area()     → 特征 9-11, 43-44                         (5 个面积特征)
        ├── find_diff1()    → 特征 14-22                               (9 个一阶微分特征)
        ├── find_freq()     → 特征 25-35, 64-78                        (26 个频域特征)
        └── find_WT()       → 特征 47-63                               (17 个小波特征)
```

**78 维特征分类:**

| 类别   | 特征编号                            | 数量  | 说明                       |
| ---- | ------------------------------- | --- | ------------------------ |
| 时间域  | 1-8, 12-13, 23-24, 36-42, 45-46 | 22  | 收缩/舒张时间、脉宽、周期、波峰幅值等      |
| 面积   | 9-11, 43-44                     | 5   | 升支/降支面积及其比值              |
| 一阶微分 | 14-22                           | 9   | 变化速率相关特征                 |
| 频域   | 25-35, 64-78                    | 26  | 基频/谐波、频带能量及能量比           |
| 小波变换 | 47-63                           | 17  | 细节系数能量、IMF 能量矩、HHT 边际谱能量 |

### 2. 信号预处理 (可选)

| 函数                     | 功能                     |
| ---------------------- | ---------------------- |
| `fselect_x(signal)`    | 检测并去除波形中幅值异常的尖峰段       |
| `fdenoise_x(signal)`   | 使用低通滤波和基线滤波去除高频噪声和基线漂移 |
| `fmoveaverage_x(wine)` | 对特征矩阵做 Hamming 窗移动平均平滑 |
| `fabnormal_x(wine)`    | 基于特征均值阈值剔除异常样本         |

### 3. 模型训练

**`train_SBP.m`** 和 **`train_DBP.m`** 分别训练收缩压和舒张压预测模型:

1. 加载训练/测试特征 (`dataset/Parameter/`) 和血压标签 (`dataset/BP/`)
2. 特征选择 (从 78 维中选取指定子集)
3. 归一化 (mapminmax, 扩展 15% 裕量)
4. SVR 训练 (`svmtrain`, ε-SVR, 径向基核)
5. 测试集评估 (MAE, STD)
6. 输出可视化:
   - 预测值 vs 真实值对比折线图
   - Bland-Altman 一致性分析图
   - 相关性散点图 (含拟合线与相关系数)

结果保存至 `result/DBP/` 和 `result/SBP/`，包含训练好的模型、归一化参数、评估指标和 300 DPI 高清图。

### 4. 结果汇总

**`find_result.m`** 遍历 `result/` 目录，汇总各特征组合的 MAE 和 STD，便于横向比较。

---

### 5. 深度学习方案 (1D-ResNet)

#### 5.1 基础模型训练 (`train_base_model.m`)

在公开数据集上训练 1D-ResNet:

1. 加载 PPG 信号 (2048 点, 125 Hz)
2. 预处理: 计算一阶/二阶导数, 堆叠为 3 通道 (C×T×B)
3. z-score 归一化 (保存统计量供后续复用)
4. 构建 1D-ResNet:
   - Conv1 (7×1, 64 filters, stride 2) → MaxPool (3×1, stride 2)
   - ResBlock1 (64 ch, ×2 conv) → ResBlock2 (128 ch, stride 2) → ResBlock3 (256 ch, stride 2)
   - GlobalAvgPool → FC (256→2, SBP+DBP)
5. 自定义训练循环 (Adam, 梯度裁剪, L2 正则, 阶梯学习率)
6. 验证集监控，保存最佳模型至 `models/base_model_best.mat`

#### 5.2 蒸馏微调 (`fine_tune_model.m`)

在自建数据集上小批量个体化再训练:

| 策略        | 说明                                                                                   |
| --------- | ------------------------------------------------------------------------------------ |
| **冻结浅层**  | Conv1 + ResBlock1 + ResBlock2 权重冻结，仅训练 ResBlock3 + FC                                |
| **新旧混合**  | 每批 50% 公开数据 + 50% 自建数据，保持通用知识不遗忘                                                     |
| **知识蒸馏**  | 基础模型作为教师，软标签监督: L = L_task(self) + (1-α)×L_task(pub) + α×T²×L_soft(pub/T, teacher/T) |
| **无 EWC** | 当前版本未启用弹性权重巩固                                                                        |

默认蒸馏参数: α=0.4, T=3.0。自建 PPG 数据 (`dataset/PPG/`) 为 cell 格式，自动转换为矩阵。

#### 5.3 评估 (`evaluate_model.m`)

在公开 + 自建测试集上评估:

- 输出 SBP/DBP 预测值 (保存为 `predictions_*.mat`)
- 计算 MAE / STD
- 生成图表:
  - Bland-Altman 一致性分析 (SBP + DBP 并排)
  - 相关性散点图 (含拟合线 + Pearson r)
  - 预测值 vs 真实值折线图
- 可选对比模式 (`evalMode = 'both'`): 基础模型 vs 微调模型横向对比柱状图

所有图表保存至 `cnn_output/`，300 DPI。

## 依赖项

### 传统方案

- **MATLAB** (R2019b 或更高版本)
- **LIBSVM** — `svmtrain`, `svmpredict`
- **Wavelet Toolbox** — `wavedec`, `wrcoef`
- **Signal Processing Toolbox** — `findpeaks`, `filter`, `sgolayfilt`
- **Statistics and Machine Learning Toolbox** — `nchoosek`, `mapminmax`

### 深度学习方案

- **MATLAB** (R2021b 或更高版本)
- **Deep Learning Toolbox** — `dlnetwork`, `adamupdate`, `dlarray`
- **Signal Processing Toolbox** — `gradient` (导数计算)
- 推荐 GPU (NVIDIA CUDA)，自动检测并使用

## 使用方法

### 特征提取

1. 将原始 PPG 信号的 `.mat` 文件放入 `dataset/PPG/` 目录（每条信号长度为 2048 点，采样率 125 Hz）
2. 在项目根目录下运行 `find_all_parameter.m`
3. 提取的特征矩阵保存在 `output/` 目录

### 模型训练

1. 确保 `dataset/` 目录包含特征文件和血压标签文件
2. 运行 `train_SBP.m` 训练收缩压模型
3. 运行 `train_DBP.m` 训练舒张压模型
4. 查看 `result/SBP/` 和 `result/DBP/` 中的评估结果

### 结果分析

运行 `find_result.m` 汇总并对比不同特征组合的预测性能。

### 深度学习方案

1. 确保 `数据集/1、公开数据集/` 包含 PPG 和 BP 标签文件，`dataset/PPG/` 和 `dataset/BP/` 包含自建数据
2. 运行 `train_base_model` 训练基础 1D-ResNet
3. 运行 `fine_tune_model` 进行蒸馏微调
4. 运行 `evaluate_model` 评估并生成图表，设置 `evalMode = 'both'` 可对比基础模型与微调模型
