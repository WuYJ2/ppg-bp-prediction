# PPG 无创血压预测项目

基于光电容积脉搏波 (PPG) 信号预测收缩压 (SBP) 和舒张压 (DBP)，包含两套方案：
- **传统方案**: 手工特征提取 + SVR 回归 (MATLAB)
- **深度学习方案**: 1D-ResNet 端到端预测 (PyTorch)

## 项目结构

```
.
├── model.py                   # [共享] 1D-ResNet 模型定义 + 数据加载 + 预处理
├── train_base_model.py        # [DL] 基础模型训练 (70% 公开数据)
├── fine_tune_model.py         # [DL] 微调 (冻结浅层 + 新旧混合 + 可选蒸馏)
├── evaluate_model.py          # [DL] 评估 (预测值 / MAE / STD / 图表)
├── split_data.py              # [工具] 公开数据 7:3 拆分
│
├── train_base_model.m         # [DL-MATLAB] 基础模型训练 (已废弃, 仅供参考)
├── fine_tune_model.m          # [DL-MATLAB] 蒸馏微调 (已废弃, 仅供参考)
├── evaluate_model.m           # [DL-MATLAB] 评估 (已废弃, 仅供参考)
│
├── find_all_parameter.m       # [主程序] 批量特征提取脚本
├── find_parameter_amend.m     # [核心函数] 单条 PPG 信号的 78 维特征计算
├── fparameter_n78.m           # [备选函数] 78 维特征的另一种实现
├── train_DBP.m / train_SBP.m  # SVR 血压预测模型训练
├── find_result.m              # SVR 训练结果汇总分析
│
├── fselect_x.m                # 预处理: 去除波形异常段 (尖峰检测)
├── fdenoise_x.m               # 预处理: 基线漂移 + 高频噪声去除
├── fmoveaverage_x.m           # 预处理: 移动平均滤波
├── fabnormal_x.m              # 预处理: 异常样本剔除
│
├── SVMcgForRegress.m          # SVM 参数网格搜索 (交叉验证)
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
├── dataset/                   # 自建数据集 (MATLAB cell 格式)
│   ├── PPG/
│   ├── BP/
│   └── Parameter/
│
├── 数据集/
│   ├── 1、公开数据集/         # 基础模型训练 + 验证 + 测试
│   └── 2、自建数据集/         # 特征 + 标签 (无原始 PPG, 未使用)
│
├── models/                    # 模型文件 (.pth) + 归一化参数 + 拆分索引
├── cnn_output/                # 评估图表 + 预测值 → 见下方图表注释
├── result/                    # SVR 模型训练结果 (MATLAB)
└── output/                    # 特征提取结果 (MATLAB)
```

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

1. 加载公开训练集, 按 7:3 拆分 (首次运行自动生成 `models/train_split.npz`)
2. 预处理: 计算一阶/二阶导数 → 3 通道输入 (PPG + dPPG + d²PPG)
3. z-score 归一化 (保存统计量至 `models/norm_params.npz`)
4. 1D-ResNet 架构:

```
Input (3, 2048)
  → Conv1 (7×1, 64, stride 2) → BN → ReLU → MaxPool (3, stride 2)  → (64, 512)
  → ResBlock1 [3×1, 64] ×2                               → (64, 512)
  → ResBlock2 [3×1, 128] ×2, stride 2, 1×1 projection    → (128, 256)
  → ResBlock3 [3×1, 256] ×2, stride 2, 1×1 projection    → (256, 128)
  → GlobalAvgPool                                         → (256,)
  → FC (2)                                                → (SBP, DBP)
```

5. 训练: Adam (lr=1e-3, step decay 0.5/25epoch), MSE loss, 梯度裁剪
6. 输出: `models/base_model_best.pth`

### 2. 微调 (`fine_tune_model.py`)

| 配置项 | 说明 |
|--------|------|
| `USE_DISTILL` | **消融实验开关**: `True`=蒸馏微调, `False`=无蒸馏 (仅冻结+混合) |
| `DISTILL_WEIGHT` | α: 蒸馏损失权重 (0.2-0.7, 默认 0.4) |
| `TEMPERATURE` | T: 温度 (2-4, 默认 3.0) |
| `FT_DATA_RATIO` | 每批中新数据占比 (默认 0.5) |
| `FROZEN_MODULES` | 冻结 `conv1 + bn1 + layer1 + layer2` |

**损失函数**:

- 蒸馏模式: `L = L_task(new) + (1-α)×L_task(old) + α×T²×MSE(pred/T, teacher/T)`
- 消融模式: `L = L_task(new) + L_task(old)`

### 3. 评估 (`evaluate_model.py`)

- 加载基础模型和/或微调模型 (`EVAL_MODE = 'base' | 'finetuned' | 'both'`)
- 在公开测试集上预测 SBP/DBP
- 计算 MAE / STD
- 生成图表 (见下方注释)

## cnn_output/ 图表注释

所有图表命名规则: `{类型}_{模型}_{数据集}.png`

### 训练曲线

| 文件 | 说明 |
|------|------|
| `base_training_curve.png` | 基础模型训练曲线: 蓝色=训练 Loss, 红色=验证 Loss (对数坐标), x=迭代次数 |
| `finetune_curve.png` | 微调训练曲线: 蓝色=每个 epoch 的平均 Loss |

### Bland-Altman 一致性分析 (`BA_*.png`)

每张图左右并排展示 SBP 和 DBP 的 Bland-Altman 图:
- **x 轴**: 真实值与预测值的均值 (mmHg)
- **y 轴**: 差值 Predicted - True (mmHg)
- 蓝色散点: 每个测试样本
- 红色实线: 平均偏差 (mean difference)
- 红色虚线: 95% 一致性界限 (mean ± 1.96×SD)

| 文件 | 模型 | 测试集 |
|------|------|--------|
| `BA_base_test.png` | 基础模型 | 公开测试集 |
| `BA_finetuned_test.png` | 微调模型 | 公开测试集 |
| `BA_base_public.png` | 基础模型 | 公开测试集 (旧版) |
| `BA_finetuned_public.png` | 微调模型 | 公开测试集 (旧版) |
| `BA_base_self.png` | 基础模型 | 自建测试集 (已废弃) |
| `BA_finetuned_self.png` | 微调模型 | 自建测试集 (已废弃) |

### 相关性散点图 (`Corr_*.png`)

每张图左右并排展示 SBP 和 DBP 的相关性:
- **x 轴**: 真实值 (mmHg)
- **y 轴**: 预测值 (mmHg)
- 黑色虚线: y=x 理想对角线
- 红色实线: 最小二乘拟合线
- 右下角: Pearson 相关系数 r

| 文件 | 模型 | 测试集 |
|------|------|--------|
| `Corr_base_test.png` | 基础模型 | 公开测试集 |
| `Corr_finetuned_test.png` | 微调模型 | 公开测试集 |
| `Corr_base_public.png` | 基础模型 | 公开测试集 (旧版) |
| `Corr_finetuned_public.png` | 微调模型 | 公开测试集 (旧版) |
| `Corr_base_self.png` | 基础模型 | 自建测试集 (已废弃) |
| `Corr_finetuned_self.png` | 微调模型 | 自建测试集 (已废弃) |

### 预测值折线图 (`Line_*.png`)

每张图左右并排展示 SBP 和 DBP 前 80 个样本:
- **蓝色实线**: 真实值
- **红色实线**: 预测值
- x 轴: 样本序号

| 文件 | 模型 | 测试集 |
|------|------|--------|
| `Line_base_test.png` | 基础模型 | 公开测试集 |
| `Line_finetuned_test.png` | 微调模型 | 公开测试集 |
| `Line_base_public.png` | 基础模型 | 公开测试集 (旧版) |
| `Line_finetuned_public.png` | 微调模型 | 公开测试集 (旧版) |
| `Line_base_self.png` | 基础模型 | 自建测试集 (已废弃) |
| `Line_finetuned_self.png` | 微调模型 | 自建测试集 (已废弃) |

### 模型对比

| 文件 | 说明 |
|------|------|
| `model_comparison.png` | 基础模型 vs 微调模型: 4 个子图分别对比 SBP_MAE, SBP_STD, DBP_MAE, DBP_STD |

### 预测值数据

| 文件 | 内容 |
|------|------|
| `predictions_base.npz` | 基础模型预测: `y_true`, `y_pred`, `res` |
| `predictions_finetuned.npz` | 微调模型预测: `y_true`, `y_pred`, `res` |

---

## 传统方案 (MATLAB SVR)

### 特征提取
**`find_all_parameter.m`** 读取 `dataset/PPG/`, 调用 `find_parameter_amend()` 计算 78 维特征, 输出至 `output/`。

**78 维特征分类:**

| 类别 | 特征编号 | 数量 | 说明 |
|------|---------|------|------|
| 时间域 | 1-8, 12-13, 23-24, 36-42, 45-46 | 22 | 收缩/舒张时间、脉宽、周期、波峰幅值等 |
| 面积 | 9-11, 43-44 | 5 | 升支/降支面积及其比值 |
| 一阶微分 | 14-22 | 9 | 变化速率相关特征 |
| 频域 | 25-35, 64-78 | 26 | 基频/谐波、频带能量及能量比 |
| 小波变换 | 47-63 | 17 | 细节系数能量、IMF 能量矩、HHT 边际谱能量 |

### SVR 模型训练
**`train_SBP.m`** / **`train_DBP.m`**: 特征选择 → 归一化 → ε-SVR (径向基核) → MAE/STD → Bland-Altman + 相关性图, 结果保存至 `result/`。

### 预处理 (可选)
| 函数 | 功能 |
|------|------|
| `fselect_x(signal)` | 检测并去除尖峰异常段 |
| `fdenoise_x(signal)` | 低通滤波 + 基线漂移去除 |
| `fmoveaverage_x(wine)` | Hamming 窗移动平均平滑 |
| `fabnormal_x(wine)` | 基于特征均值阈值剔除异常样本 |

---

## 分支管理

| 分支 | 说明 |
|------|------|
| `v1.0` | 初始版本: MATLAB SVR + CNN, 自建数据微调 |
| `v1.1` | 数据策略改为公开数据集 7:3 内部分割 |
| `v1.2` | 消融实验: 新增 `USE_DISTILL` 开关, 关闭蒸馏 |
| `master` | 开发主线 (当前 = v1.2) |

---

## 依赖项

### 深度学习方案
- **Python 3.9+**
- **PyTorch** 2.x (CUDA 推荐)
- **numpy, scipy, h5py** — 数据加载
- **matplotlib** — 图表绘制

### 传统方案 (MATLAB)
- **MATLAB** R2019b+
- **LIBSVM** / **Wavelet Toolbox** / **Signal Processing Toolbox** / **Statistics and ML Toolbox**

---

## 使用方法

### 深度学习方案

```bash
python split_data.py            # 1. 生成数据拆分 (可选, 训练时自动)
python train_base_model.py      # 2. 基础模型训练 (70% Train)
python fine_tune_model.py       # 3. 微调 (30% Train, 消融/蒸馏)
python evaluate_model.py        # 4. 评估, 图表输出至 cnn_output/
```

### 传统方案 (MATLAB)

1. 运行 `find_all_parameter.m` 提取特征
2. 运行 `train_SBP.m` / `train_DBP.m` 训练 SVR
3. 运行 `find_result.m` 汇总结果
