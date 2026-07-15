%% train_gpr.m
% 使用公开数据集训练 GPR 基础模型 (7 特征精简版)
% 流程: 公开预提取 78 维特征 → 选取 7 维 → GPR 回归
%
% 特征选择: 沿用 v1.0 SVR 方案中确定的最优 7 特征组合
%   编号 [7, 23, 26, 32, 42, 43, 44], 覆盖时域/面积/频域
%
% 依赖: Statistics and Machine Learning Toolbox (fitrgp)

clear; clc;

%% ==================== 配置 ====================
[scriptDir, ~, ~] = fileparts(mfilename('fullpath'));
publicDataPath = fullfile(scriptDir, '数据集', '1、公开数据集');
modelSavePath  = fullfile(scriptDir, 'gpr_models');
outputPath     = fullfile(scriptDir, 'gpr_output');
mkdir(modelSavePath); mkdir(outputPath);

% 选择 7 个特征 (1-indexed, 对应 78 维特征矩阵的列号)
% 来源: v1.0 SVR 方案的 dataSortList = [26,7,32,43,44,23,42]
FEATURE_IDS = [7, 23, 26, 32, 42, 43, 44];  % 排序后
N_FEATURES = length(FEATURE_IDS);

% GPR 核函数: 'squaredexponential' | 'matern32' | 'matern52' | 'rationalquadratic' | 'ardsquaredexponential'
KERNEL = 'matern52';  % Matérn 5/2: 单长度尺度, 2 参数

% 拟合方法: 'exact'(精确) | 'sd'(子集, 快) | 'fic'(稀疏, 快)
FIT_METHOD = 'exact';

fprintf('GPR 核函数: %s, 拟合方法: %s, 特征数: %d\n', KERNEL, FIT_METHOD, N_FEATURES);

%% ==================== 加载特征名称表 ====================
try
    ParameterVerge = table2cell(readtable(fullfile(scriptDir, 'ParameterVerge.xlsx'), ...
        'VariableNamingRule', 'preserve'))';
    ParameterFull = ParameterVerge(1, :);  % 78 个特征名称
    Parameter = ParameterFull(FEATURE_IDS);
catch
    % 硬编码回退: 7 特征名称 (编号 [7,23,26,32,42,43,44])
    Parameter = {'width1', 'SV', 'f2', 'fs4', 'Z', 'SR_DA', 'SR_SA'};
    fprintf('(使用硬编码特征名称)\n');
end
fprintf('选取特征: %s\n', strjoin(Parameter, ', '));

%% ==================== 加载公开特征 + 选取 7 维 ====================
fprintf('=== 加载公开训练集特征 (78→7) ===\n');

% 加载预提取的 78 维特征
trainParam = load(fullfile(publicDataPath, 'TrainParameter.mat'));
trainSBP   = load(fullfile(publicDataPath, 'TrainSBP.mat'));
trainDBP   = load(fullfile(publicDataPath, 'TrainDBP.mat'));

X_train_full = trainParam.TrainParameter;
% 自动检测方向: (N,78) 或 (78,N)
if size(X_train_full, 2) ~= 78 && size(X_train_full, 1) == 78
    X_train_full = X_train_full';
end
% 选取 7 特征
X_train = X_train_full(:, FEATURE_IDS);

Y_sbp = trainSBP.TrainSBP(:);
Y_dbp = trainDBP.TrainDBP(:);

% 处理 Inf/NaN
validIdx = all(isfinite(X_train), 2);
if ~all(validIdx)
    fprintf('移除 %d 个无效样本\n', sum(~validIdx));
    X_train = X_train(validIdx, :);
    Y_sbp = Y_sbp(validIdx);
    Y_dbp = Y_dbp(validIdx);
end

fprintf('有效样本: %d, 特征维度: %d (从78维中选取)\n', size(X_train, 1), size(X_train, 2));
fprintf('SBP 范围: [%.1f, %.1f], DBP 范围: [%.1f, %.1f]\n', ...
    min(Y_sbp), max(Y_sbp), min(Y_dbp), max(Y_dbp));

%% ==================== 特征标准化 ====================
X_mean = mean(X_train, 1);
X_std  = std(X_train, 0, 1);
X_std(X_std == 0) = 1;
X_train_norm = (X_train - X_mean) ./ X_std;

%% ==================== 训练 GPR ====================

% --- SBP ---
fprintf('\n=== 训练 SBP GPR 模型 ===\n');
tic;
gprSBP = fitrgp(X_train_norm, Y_sbp, ...
    'KernelFunction', KERNEL, ...
    'Standardize', false, ...
    'FitMethod', FIT_METHOD, ...
    'PredictMethod', 'exact');
toc;

% --- DBP ---
fprintf('\n=== 训练 DBP GPR 模型 ===\n');
tic;
gprDBP = fitrgp(X_train_norm, Y_dbp, ...
    'KernelFunction', KERNEL, ...
    'Standardize', false, ...
    'FitMethod', FIT_METHOD, ...
    'PredictMethod', 'exact');
toc;

%% ==================== 训练集自评估 ====================
fprintf('\n=== 训练集自评估 ===\n');

[predSBP, ~, ciSBP] = predict(gprSBP, X_train_norm);
[predDBP, ~, ciDBP] = predict(gprDBP, X_train_norm);

fprintf('训练集 SBP: MAE=%.2f, STD=%.2f mmHg\n', ...
    mean(abs(predSBP - Y_sbp)), std(predSBP - Y_sbp));
fprintf('训练集 DBP: MAE=%.2f, STD=%.2f mmHg\n', ...
    mean(abs(predDBP - Y_dbp)), std(predDBP - Y_dbp));

%% ==================== 保存基础模型 ====================
save(fullfile(modelSavePath, 'gpr_base.mat'), ...
    'gprSBP', 'gprDBP', 'X_mean', 'X_std', 'KERNEL', 'Parameter');
fprintf('\n=== 基础模型保存至 %s ===\n', modelSavePath);
