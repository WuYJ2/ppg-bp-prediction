%% train_gpr.m
% 使用公开数据集训练 GPR 基础模型
% 流程: 公开 PPG → find_parameter_amend 提取 78 维特征 → GPR 回归
%
% 依赖: find_parameter_amend.m, ParameterVerge.xlsx
%       Statistics and Machine Learning Toolbox (fitrgp)

clear; clc;

%% ==================== 配置 ====================
[scriptDir, ~, ~] = fileparts(mfilename('fullpath'));
publicDataPath = fullfile(scriptDir, '数据集', '1、公开数据集');
modelSavePath  = fullfile(scriptDir, 'gpr_models');
outputPath     = fullfile(scriptDir, 'gpr_output');
mkdir(modelSavePath); mkdir(outputPath);

% GPR 核函数: 'squaredexponential' | 'matern32' | 'matern52' | 'rationalquadratic' | 'ardsquaredexponential'
% 注: ARD 核在 78 特征 × 4745 样本下极慢 (79 超参数 × 200 次迭代)
KERNEL = 'matern52';  % Matérn 5/2: 单长度尺度, 2 参数, 较快

% 拟合方法: 'exact'(精确) | 'sd'(子集, 快) | 'fic'(稀疏, 快)
FIT_METHOD = 'exact';

fprintf('GPR 核函数: %s, 拟合方法: %s\n', KERNEL, FIT_METHOD);

%% ==================== 加载特征名称表 ====================
ParameterVerge = table2cell(readtable(fullfile(scriptDir, 'ParameterVerge.xlsx'), ...
    'VariableNamingRule', 'preserve'))';
Parameter = ParameterVerge(1, :);  % 78 个特征名称

%% ==================== 加载公开 PPG + 提取特征 ====================
fprintf('=== 加载公开训练集特征 ===\n');

% 直接加载预提取的 78 维特征
trainParam = load(fullfile(publicDataPath, 'TrainParameter.mat'));
trainSBP   = load(fullfile(publicDataPath, 'TrainSBP.mat'));
trainDBP   = load(fullfile(publicDataPath, 'TrainDBP.mat'));

X_train = trainParam.TrainParameter;
% 自动检测方向: (N,78) 或 (78,N)
if size(X_train, 2) ~= 78 && size(X_train, 1) == 78
    X_train = X_train';
end
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

fprintf('有效样本: %d, 特征维度: %d\n', size(X_train, 1), size(X_train, 2));
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
