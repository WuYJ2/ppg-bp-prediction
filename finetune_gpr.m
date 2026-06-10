%% finetune_gpr.m
% 使用自建数据集对 GPR 模型进行再训练 (微调)
% 自建数据已有 78 维 Parameter 特征, 直接训练 GPR
%
% 策略: 新旧数据混合训练 GPR, 使模型同时保留公开数据知识并适配自建数据

clear; clc;

%% ==================== 配置 ====================
[scriptDir, ~, ~] = fileparts(mfilename('fullpath'));
publicDataPath = fullfile(scriptDir, '数据集', '1、公开数据集');
selfDataPath   = fullfile(scriptDir, '数据集', '2、自建数据集');
modelSavePath  = fullfile(scriptDir, 'gpr_models');
outputPath     = fullfile(scriptDir, 'gpr_output');
mkdir(modelSavePath); mkdir(outputPath);

% 加载基础模型及特征名称表
baseModel = load(fullfile(modelSavePath, 'gpr_base.mat'));
Parameter = baseModel.Parameter;
KERNEL    = baseModel.KERNEL;

FIT_METHOD = 'sd';  % 子集近似, 5519 样本用 exact 太慢
fprintf('GPR 核函数: %s, 拟合方法: %s\n', KERNEL, FIT_METHOD);

%% ==================== 提取公开数据特征 (作为旧数据) ====================
fprintf('=== 提取公开数据特征 (旧数据) ===\n');

trainParam = load(fullfile(publicDataPath, 'TrainParameter.mat'));
trainSBP   = load(fullfile(publicDataPath, 'TrainSBP.mat'));
trainDBP   = load(fullfile(publicDataPath, 'TrainDBP.mat'));

X_pub = trainParam.TrainParameter;
if size(X_pub, 2) ~= 78 && size(X_pub, 1) == 78
    X_pub = X_pub';
end
Y_pub_sbp = trainSBP.TrainSBP(:);
Y_pub_dbp = trainDBP.TrainDBP(:);

validPub = all(isfinite(X_pub), 2);
X_pub = X_pub(validPub, :);
Y_pub_sbp = Y_pub_sbp(validPub);
Y_pub_dbp = Y_pub_dbp(validPub);
fprintf('公开旧数据: %d 样本\n', size(X_pub, 1));

%% ==================== 加载自建数据特征 (作为新数据) ====================
fprintf('=== 加载自建数据特征 (新数据) ===\n');

selfParam = load(fullfile(selfDataPath, 'TrainParameter.mat'));
selfSBP   = load(fullfile(selfDataPath, 'TrainSBP.mat'));
selfDBP   = load(fullfile(selfDataPath, 'TrainDBP.mat'));

X_self_raw = selfParam.TrainParameter;
if size(X_self_raw, 2) == 78
    X_self = X_self_raw;
elseif size(X_self_raw, 1) == 78
    X_self = X_self_raw';
else
    error('特征矩阵形状异常: %dx%d', size(X_self_raw, 1), size(X_self_raw, 2));
end
Y_self_sbp = selfSBP.TrainSBP(:);
Y_self_dbp = selfDBP.TrainDBP(:);

validSelf = all(isfinite(X_self), 2);
X_self = X_self(validSelf, :);
Y_self_sbp = Y_self_sbp(validSelf);
Y_self_dbp = Y_self_dbp(validSelf);
fprintf('自建新数据: %d 样本\n', size(X_self, 1));

%% ==================== 标准化 ====================
% 使用所有数据 (公开+自建) 计算标准化参数
X_all = [X_pub; X_self];
X_mean = mean(X_all, 1);
X_std  = std(X_all, 0, 1);
X_std(X_std == 0) = 1;

X_pub_norm  = (X_pub  - X_mean) ./ X_std;
X_self_norm = (X_self - X_mean) ./ X_std;

%% ==================== 混合训练 GPR ====================
% 将新旧数据合并训练，使 GPR 同时学习两个分布
X_mix = [X_pub_norm; X_self_norm];
Y_mix_sbp = [Y_pub_sbp; Y_self_sbp];
Y_mix_dbp = [Y_pub_dbp; Y_self_dbp];

fprintf('\n=== 训练 SBP GPR 模型 (混合数据: %d 样本) ===\n', length(Y_mix_sbp));
tic;
gprSBP = fitrgp(X_mix, Y_mix_sbp, ...
    'KernelFunction', KERNEL, ...
    'Standardize', false, ...
    'FitMethod', FIT_METHOD, ...
    'PredictMethod', 'exact');
toc;

fprintf('\n=== 训练 DBP GPR 模型 (混合数据: %d 样本) ===\n', length(Y_mix_dbp));
tic;
gprDBP = fitrgp(X_mix, Y_mix_dbp, ...
    'KernelFunction', KERNEL, ...
    'Standardize', false, ...
    'FitMethod', FIT_METHOD, ...
    'PredictMethod', 'exact');
toc;

%% ==================== 自评估 ====================
fprintf('\n=== 训练集自评估 ===\n');
predSBP = predict(gprSBP, X_mix);
predDBP = predict(gprDBP, X_mix);

fprintf('SBP: MAE=%.2f, STD=%.2f mmHg\n', ...
    mean(abs(predSBP - Y_mix_sbp)), std(predSBP - Y_mix_sbp));
fprintf('DBP: MAE=%.2f, STD=%.2f mmHg\n', ...
    mean(abs(predDBP - Y_mix_dbp)), std(predDBP - Y_mix_dbp));

%% ==================== 保存微调模型 ====================
save(fullfile(modelSavePath, 'gpr_finetuned.mat'), ...
    'gprSBP', 'gprDBP', 'X_mean', 'X_std', 'KERNEL');
fprintf('\n=== 微调模型保存至 %s ===\n', modelSavePath);
