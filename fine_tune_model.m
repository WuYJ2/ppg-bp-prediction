%% fine_tune_model.m
% 在基础模型上使用自建数据集进行小批量个体化微调
% 策略: 冻结浅层权重 + 新旧数据混合批 + 知识蒸馏 (软标签)
%
% 依赖: train_base_model.m 输出的基础模型和归一化参数

clear; clc;

%% ==================== 配置 ====================
[scriptDir, ~, ~] = fileparts(mfilename('fullpath'));
publicDataPath = fullfile(scriptDir, '数据集', '1、公开数据集');
selfDataPath   = fullfile(scriptDir, 'dataset');  % 自建数据: PPG/ + BP/
modelSavePath  = fullfile(scriptDir, 'models');
outputPath     = fullfile(scriptDir, 'cnn_output');
mkdir(modelSavePath); mkdir(outputPath);

% 微调超参数
params.numEpochs        = 50;
params.miniBatchSize    = 32;
params.learnRate        = 1e-4;      % 较小学习率
params.L2Regularization = 1e-5;
params.validationFreq   = 20;
params.gradientThreshold = 5;

% 蒸馏参数
params.distillWeight = 0.4;    % α: 蒸馏损失权重 (0.2-0.7)
params.temperature   = 3.0;    % T: 温度 (2-4)

% 新旧数据混合比 (每批中自建数据占比)
params.selfDataRatio = 0.5;    % 50% 自建 + 50% 公开

% 冻结层名称前缀 (这些层的参数不参与梯度更新)
frozenLayerPrefixes = {
    'conv1', 'bn1',                          % 初始卷积
    'res1_conv1', 'res1_bn1', 'res1_conv2', 'res1_bn2',  % ResBlock 1
    'res2_conv1', 'res2_bn1', 'res2_conv2', 'res2_bn2', 'res2_proj', 'res2_proj_bn'  % ResBlock 2
};

% 信号参数
params.Fs      = 125;
params.sigLen  = 2048;

fprintf('蒸馏权重 α = %.2f, 温度 T = %.1f\n', params.distillWeight, params.temperature);
fprintf('冻结层: %s\n', strjoin(frozenLayerPrefixes, ', '));

%% ==================== 加载基础模型和归一化参数 ====================
fprintf('=== 加载基础模型 ===\n');
baseModelFile = fullfile(modelSavePath, 'base_model_best.mat');
if ~exist(baseModelFile, 'file')
    baseModelFile = fullfile(modelSavePath, 'base_model_final.mat');
end
loaded = load(baseModelFile);
dlnet  = loaded.dlnet;
teacherNet = loaded.dlnet;  % 教师网络 = 冻结的基础模型 (不更新)

% 加载归一化参数
normLoaded = load(fullfile(modelSavePath, 'norm_params.mat'));
normParams = normLoaded.normParams;
bpMean     = normLoaded.bpMean;
bpStd      = normLoaded.bpStd;

fprintf('基础模型加载完成\n');

%% ==================== 加载公开数据 (用于混合批) ====================
fprintf('=== 加载公开数据集 (混合训练用) ===\n');

trainPPG  = load(fullfile(publicDataPath, 'TrainPPG.mat'));
trainSBP  = load(fullfile(publicDataPath, 'TrainSBP.mat'));
trainDBP  = load(fullfile(publicDataPath, 'TrainDBP.mat'));
X_pub_raw = trainPPG.TrainPPG;
Y_pub_sbp = trainSBP.TrainSBP(:);
Y_pub_dbp = trainDBP.TrainDBP(:);

% 预处理公开数据
X_pub_3ch = preprocessPPG_batch(X_pub_raw, params.Fs);
X_pub     = applyNormalize3Ch(X_pub_3ch, normParams);
Y_pub     = [Y_pub_sbp, Y_pub_dbp];
Y_pub_norm = (Y_pub - bpMean) ./ bpStd;

fprintf('公开训练数据: %d 样本\n', size(X_pub_raw, 1));

%% ==================== 加载自建数据 (cell 格式) ====================
fprintf('=== 加载自建数据集 ===\n');

% PPG (cell 格式)
selfPPG = load(fullfile(selfDataPath, 'PPG', 'TrainPPG_cell.mat'));
selfSBP = load(fullfile(selfDataPath, 'BP', 'TrainSBP.mat'));
selfDBP = load(fullfile(selfDataPath, 'BP', 'TrainDBP.mat'));

TrainPPG_cell = selfPPG.TrainPPG_cell;  % 1×N cell, 每个 cell 为 1×2048
Y_self_sbp = selfSBP.TrainSBP(:);
Y_self_dbp = selfDBP.TrainDBP(:);

% Cell 转矩阵
numSelf = length(TrainPPG_cell);
X_self_raw = zeros(numSelf, params.sigLen);
for i = 1:numSelf
    sig = TrainPPG_cell{i};
    if iscell(sig), sig = sig{1}; end
    if length(sig) >= params.sigLen
        X_self_raw(i, :) = sig(1:params.sigLen);
    else
        X_self_raw(i, 1:length(sig)) = sig;
    end
end

% 预处理自建数据
X_self_3ch = preprocessPPG_batch(X_self_raw, params.Fs);
X_self     = applyNormalize3Ch(X_self_3ch, normParams);
Y_self     = [Y_self_sbp, Y_self_dbp];
Y_self_norm = (Y_self - bpMean) ./ bpStd;

fprintf('自建训练数据: %d 样本\n', numSelf);

%% ==================== 冻结浅层 ====================
fprintf('=== 冻结浅层权重 ===\n');

% 获取所有可学习参数名称
learnableNames = dlnet.Learnables.Parameter;

% 标记哪些参数需要训练
isTrainable = true(size(learnableNames));
for i = 1:length(learnableNames)
    paramName = learnableNames(i);
    for j = 1:length(frozenLayerPrefixes)
        if startsWith(paramName, frozenLayerPrefixes{j})
            isTrainable(i) = false;
            break;
        end
    end
end

trainableParams = learnableNames(isTrainable);
fprintf('可训练参数: %d / %d\n', sum(isTrainable), length(learnableNames));
fprintf('冻结参数: %d\n', sum(~isTrainable));

%% ==================== 微调训练 ====================
fprintf('=== 开始微调训练 (蒸馏 + 混合批) ===\n');

% 转为 dlarray
X_pub_dl  = dlarray(single(X_pub),  'CTB');
Y_pub_dl  = dlarray(single(Y_pub_norm'), 'CB');
X_self_dl = dlarray(single(X_self), 'CTB');
Y_self_dl = dlarray(single(Y_self_norm'), 'CB');

numPub  = size(X_pub_dl, 3);
numSelf = size(X_self_dl, 3);

% 每批中各数据集的样本数
batchPub  = ceil(params.miniBatchSize * (1 - params.selfDataRatio));
batchSelf = params.miniBatchSize - batchPub;

numBatches = ceil(max(numPub / batchPub, numSelf / batchSelf));

% Adam 优化器状态
avgGrad   = [];
avgSqGrad = [];

trainLossHist = [];
iteration = 0;
bestLoss  = inf;

for epoch = 1:params.numEpochs
    % 打乱数据
    idxPub  = randperm(numPub);
    idxSelf = randperm(numSelf);

    epochLoss = 0;
    tic;

    for b = 1:numBatches
        % 从公开数据采样
        pubStart = mod((b-1) * batchPub, numPub) + 1;
        pubEnd   = min(pubStart + batchPub - 1, numPub);
        pubIdx   = idxPub(pubStart:pubEnd);

        % 从自建数据采样
        selfStart = mod((b-1) * batchSelf, numSelf) + 1;
        selfEnd   = min(selfStart + batchSelf - 1, numSelf);
        selfIdx   = idxSelf(selfStart:selfEnd);

        dlX_pub  = X_pub_dl(:, :, pubIdx);
        dlY_pub  = Y_pub_dl(:, pubIdx);
        dlX_self = X_self_dl(:, :, selfIdx);
        dlY_self = Y_self_dl(:, selfIdx);

        % 计算损失和梯度 (包含蒸馏)
        [loss, gradients, state] = dlfeval(@distillLoss, ...
            dlX_pub, dlY_pub, dlX_self, dlY_self, ...
            dlnet, teacherNet, trainableParams, ...
            params.distillWeight, params.temperature);
        dlnet.State = state;

        % 梯度裁剪
        gradVec = [];
        for g = 1:length(trainableParams)
            gv = gradients.(trainableParams(g));
            if ~isempty(gv), gradVec = [gradVec; gv(:)]; end
        end
        gradNorm = norm(gradVec);
        if gradNorm > params.gradientThreshold
            scale = params.gradientThreshold / gradNorm;
            for g = 1:length(trainableParams)
                if ~isempty(gradients.(trainableParams(g)))
                    gradients.(trainableParams(g)) = gradients.(trainableParams(g)) * scale;
                end
            end
        end

        % 更新参数 (仅更新可训练部分)
        [dlnet, avgGrad, avgSqGrad] = adamupdate(dlnet, gradients, ...
            avgGrad, avgSqGrad, iteration + 1, params.learnRate);

        iteration = iteration + 1;
        epochLoss = epochLoss + double(extractdata(loss));
    end

    epochLoss = epochLoss / numBatches;
    fprintf('Epoch %d/%d, Loss=%.4f, 耗时 %.1fs\n', ...
        epoch, params.numEpochs, epochLoss, toc);

    trainLossHist = [trainLossHist; epoch, epochLoss];

    % 保存最佳模型
    if epochLoss < bestLoss
        bestLoss = epochLoss;
        save(fullfile(modelSavePath, 'finetuned_model_best.mat'), 'dlnet');
        fprintf('  最佳模型已保存\n');
    end
end

%% ==================== 保存最终微调模型 ====================
save(fullfile(modelSavePath, 'finetuned_model_final.mat'), 'dlnet');
fprintf('=== 微调完成，模型保存至 %s ===\n', modelSavePath);

% 绘制微调曲线
figure('Position', [100, 100, 500, 350]);
plot(trainLossHist(:,1), trainLossHist(:,2), 'b-', 'LineWidth', 1.2);
xlabel('Epoch'); ylabel('Loss');
title('Fine-tuning Loss Curve');
grid on;
saveas(gcf, fullfile(outputPath, 'finetune_curve.png'));
close(gcf);

%% ==================== 辅助函数 ====================

function X_3ch = preprocessPPG_batch(X_raw, Fs)
    dt = 1 / Fs;
    N = size(X_raw, 1);
    L = size(X_raw, 2);
    X_3ch = zeros(3, L, N, 'single');
    for i = 1:N
        ppg = double(X_raw(i, :));
        d1  = gradient(ppg, dt);
        d2  = gradient(d1, dt);
        X_3ch(1, :, i) = single(ppg);
        X_3ch(2, :, i) = single(d1);
        X_3ch(3, :, i) = single(d2);
    end
end

function X_norm = applyNormalize3Ch(X, normParams)
    X_norm = zeros(size(X), 'like', X);
    for c = 1:3
        X_norm(c, :, :) = (X(c, :, :) - normParams.chMean(c)) / normParams.chStd(c);
    end
end

function [loss, gradients, state] = distillLoss(dlX_pub, dlY_pub, dlX_self, dlY_self, ...
        dlnet, teacherNet, trainableParams, alpha, T)

    % === 自建数据: 仅任务损失 ===
    [dlYPred_self, state] = forward(dlnet, dlX_self);
    dlYPred_self = mean(dlYPred_self, 2);
    lossSelf = mse(dlYPred_self, dlY_self);

    % === 公开数据: 任务损失 + 蒸馏损失 ===
    [dlYPred_pub, state] = forward(dlnet, dlX_pub);
    dlYPred_pub = mean(dlYPred_pub, 2);
    taskPub = mse(dlYPred_pub, dlY_pub);

    % 教师预测 (不计算梯度)
    dlYTeacher = mean(predict(teacherNet, dlX_pub), 2);

    % 软标签蒸馏损失: T² * MSE(pred/T, teacher/T)
    distillPub = mse(dlYPred_pub / T, dlYTeacher / T) * T^2;

    % 综合损失
    loss = lossSelf + (1 - alpha) * taskPub + alpha * distillPub;

    % 仅对可训练参数计算梯度
    gradients = dlgradient(loss, dlnet.Learnables);

    % 将不可训练参数的梯度置零
    learnableNames = dlnet.Learnables.Parameter;
    for i = 1:length(learnableNames)
        if ~ismember(learnableNames(i), trainableParams)
            gradients.(learnableNames(i)) = [];
        end
    end
end
