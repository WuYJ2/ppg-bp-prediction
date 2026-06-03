%% train_base_model.m
% 使用公开数据集训练 1D-ResNet 基础模型
% 输入: 原始 PPG + 一阶导数 + 二阶导数 (3 通道)
% 输出: SBP / DBP (双输出回归)
%
% 依赖: Deep Learning Toolbox, Signal Processing Toolbox

clear; clc;

%% ==================== 配置 ====================
% 数据集路径
[scriptDir, ~, ~] = fileparts(mfilename('fullpath'));
publicDataPath = fullfile(scriptDir, '数据集', '1、公开数据集');
modelSavePath   = fullfile(scriptDir, 'models');
outputPath      = fullfile(scriptDir, 'cnn_output');
mkdir(modelSavePath); mkdir(outputPath);

% 训练超参数
params.numEpochs       = 80;
params.miniBatchSize   = 64;
params.initialLearnRate = 1e-3;
params.learnRateDropFactor = 0.5;
params.learnRateDropEpoch  = 25;
params.L2Regularization    = 1e-4;
params.validationFreq  = 50;   % 每 N 次迭代做一次验证
params.gradientThreshold = 5;

% 信号参数
params.Fs   = 125;       % 采样率 (Hz)
params.sigLen = 2048;    % 信号长度 (点)

% 模型参数
params.numChannels = 3;   % PPG + 一阶导 + 二阶导
params.numOutputs  = 2;   % SBP + DBP

%% ==================== 加载数据 ====================
fprintf('=== 加载公开数据集 ===\n');

% 训练集
trainPPG  = load(fullfile(publicDataPath, 'TrainPPG.mat'));
trainSBP  = load(fullfile(publicDataPath, 'TrainSBP.mat'));
trainDBP  = load(fullfile(publicDataPath, 'TrainDBP.mat'));
X_train_raw = trainPPG.TrainPPG;          % (N, 2048)
Y_train_sbp = trainSBP.TrainSBP(:);       % (N, 1)
Y_train_dbp = trainDBP.TrainDBP(:);

% 验证集
valPPG    = load(fullfile(publicDataPath, 'ValPPG.mat'));
valSBP    = load(fullfile(publicDataPath, 'ValSBP.mat'));
valDBP    = load(fullfile(publicDataPath, 'ValDBP.mat'));
X_val_raw = valPPG.ValPPG;
Y_val_sbp = valSBP.ValSBP(:);
Y_val_dbp = valDBP.ValDBP(:);

% 测试集 (公开)
testPPG   = load(fullfile(publicDataPath, 'TestPPG.mat'));
testSBP   = load(fullfile(publicDataPath, 'TestSBP.mat'));
testDBP   = load(fullfile(publicDataPath, 'TestDBP.mat'));
X_test_raw = testPPG.TestPPG;
Y_test_sbp = testSBP.TestSBP(:);
Y_test_dbp = testDBP.TestDBP(:);

fprintf('训练集: %d, 验证集: %d, 测试集: %d\n', ...
    size(X_train_raw,1), size(X_val_raw,1), size(X_test_raw,1));

%% ==================== 预处理 ====================
fprintf('=== 数据预处理 (计算导数 + 归一化) ===\n');

% 计算一阶和二阶导数，并堆叠为 3 通道
[X_train_3ch, X_train_raw] = preprocessPPG(X_train_raw, params.Fs);
[X_val_3ch,   ~]            = preprocessPPG(X_val_raw,   params.Fs);
[X_test_3ch,  ~]            = preprocessPPG(X_test_raw,  params.Fs);

% 对每通道做 z-score 归一化 (使用训练集统计量)
[X_train, normParams] = normalize3Ch(X_train_3ch);
X_val   = applyNormalize3Ch(X_val_3ch,   normParams);
X_test  = applyNormalize3Ch(X_test_3ch,  normParams);

% BP 标签 z-score 归一化
Y_train = [Y_train_sbp, Y_train_dbp];
Y_val   = [Y_val_sbp,   Y_val_dbp];
Y_test  = [Y_test_sbp,  Y_test_dbp];

bpMean = mean(Y_train, 1);
bpStd  = std(Y_train, 0, 1);
Y_train_norm = (Y_train - bpMean) ./ bpStd;
Y_val_norm   = (Y_val   - bpMean) ./ bpStd;
Y_test_norm  = (Y_test  - bpMean) ./ bpStd;

% 保存归一化参数，供微调和评估使用
save(fullfile(modelSavePath, 'norm_params.mat'), ...
    'normParams', 'bpMean', 'bpStd');

fprintf('PPG 归一化: 均值范围 [%.4f, %.4f], 标准差范围 [%.4f, %.4f]\n', ...
    min(normParams.chMean), max(normParams.chMean), ...
    min(normParams.chStd),  max(normParams.chStd));
fprintf('SBP: mean=%.2f, std=%.2f | DBP: mean=%.2f, std=%.2f\n', ...
    bpMean(1), bpStd(1), bpMean(2), bpStd(2));

%% ==================== 构建 1D-ResNet ====================
fprintf('=== 构建 1D-ResNet 网络 ===\n');
lgraph = buildResNet1D(params.sigLen, params.numChannels, params.numOutputs);
dlnet  = dlnetwork(lgraph);
fprintf('1D-ResNet 构建完成, 可学习参数: %d\n', ...
    sum(cellfun(@numel, dlnet.Learnables.Value)));
    %sum(cellfun(@numel, dlnet.Learnables.Value));

%% ==================== 训练 ====================
fprintf('=== 开始训练 ===\n');

% 将数据转为 dlarray 格式
X_train_dl = dlarray(single(X_train), 'CTB');  % (C, T, B)
Y_train_dl = dlarray(single(Y_train_norm'), 'CB');  % (2, B)
X_val_dl   = dlarray(single(X_val),   'CTB');
Y_val_dl   = dlarray(single(Y_val_norm'), 'CB');

numSamples  = size(X_train_dl, 3);
numBatches  = ceil(numSamples / params.miniBatchSize);

% Adam 优化器状态
avgGrad  = [];
avgSqGrad = [];

% 训练记录
trainLossHist = [];
valLossHist   = [];
valIterHist   = [];
bestValLoss   = inf;
bestEpoch     = 0;
iteration     = 0;

for epoch = 1:params.numEpochs
    % 学习率阶梯下降
    if mod(epoch, params.learnRateDropEpoch) == 0 && epoch > 0
        params.initialLearnRate = params.initialLearnRate * params.learnRateDropFactor;
        fprintf('--- Epoch %d: 学习率降至 %.2e ---\n', epoch, params.initialLearnRate);
    end

    % 打乱训练数据
    idx = randperm(numSamples);
    X_shuf = X_train_dl(:, :, idx);
    Y_shuf = Y_train_dl(:, idx);

    epochLoss = 0;
    tic;

    for b = 1:numBatches
        startIdx = (b-1) * params.miniBatchSize + 1;
        endIdx   = min(b * params.miniBatchSize, numSamples);

        dlX = X_shuf(:, :, startIdx:endIdx);
        dlY = Y_shuf(:, startIdx:endIdx);

        % 计算损失和梯度
        [loss, gradients, state] = dlfeval(@modelLoss, dlX, dlY, dlnet);
        dlnet.State = state;

        % 梯度裁剪
        gradVec = [];
        gradNames = fieldnames(gradients);
        for g = 1:numel(gradNames)
            gv = gradients.(gradNames{g});
            if ~isempty(gv)
                gradVec = [gradVec; gv(:)];
            end
        end
        gradNorm = norm(gradVec);
        if gradNorm > params.gradientThreshold
            scale = params.gradientThreshold / gradNorm;
            for g = 1:numel(gradNames)
                if ~isempty(gradients.(gradNames{g}))
                    gradients.(gradNames{g}) = gradients.(gradNames{g}) * scale;
                end
            end
        end

        % 更新参数
        [dlnet, avgGrad, avgSqGrad] = adamupdate(dlnet, gradients, ...
            avgGrad, avgSqGrad, iteration + 1, params.initialLearnRate);

        iteration = iteration + 1;
        epochLoss = epochLoss + double(extractdata(loss));

        % 定期验证
        if mod(iteration, params.validationFreq) == 0
            valLoss = computeValLoss(dlnet, X_val_dl, Y_val_dl);
            trainLossHist = [trainLossHist; iteration, epochLoss / b];
            valLossHist   = [valLossHist;   iteration, valLoss];
            valIterHist   = [valIterHist;   iteration];

            if valLoss < bestValLoss
                bestValLoss = valLoss;
                bestEpoch = epoch;
                save(fullfile(modelSavePath, 'base_model_best.mat'), 'dlnet');
                fprintf('  [Iter %d] 最佳模型已保存, ValLoss=%.4f\n', iteration, valLoss);
            end
        end
    end

    epochLoss = epochLoss / numBatches;
    fprintf('Epoch %d/%d, Loss=%.4f, 耗时 %.1fs\n', ...
        epoch, params.numEpochs, epochLoss, toc);

    % 每个 epoch 末验证
    valLoss = computeValLoss(dlnet, X_val_dl, Y_val_dl);
    fprintf('  ValLoss=%.4f, BestValLoss=%.4f (Epoch %d)\n', ...
        valLoss, bestValLoss, bestEpoch);
end

%% ==================== 保存最终模型 ====================
save(fullfile(modelSavePath, 'base_model_final.mat'), 'dlnet');
fprintf('=== 训练完成，模型保存至 %s ===\n', modelSavePath);

% 绘制训练曲线
figure('Position', [100, 100, 600, 400]);
semilogy(trainLossHist(:,1), trainLossHist(:,2), 'b-', 'LineWidth', 1);
hold on;
semilogy(valIterHist, valLossHist(:,2), 'r-', 'LineWidth', 1);
xlabel('Iteration'); ylabel('Loss (MSE)');
legend('Training Loss', 'Validation Loss');
title('1D-ResNet Base Model Training Curve');
grid on;
saveas(gcf, fullfile(outputPath, 'base_training_curve.png'));
close(gcf);

%% ==================== 辅助函数 ====================

function [X_3ch, X_raw] = preprocessPPG(X_raw, Fs)
    % 输入: X_raw: (N, L) — N 样本, L 信号长度
    % 输出: X_3ch: (3, L, N) — 3 通道 (PPG, 一阶导, 二阶导)
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

function [X_norm, normParams] = normalize3Ch(X)
    % 对 3 通道分别做 z-score 归一化
    % X: (3, L, N)
    normParams.chMean = zeros(3, 1);
    normParams.chStd  = zeros(3, 1);
    X_norm = zeros(size(X), 'like', X);

    for c = 1:3
        chData = squeeze(X(c, :, :));
        chData = chData(:);
        normParams.chMean(c) = mean(chData);
        normParams.chStd(c)  = std(chData);
        X_norm(c, :, :) = (X(c, :, :) - normParams.chMean(c)) / normParams.chStd(c);
    end
end

function X_norm = applyNormalize3Ch(X, normParams)
    X_norm = zeros(size(X), 'like', X);
    for c = 1:3
        X_norm(c, :, :) = (X(c, :, :) - normParams.chMean(c)) / normParams.chStd(c);
    end
end

function [loss, gradients, state] = modelLoss(dlX, dlY, dlnet)
    % 前向传播 + MSE 损失
    [dlYPred, state] = forward(dlnet, dlX);
    dlYPred = mean(dlYPred, 2);  % 沿 T 维度求均值折叠 T=1
    loss = mse(dlYPred, dlY);

    % L2 正则化
    learnables = dlnet.Learnables;
    l2Loss = dlarray(0);
    for i = 1:height(learnables)
        if contains(learnables.Parameter(i), 'Weights')
            l2Loss = l2Loss + sum(learnables.Value{i}(:).^2);
        end
    end
    loss = loss + 1e-4 * l2Loss;

    gradients = dlgradient(loss, dlnet.Learnables);
end

function valLoss = computeValLoss(dlnet, X_val, Y_val)
    dlYPred = predict(dlnet, X_val);
    dlYPred = squeeze(dlYPred);  % 去掉 T=1 维度
    valLoss = double(extractdata(mse(dlYPred, Y_val)));
end

function lgraph = buildResNet1D(inputLen, numCh, numOut)
    % 构建 1D ResNet (3 个残差块)
    % 输入形状: (numCh, inputLen) CTB 格式

    % 从输入层开始创建空图层 (仅含1层, 避免自动串联)
    lgraph = layerGraph(sequenceInputLayer(numCh, ...
        'Name', 'input', 'Normalization', 'none'));

    % 逐组添加所有层
    mainLayers = [
        convolution1dLayer(7, 64, 'Stride', 2, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        maxPooling1dLayer(3, 'Stride', 2, 'Padding', 'same', 'Name', 'pool1')

        convolution1dLayer(3, 64, 'Padding', 'same', 'Name', 'res1_conv1')
        batchNormalizationLayer('Name', 'res1_bn1')
        reluLayer('Name', 'res1_relu1')
        convolution1dLayer(3, 64, 'Padding', 'same', 'Name', 'res1_conv2')
        batchNormalizationLayer('Name', 'res1_bn2')

        convolution1dLayer(3, 128, 'Stride', 2, 'Padding', 'same', 'Name', 'res2_conv1')
        batchNormalizationLayer('Name', 'res2_bn1')
        reluLayer('Name', 'res2_relu1')
        convolution1dLayer(3, 128, 'Padding', 'same', 'Name', 'res2_conv2')
        batchNormalizationLayer('Name', 'res2_bn2')

        convolution1dLayer(3, 256, 'Stride', 2, 'Padding', 'same', 'Name', 'res3_conv1')
        batchNormalizationLayer('Name', 'res3_bn1')
        reluLayer('Name', 'res3_relu1')
        convolution1dLayer(3, 256, 'Padding', 'same', 'Name', 'res3_conv2')
        batchNormalizationLayer('Name', 'res3_bn2')

        globalAveragePooling1dLayer('Name', 'gap')
        fullyConnectedLayer(2, 'Name', 'fc_output')
    ];

    projLayers = [
        convolution1dLayer(1, 128, 'Stride', 2, 'Name', 'res2_proj')
        batchNormalizationLayer('Name', 'res2_proj_bn')
        convolution1dLayer(1, 256, 'Stride', 2, 'Name', 'res3_proj')
        batchNormalizationLayer('Name', 'res3_proj_bn')
    ];

    addBlocks = [
        additionLayer(2, 'Name', 'res1_add')
        additionLayer(2, 'Name', 'res2_add')
        additionLayer(2, 'Name', 'res3_add')
    ];

    reluEnds = [
        reluLayer('Name', 'res1_relu2')
        reluLayer('Name', 'res2_relu2')
        reluLayer('Name', 'res3_relu2')
    ];

    % 逐个添加所有层 (避免 addLayers 对数组自动串联)
    allLayers = [mainLayers(:); projLayers(:); addBlocks(:); reluEnds(:)];
    for i = 1:length(allLayers)
        lgraph = addLayers(lgraph, allLayers(i));
    end

    % 主路径顺序连接
    lgraph = connectLayers(lgraph, 'input',      'conv1');
    lgraph = connectLayers(lgraph, 'conv1',      'bn1');
    lgraph = connectLayers(lgraph, 'bn1',        'relu1');
    lgraph = connectLayers(lgraph, 'relu1',      'pool1');

    % ResBlock 1: 残差连接 (pool1 → res1_add/in2 作为 skip)
    lgraph = connectLayers(lgraph, 'pool1',      'res1_conv1');
    lgraph = connectLayers(lgraph, 'res1_conv1', 'res1_bn1');
    lgraph = connectLayers(lgraph, 'res1_bn1',   'res1_relu1');
    lgraph = connectLayers(lgraph, 'res1_relu1', 'res1_conv2');
    lgraph = connectLayers(lgraph, 'res1_conv2', 'res1_bn2');
    lgraph = connectLayers(lgraph, 'res1_bn2',   'res1_add/in1');
    lgraph = connectLayers(lgraph, 'pool1',      'res1_add/in2');
    lgraph = connectLayers(lgraph, 'res1_add',   'res1_relu2');

    % ResBlock 2: 残差连接 + 1×1 投影 shortcut
    lgraph = connectLayers(lgraph, 'res1_relu2',  'res2_conv1');
    lgraph = connectLayers(lgraph, 'res2_conv1',  'res2_bn1');
    lgraph = connectLayers(lgraph, 'res2_bn1',    'res2_relu1');
    lgraph = connectLayers(lgraph, 'res2_relu1',  'res2_conv2');
    lgraph = connectLayers(lgraph, 'res2_conv2',  'res2_bn2');
    lgraph = connectLayers(lgraph, 'res2_bn2',    'res2_add/in1');
    lgraph = connectLayers(lgraph, 'res1_relu2',  'res2_proj');
    lgraph = connectLayers(lgraph, 'res2_proj',   'res2_proj_bn');
    lgraph = connectLayers(lgraph, 'res2_proj_bn','res2_add/in2');
    lgraph = connectLayers(lgraph, 'res2_add',    'res2_relu2');

    % ResBlock 3: 残差连接 + 1×1 投影 shortcut
    lgraph = connectLayers(lgraph, 'res2_relu2',  'res3_conv1');
    lgraph = connectLayers(lgraph, 'res3_conv1',  'res3_bn1');
    lgraph = connectLayers(lgraph, 'res3_bn1',    'res3_relu1');
    lgraph = connectLayers(lgraph, 'res3_relu1',  'res3_conv2');
    lgraph = connectLayers(lgraph, 'res3_conv2',  'res3_bn2');
    lgraph = connectLayers(lgraph, 'res3_bn2',    'res3_add/in1');
    lgraph = connectLayers(lgraph, 'res2_relu2',  'res3_proj');
    lgraph = connectLayers(lgraph, 'res3_proj',   'res3_proj_bn');
    lgraph = connectLayers(lgraph, 'res3_proj_bn','res3_add/in2');
    lgraph = connectLayers(lgraph, 'res3_add',    'res3_relu2');

    % 输出
    lgraph = connectLayers(lgraph, 'res3_relu2',  'gap');
    lgraph = connectLayers(lgraph, 'gap',         'fc_output');
end
