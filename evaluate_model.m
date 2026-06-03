%% evaluate_model.m
% 评估微调模型在测试集上的血压预测性能
% 输出: SBP/DBP 预测值, MAE/STD, Bland-Altman 图, 相关性散点图
%
% 可切换 modelType = 'base' / 'finetuned' / 'both' 对比基础模型和微调模型

clear; clc;

%% ==================== 配置 ====================
[scriptDir, ~, ~] = fileparts(mfilename('fullpath'));
publicDataPath = fullfile(scriptDir, '数据集', '1、公开数据集');
selfDataPath   = fullfile(scriptDir, 'dataset');
modelSavePath  = fullfile(scriptDir, 'models');
outputPath     = fullfile(scriptDir, 'cnn_output');
mkdir(outputPath);

% 选择评估模式: 'finetuned' | 'base' | 'both'
evalMode = 'both';

% 信号参数
Fs     = 125;
sigLen = 2048;

%% ==================== 加载模型和归一化参数 ====================
normLoaded = load(fullfile(modelSavePath, 'norm_params.mat'));
normParams = normLoaded.normParams;
bpMean     = normLoaded.bpMean;
bpStd      = normLoaded.bpStd;

models = struct();
if ismember(evalMode, {'base', 'both'})
    bf = fullfile(modelSavePath, 'base_model_best.mat');
    if ~exist(bf, 'file'), bf = fullfile(modelSavePath, 'base_model_final.mat'); end
    loaded = load(bf);
    models.base = loaded.dlnet;
    fprintf('基础模型已加载\n');
end
if ismember(evalMode, {'finetuned', 'both'})
    ff = fullfile(modelSavePath, 'finetuned_model_best.mat');
    if ~exist(ff, 'file'), ff = fullfile(modelSavePath, 'finetuned_model_final.mat'); end
    loaded = load(ff);
    models.finetuned = loaded.dlnet;
    fprintf('微调模型已加载\n');
end

%% ==================== 加载测试数据 ====================
fprintf('=== 加载测试数据 ===\n');

% 公开测试集
pubPPG  = load(fullfile(publicDataPath, 'TestPPG.mat'));
pubSBP  = load(fullfile(publicDataPath, 'TestSBP.mat'));
pubDBP  = load(fullfile(publicDataPath, 'TestDBP.mat'));
X_pub_raw = pubPPG.TestPPG;
Y_pub_sbp = pubSBP.TestSBP(:);
Y_pub_dbp = pubDBP.TestDBP(:);

% 自建测试集 (cell 格式)
selfPPG = load(fullfile(selfDataPath, 'PPG', 'TestPPG_cell.mat'));
selfSBP = load(fullfile(selfDataPath, 'BP', 'TestSBP.mat'));
selfDBP = load(fullfile(selfDataPath, 'BP', 'TestDBP.mat'));

TestPPG_cell = selfPPG.TestPPG_cell;
numSelf = length(TestPPG_cell);
X_self_raw = zeros(numSelf, sigLen);
for i = 1:numSelf
    sig = TestPPG_cell{i};
    if iscell(sig), sig = sig{1}; end
    if length(sig) >= sigLen
        X_self_raw(i, :) = sig(1:sigLen);
    else
        X_self_raw(i, 1:length(sig)) = sig;
    end
end
Y_self_sbp = selfSBP.TestSBP(:);
Y_self_dbp = selfDBP.TestDBP(:);

fprintf('公开测试集: %d, 自建测试集: %d\n', size(X_pub_raw,1), numSelf);

%% ==================== 预处理 ====================
X_pub_3ch  = preprocessPPG_batch(X_pub_raw, Fs);
X_self_3ch = preprocessPPG_batch(X_self_raw, Fs);
X_pub_norm  = applyNormalize3Ch(X_pub_3ch, normParams);
X_self_norm = applyNormalize3Ch(X_self_3ch, normParams);

%% ==================== 评估 ====================
allResults = struct();

for mIdx = 1:length(fieldnames(models))
    modelNames = fieldnames(models);
    modelName  = modelNames{mIdx};
    dlnet      = models.(modelName);

    fprintf('\n========== 评估: %s 模型 ==========\n', modelName);

    % --- 公开测试集 ---
    fprintf('--- 公开测试集 ---\n');
    dlX = dlarray(single(X_pub_norm), 'CTB');
    dlY_pred = mean(predict(dlnet, dlX), 2);  % (2, 1, N) → (2, N)
    Y_pred_norm = double(extractdata(dlY_pred))';  % (N, 2)
    Y_pred = Y_pred_norm .* bpStd + bpMean;
    Y_true = [Y_pub_sbp, Y_pub_dbp];

    pubResults = computeMetrics(Y_true, Y_pred, '公开测试集');
    allResults.(modelName).public = pubResults;

    % --- 自建测试集 ---
    fprintf('--- 自建测试集 ---\n');
    dlX_self = dlarray(single(X_self_norm), 'CTB');
    dlY_pred_self = squeeze(predict(dlnet, dlX_self));
    Y_pred_self_norm = double(extractdata(dlY_pred_self))';
    Y_pred_self = Y_pred_self_norm .* bpStd + bpMean;
    Y_true_self = [Y_self_sbp, Y_self_dbp];

    selfResults = computeMetrics(Y_true_self, Y_pred_self, '自建测试集');
    allResults.(modelName).self = selfResults;

    % --- 画图 ---
    plotEvaluation(Y_true,      Y_pred,      ...
                   Y_true_self, Y_pred_self, ...
                   modelName, outputPath);

    % 保存预测值
    save(fullfile(outputPath, sprintf('predictions_%s.mat', modelName)), ...
        'Y_true', 'Y_pred', 'Y_true_self', 'Y_pred_self', ...
        'pubResults', 'selfResults');
end

%% ==================== 对比汇总 ====================
if ismember(evalMode, 'both')
    fprintf('\n========== 模型对比 ==========\n');
    fprintf('%-14s %-12s %-12s %-12s %-12s\n', ...
        '数据集', '模型', 'SBP_MAE', 'SBP_STD', 'DBP_MAE', 'DBP_STD');
    fprintf('%s\n', repmat('-', 1, 72));

    datasets = {'public', 'self'};
    datasetLabels = {'公开测试集', '自建测试集'};
    for d = 1:2
        for m = 1:2
            mn = modelNames{m};
            if strcmp(datasets{d}, 'public')
                r = allResults.(mn).public;
            else
                r = allResults.(mn).self;
            end
            fprintf('%-14s %-12s %-12.2f %-12.2f %-12.2f %-12.2f\n', ...
                datasetLabels{d}, mn, ...
                r.SBP_MAE, r.SBP_STD, r.DBP_MAE, r.DBP_STD);
        end
    end

    % 对比柱状图
    figure('Position', [100, 100, 700, 400]);
    metrics = {'SBP_MAE', 'SBP_STD', 'DBP_MAE', 'DBP_STD'};
    metricLabels = {'SBP MAE', 'SBP STD', 'DBP MAE', 'DBP STD'};
    barData = zeros(2, 2, 4);  % model × dataset × metric

    for d = 1:2
        for m = 1:2
            mn = modelNames{m};
            ds = datasets{d};
            r = allResults.(mn).(ds);
            for k = 1:4
                barData(m, d, k) = r.(metrics{k});
            end
        end
    end

    colors = lines(2);
    for k = 1:4
        subplot(2, 2, k);
        b = bar(squeeze(barData(:, :, k))');
        for c = 1:2, b(c).FaceColor = colors(c, :); end
        set(gca, 'XTickLabel', datasetLabels);
        ylabel('mmHg');
        title(metricLabels{k});
        legend({'Base', 'Fine-tuned'}, 'Location', 'best');
        grid on;
    end
    sgtitle('Base vs Fine-tuned Model Comparison');
    saveas(gcf, fullfile(outputPath, 'model_comparison.png'));
    close(gcf);
end

fprintf('\n=== 评估完成，结果保存至 %s ===\n', outputPath);

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

function results = computeMetrics(Y_true, Y_pred, setName)
    errSBP = Y_pred(:, 1) - Y_true(:, 1);
    errDBP = Y_pred(:, 2) - Y_true(:, 2);

    results.SBP_MAE = mean(abs(errSBP));
    results.SBP_STD = std(errSBP);
    results.DBP_MAE = mean(abs(errDBP));
    results.DBP_STD = std(errDBP);

    fprintf('  SBP: MAE=%.2f mmHg, STD=%.2f mmHg\n', results.SBP_MAE, results.SBP_STD);
    fprintf('  DBP: MAE=%.2f mmHg, STD=%.2f mmHg\n', results.DBP_MAE, results.DBP_STD);
end

function plotEvaluation(Y_pub, Y_pub_pred, Y_self, Y_self_pred, modelName, outputPath)
    datasets = {Y_pub, Y_pub_pred, '公开测试集'; ...
                Y_self, Y_self_pred, '自建测试集'};

    for d = 1:2
        Yt = datasets{d, 1};
        Yp = datasets{d, 2};
        label = datasets{d, 3};
        shortLabel = erase(label, '测试集');

        % === Bland-Altman (SBP + DBP 并排) ===
        figure('Visible', 'off', 'Position', [100, 100, 900, 380]);
        for bp = 1:2
            subplot(1, 2, bp);
            bpName = {'SBP', 'DBP'};
            diff = Yp(:, bp) - Yt(:, bp);
            meanVal = (Yt(:, bp) + Yp(:, bp)) / 2;
            meanDiff = mean(diff);
            stdDiff  = std(diff);

            scatter(meanVal, diff, 15, 'b', 'filled', 'MarkerFaceAlpha', 0.4);
            hold on;
            yline(meanDiff, 'r-', 'LineWidth', 1.5);
            yline(meanDiff + 1.96*stdDiff, 'r--', 'LineWidth', 1);
            yline(meanDiff - 1.96*stdDiff, 'r--', 'LineWidth', 1);
            xlabel(['Mean (mmHg)'], 'FontSize', 10);
            ylabel('Difference (mmHg)', 'FontSize', 10);
            title(sprintf('%s %s Bland-Altman', bpName{bp}, shortLabel), 'FontSize', 11);
            grid on;
        end
        saveas(gcf, fullfile(outputPath, ...
            sprintf('BA_%s_%s.png', modelName, matlab.lang.makeValidName(shortLabel))));
        close(gcf);

        % === 相关性散点图 (SBP + DBP 并排) ===
        figure('Visible', 'off', 'Position', [100, 100, 900, 400]);
        for bp = 1:2
            subplot(1, 2, bp);
            bpName = {'SBP', 'DBP'};
            yt = Yt(:, bp); yp = Yp(:, bp);

            scatter(yt, yp, 15, 'b', 'filled', 'MarkerFaceAlpha', 0.4);
            hold on;

            % 对角线
            mn = min([yt; yp]); mx = max([yt; yp]);
            range = mx - mn;
            plot([mn-0.05*range, mx+0.05*range], [mn-0.05*range, mx+0.05*range], ...
                'k--', 'LineWidth', 1.2);

            % 拟合线
            coeffs = polyfit(yt, yp, 1);
            xFit = linspace(mn, mx, 100);
            yFit = polyval(coeffs, xFit);
            plot(xFit, yFit, 'r-', 'LineWidth', 1.5);

            % 相关系数
            R = corrcoef(yt, yp);
            r = R(1, 2);

            xlabel(['True ', bpName{bp}, ' (mmHg)'], 'FontSize', 10);
            ylabel(['Predicted ', bpName{bp}, ' (mmHg)'], 'FontSize', 10);
            title(sprintf('%s %s Correlation (r=%.3f)', bpName{bp}, shortLabel, r), 'FontSize', 11);
            axis equal; grid on;
        end
        saveas(gcf, fullfile(outputPath, ...
            sprintf('Corr_%s_%s.png', modelName, matlab.lang.makeValidName(shortLabel))));
        close(gcf);

        % === 预测 vs 真值折线图 (前 80 个样本) ===
        figure('Visible', 'off', 'Position', [100, 100, 900, 380]);
        nShow = min(80, size(Yt, 1));
        for bp = 1:2
            subplot(1, 2, bp);
            bpName = {'SBP', 'DBP'};
            plot(1:nShow, Yt(1:nShow, bp), 'b-', 'LineWidth', 1, 'DisplayName', 'True');
            hold on;
            plot(1:nShow, Yp(1:nShow, bp), 'r-', 'LineWidth', 1, 'DisplayName', 'Predicted');
            xlabel('Sample Index'); ylabel([bpName{bp}, ' (mmHg)']);
            title(sprintf('%s %s Predictions', bpName{bp}, shortLabel));
            legend('Location', 'best'); grid on;
        end
        saveas(gcf, fullfile(outputPath, ...
            sprintf('Line_%s_%s.png', modelName, matlab.lang.makeValidName(shortLabel))));
        close(gcf);
    end
end
