%% evaluate_gpr.m
% 评估 GPR 基础模型与微调模型在公开/自建测试集上的性能 — 7 特征版
%
% 依赖: train_gpr.m 输出的 gpr_base.mat
%       finetune_gpr.m 输出的 gpr_finetuned.mat (可选)

clear; clc;

%% ==================== 配置 ====================
[scriptDir, ~, ~] = fileparts(mfilename('fullpath'));
publicDataPath = fullfile(scriptDir, '数据集', '1、公开数据集');
selfDataPath   = fullfile(scriptDir, '数据集', '2、自建数据集');
modelSavePath  = fullfile(scriptDir, 'gpr_models');
outputPath     = fullfile(scriptDir, 'gpr_output');
mkdir(outputPath);

% 选择 7 个特征 (1-indexed, 与 train_gpr.m / finetune_gpr.m 一致)
FEATURE_IDS = [7, 23, 26, 32, 42, 43, 44];

EVAL_MODE = 'both';  % 'base' | 'finetuned' | 'both'

%% ==================== 加载模型 ====================
models = struct();

baseFile = fullfile(modelSavePath, 'gpr_base.mat');
if ismember(EVAL_MODE, {'base', 'both'}) && exist(baseFile, 'file')
    m = load(baseFile);
    models.base.gprSBP = m.gprSBP;
    models.base.gprDBP = m.gprDBP;
    models.base.X_mean = m.X_mean;
    models.base.X_std  = m.X_std;
    models.base.KERNEL = m.KERNEL;
    fprintf('基础模型已加载 (%s)\n', m.KERNEL);
end

ftFile = fullfile(modelSavePath, 'gpr_finetuned.mat');
if ismember(EVAL_MODE, {'finetuned', 'both'}) && exist(ftFile, 'file')
    m = load(ftFile);
    models.finetuned.gprSBP = m.gprSBP;
    models.finetuned.gprDBP = m.gprDBP;
    models.finetuned.X_mean = m.X_mean;
    models.finetuned.X_std  = m.X_std;
    models.finetuned.KERNEL = m.KERNEL;
    fprintf('微调模型已加载 (%s)\n', m.KERNEL);
end

if isempty(fieldnames(models))
    error('未找到任何模型文件, 请先运行 train_gpr.m');
end

%% ==================== 加载测试数据 (78→7 特征) ====================
fprintf('\n=== 加载测试数据 (78维→7维) ===\n');

% --- 公开测试集 ---
pubParam = load(fullfile(publicDataPath, 'TestParameter.mat'));
pubSBP   = load(fullfile(publicDataPath, 'TestSBP.mat'));
pubDBP   = load(fullfile(publicDataPath, 'TestDBP.mat'));

X_pub_full = pubParam.TestParameter;
if size(X_pub_full, 2) ~= 78 && size(X_pub_full, 1) == 78, X_pub_full = X_pub_full'; end
X_pub = X_pub_full(:, FEATURE_IDS);  % 选取 7 特征
Y_pub_sbp = pubSBP.TestSBP(:);
Y_pub_dbp = pubDBP.TestDBP(:);
valid = all(isfinite(X_pub), 2);
X_pub = X_pub(valid, :); Y_pub_sbp = Y_pub_sbp(valid); Y_pub_dbp = Y_pub_dbp(valid);

% --- 自建测试集 ---
selfParam = load(fullfile(selfDataPath, 'TestParameter.mat'));
selfSBP   = load(fullfile(selfDataPath, 'TestSBP.mat'));
selfDBP   = load(fullfile(selfDataPath, 'TestDBP.mat'));

X_self_full = selfParam.TestParameter;
if size(X_self_full, 2) ~= 78 && size(X_self_full, 1) == 78, X_self_full = X_self_full'; end
X_self = X_self_full(:, FEATURE_IDS);  % 选取 7 特征
Y_self_sbp = selfSBP.TestSBP(:);
Y_self_dbp = selfDBP.TestDBP(:);
valid = all(isfinite(X_self), 2);
X_self = X_self(valid, :); Y_self_sbp = Y_self_sbp(valid); Y_self_dbp = Y_self_dbp(valid);

fprintf('公开测试集: %d 样本, %d 特征\n', size(X_pub, 1), size(X_pub, 2));
fprintf('自建测试集: %d 样本, %d 特征\n', size(X_self, 1), size(X_self, 2));

%% ==================== 评估 ====================
datasets = struct('pub', struct('X', X_pub, 'sbp', Y_pub_sbp, 'dbp', Y_pub_dbp, 'label', '公开测试集'), ...
                  'self', struct('X', X_self, 'sbp', Y_self_sbp, 'dbp', Y_self_dbp, 'label', '自建测试集'));
dsKeys = fieldnames(datasets);
modelNames = fieldnames(models);

allResults = struct();

for di = 1:length(dsKeys)
    dk = dsKeys{di};
    ds = datasets.(dk);
    fprintf('\n========== %s ==========\n', ds.label);

    for mi = 1:length(modelNames)
        mn = modelNames{mi};
        m  = models.(mn);

        X_norm = (ds.X - m.X_mean) ./ m.X_std;
        [predSBP, sdSBP, ciSBP] = predict(m.gprSBP, X_norm);
        [predDBP, sdDBP, ciDBP] = predict(m.gprDBP, X_norm);

        maeSBP = mean(abs(predSBP - ds.sbp));
        stdSBP = std(predSBP - ds.sbp);
        maeDBP = mean(abs(predDBP - ds.dbp));
        stdDBP = std(predDBP - ds.dbp);

        fprintf('[%s] SBP: MAE=%.2f, STD=%.2f | DBP: MAE=%.2f, STD=%.2f mmHg\n', ...
            mn, maeSBP, stdSBP, maeDBP, stdDBP);

        allResults.(dk).(mn) = struct('maeSBP', maeSBP, 'stdSBP', stdSBP, ...
            'maeDBP', maeDBP, 'stdDBP', stdDBP);

        % 画图
        plotGPR(ds.sbp, ds.dbp, predSBP, predDBP, ciSBP, ciDBP, ...
            [mn '_' dk], m.KERNEL, outputPath);

        % 保存
        save(fullfile(outputPath, sprintf('predictions_%s_%s.mat', mn, dk)), ...
            'predSBP', 'predDBP', 'sdSBP', 'sdDBP', 'ciSBP', 'ciDBP', ...
            'maeSBP', 'stdSBP', 'maeDBP', 'stdDBP');
    end
end

%% ==================== 对比汇总 ====================
if length(modelNames) == 2
    fprintf('\n==================== 模型对比 ====================\n');
    for di = 1:length(dsKeys)
        dk = dsKeys{di};
        ds = datasets.(dk);
        fprintf('\n--- %s ---\n', ds.label);
        fprintf('%-10s %-10s %-10s %-10s %-10s\n', '模型', 'SBP_MAE', 'SBP_STD', 'DBP_MAE', 'DBP_STD');
        fprintf('%s\n', repmat('-', 1, 50));
        for mi = 1:length(modelNames)
            mn = modelNames{mi};
            r = allResults.(dk).(mn);
            fprintf('%-10s %-10.2f %-10.2f %-10.2f %-10.2f\n', ...
                mn, r.maeSBP, r.stdSBP, r.maeDBP, r.stdDBP);
        end
    end

    % 对比柱状图 (两个数据集)
    figure('Visible', 'off', 'Position', [100, 100, 900, 380]);
    for di = 1:length(dsKeys)
        dk = dsKeys{di};
        subplot(1, 2, di);
        metrics = [allResults.(dk).base.maeSBP, allResults.(dk).base.stdSBP, ...
                   allResults.(dk).base.maeDBP, allResults.(dk).base.stdDBP;
                   allResults.(dk).finetuned.maeSBP, allResults.(dk).finetuned.stdSBP, ...
                   allResults.(dk).finetuned.maeDBP, allResults.(dk).finetuned.stdDBP];
        bar(metrics);
        set(gca, 'XTickLabel', modelNames');
        ylabel('mmHg'); grid on;
        legend({'SBP MAE', 'SBP STD', 'DBP MAE', 'DBP STD'}, 'Location', 'best');
        title(datasets.(dk).label);
    end
    sgtitle('GPR Base vs Fine-tuned');
    saveas(gcf, fullfile(outputPath, 'model_comparison.png'));
    close(gcf);
end

fprintf('\n=== 评估完成, 结果保存至 %s ===\n', outputPath);

%% ==================== 画图函数 ====================
function plotGPR(Y_sbp, Y_dbp, predSBP, predDBP, ciSBP, ciDBP, name, kernel, outputPath)
    nShow = min(80, length(Y_sbp));
    bpData = {Y_sbp, predSBP, 'SBP'; Y_dbp, predDBP, 'DBP'};

    % --- Bland-Altman ---
    figure('Visible', 'off', 'Position', [100, 100, 900, 380]);
    for bp = 1:2
        subplot(1, 2, bp);
        yt = bpData{bp, 1}; yp = bpData{bp, 2};
        diff = yp - yt; meanVal = (yt + yp) / 2;
        md = mean(diff); sd = std(diff);
        scatter(meanVal, diff, 15, 'b', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
        yline(md, 'r-', 'LineWidth', 1.5);
        yline(md + 1.96*sd, 'r--', 'LineWidth', 1);
        yline(md - 1.96*sd, 'r--', 'LineWidth', 1);
        xlabel('Mean (mmHg)'); ylabel('Difference (mmHg)');
        title(sprintf('%s BA (%s)', bpData{bp,3}, name)); grid on;
    end
    saveas(gcf, fullfile(outputPath, sprintf('BA_%s.png', name)));
    close(gcf);

    % --- Correlation ---
    figure('Visible', 'off', 'Position', [100, 100, 900, 420]);
    for bp = 1:2
        subplot(1, 2, bp);
        yt = bpData{bp, 1}; yp = bpData{bp, 2};
        scatter(yt, yp, 15, 'b', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
        mn = min([yt; yp]); mx = max([yt; yp]); rng = mx - mn;
        plot([mn-0.05*rng, mx+0.05*rng], [mn-0.05*rng, mx+0.05*rng], 'k--', 'LineWidth', 1.2);
        coeffs = polyfit(yt, yp, 1);
        xFit = linspace(mn, mx, 100);
        plot(xFit, polyval(coeffs, xFit), 'r-', 'LineWidth', 1.5);
        r = corrcoef(yt, yp); r = r(1,2);
        xlabel(sprintf('True %s (mmHg)', bpData{bp,3}));
        ylabel(sprintf('Predicted %s (mmHg)', bpData{bp,3}));
        title(sprintf('%s Corr r=%.3f (%s)', bpData{bp,3}, r, name)); axis equal; grid on;
    end
    saveas(gcf, fullfile(outputPath, sprintf('Corr_%s.png', name)));
    close(gcf);

    % --- Line ---
    figure('Visible', 'off', 'Position', [100, 100, 900, 380]);
    ciData = {ciSBP, ciDBP};
    for bp = 1:2
        subplot(1, 2, bp);
        yt = bpData{bp, 1}; yp = bpData{bp, 2};
        plot(1:nShow, yt(1:nShow), 'b-', 'LineWidth', 1); hold on;
        plot(1:nShow, yp(1:nShow), 'r-', 'LineWidth', 1);
        ci = ciData{bp};
        fill([1:nShow, nShow:-1:1], [ci(1:nShow,1)', fliplr(ci(1:nShow,2)')], ...
            'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
        xlabel('Sample Index'); ylabel(sprintf('%s (mmHg)', bpData{bp,3}));
        title(sprintf('%s (%s)', bpData{bp,3}, name));
        legend('True', 'Predicted', '95% CI', 'Location', 'best'); grid on;
    end
    saveas(gcf, fullfile(outputPath, sprintf('Line_%s.png', name)));
    close(gcf);
end
