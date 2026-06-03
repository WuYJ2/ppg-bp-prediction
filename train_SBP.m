clear; clc;

TrainSBPFirst = load("dataset\BP\TrainSBP.mat").TrainSBP;
if ~isa(TrainSBPFirst, 'double')
    TrainSBPFirst = double(TrainSBPFirst);
end

TrainDBPFirst = load("dataset\BP\TrainDBP.mat").TrainDBP;
if ~isa(TrainDBPFirst, 'double')
    TrainDBPFirst = double(TrainDBPFirst);
end

TrainPPGFirst = load("dataset\Parameter\TrainParameter.mat").TrainParameter;
if ~isa(TrainPPGFirst, 'double')
    TrainPPGFirst = double(TrainPPGFirst);
end

TestSBPFirst = load("dataset\BP\TestSBP.mat").TestSBP;
if ~isa(TestSBPFirst, 'double')
    TestSBPFirst = double(TestSBPFirst);
end

TestDBPFirst = load("dataset\BP\TestDBP.mat").TestDBP;
if ~isa(TestDBPFirst, 'double')
    TestDBPFirst = double(TestDBPFirst);
end

TestPPGFirst = load("dataset\Parameter\TestParameter.mat").TestParameter;  % 请确认文件名正确
if ~isa(TestPPGFirst, 'double')
    TestPPGFirst = double(TestPPGFirst);
end

fixed_SBP_min = 80;
fixed_SBP_max = 150;

% 选择训练SBP模型的特征
dataSortList = [26,7,32,43,44,23,42];
numberofSort = 7;
choiceSortList = nchoosek(dataSortList, numberofSort);
totalCombinations = size(choiceSortList, 1);

fprintf("特征组合总数: %d (从%d个特征中选择%d个)\n", totalCombinations, length(dataSortList), numberofSort);

% 创建结果保存目录
mkdir(fullfile("result\SBP\", mat2str(numberofSort)));

% 保存选择方式
save(fullfile("result\SBP\", mat2str(numberofSort), "choicesORTList"), "choiceSortList");

for i = 1: size(choiceSortList, 1)
    progress = (i / totalCombinations) * 100;
    fprintf("\n===== 正在处理第%d/%d种特征组合 (进度: %.1f%%) =====\n", i, totalCombinations, progress);

    % 创建当前组合的保存目录
    mkdir(fullfile("result\SBP\", mat2str(numberofSort), mat2str(i)));

    % 提取当前组合的特征数据
    TestPPGUse = TestPPGFirst(:, choiceSortList(i, :));
    TrainPPGUse = TrainPPGFirst(:, choiceSortList(i, :));
    
    TrainSort = zeros(size(TrainPPGUse));
    TestSort = zeros(size(TestPPGUse));
    PPGmapminmax = [];
    
    for j = 1: numberofSort
        % 归一化处理
        current_col_train = TrainPPGUse(:, j);
        current_col_test = TestPPGUse(:, j);
        
        col_train_min = min(current_col_train);
        col_train_max = max(current_col_train);
        col_min_ext = col_train_min * 0.85;
        col_max_ext = col_train_max * 1.15;
        
        TrainPPGList = [col_min_ext; col_max_ext]';
        [~, TrainPPGs] = mapminmax(TrainPPGList, 0, 1);
        PPGmapminmax = [PPGmapminmax, TrainPPGs];
        
        TrainSort(:, j) = mapminmax('apply', current_col_train', TrainPPGs)';
        TestSort(:, j) = mapminmax('apply', current_col_test', TrainPPGs)';
    end
    
    % SBP归一化
    SBP_ref_list = [fixed_SBP_min; fixed_SBP_max];
    [~, SBPmapminmax] = mapminmax(SBP_ref_list', 0, 1);
    TrainSBP = mapminmax('apply', TrainSBPFirst', SBPmapminmax)';
    TestSBP = mapminmax('apply', TestSBPFirst', SBPmapminmax)';
    
    % 保存归一化参数
    save(fullfile("result\SBP\", mat2str(numberofSort), mat2str(i), "归一化参数"), ...
        'PPGmapminmax', 'SBPmapminmax', 'fixed_SBP_min', 'fixed_SBP_max');
    
    %% SVM参数优化与模型训练
    fprintf("正在优化SVM参数...\n");
    % [mse1, bestc1, bestg1] = SVMcgForRegress(TrainSBP, TrainSort,...
    %     0, 5, 5, 8, 5, 1, 1, 0.05);
    % 
    % [mse3, bestc, bestg] = SVMcgForRegress(TrainSBP, TrainSort,...
    %     log2(bestc1)-0.5, log2(bestc1)+0.5, log2(bestg1)-0.5, log2(bestg1)+0.5,...
    %     5, 0.1, 0.1, 0.05);
    bestc=0.757858283255199;bestg=90.5096679918781;
    fprintf("最优SVM参数 - C: %.4f, gamma: %.4f\n", bestc, bestg);
    
    cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg) , ' -s 3 -p 0.05 -b 1'];
    tic;
    model = svmtrain(TrainSBP, TrainSort, cmd);
    toc;
    
    % 保存模型与最优参数
    varargout = {model, bestc, bestg};
    save(fullfile("result\SBP\", mat2str(numberofSort), mat2str(i), "SBP"), 'varargout');
    
    % 测试集预测与结果计算
    fprintf("测试集预测中...\n");
    [TestPredSBP, AccTestPredSBP, ~] = svmpredict(TestSBP, TestSort, model, '-b 1');
    TestPredSBP = mapminmax('reverse', TestPredSBP', SBPmapminmax)';
    TestTrueSBP = mapminmax('reverse', TestSBP', SBPmapminmax)';
    SBP_MAE_test = mae(TestTrueSBP, TestPredSBP);
    SBP_STD_test = std(TestPredSBP - TestTrueSBP);
    fprintf("测试集结果 - MAE: %.2f, STD: %.2f\n", SBP_MAE_test, SBP_STD_test);
    
    % 保存测试集结果
    save(fullfile("result\SBP\", mat2str(numberofSort), mat2str(i), "TestPredSBP"), ...
        "SBP_MAE_test", "SBP_STD_test", "TestSBP", "TestPredSBP", "AccTestPredSBP");

                %% 绘制真实值与预测值对比图（折线图）- 改回长方形
    figure('Visible', 'off', 'Position', [100, 100, 500, 300]); % 改回长方形 500x300
    plot(1:length(TestTrueSBP), TestTrueSBP, 'b-', 'LineWidth', 1, 'DisplayName', 'True SBP');
    hold on;
    plot(1:length(TestPredSBP), TestPredSBP, 'r-', 'LineWidth', 1, 'DisplayName', 'Predicted SBP');
    xlabel('Sample Index');
    ylabel('SBP (mmHg)');
    title(sprintf('True vs Predicted SBP (Combination %d)', i));
    legend('Location', 'best');
    grid on;
    % 300 DPI 高清保存保持不变
    comp_filename = fullfile("result\SBP\", mat2str(numberofSort), mat2str(i), "Comparison.png");
    print(gcf, comp_filename, '-dpng', '-r300');
    close(gcf);

        %% 绘制 Bland-Altman 图
    figure('Visible', 'off', 'Position', [100, 100, 500, 500]); % 改为正方形
    diff = TestPredSBP - TestTrueSBP;
    meanVal = (TestTrueSBP + TestPredSBP) / 2;
    meanDiff = mean(diff);
    stdDiff = std(diff);
    upperLoA = meanDiff + 1.96 * stdDiff;
    lowerLoA = meanDiff - 1.96 * stdDiff;

    scatter(meanVal, diff, 20, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    yline(meanDiff, 'r-', 'LineWidth', 1.5);
    yline(upperLoA, 'r--', 'LineWidth', 1.2);
    yline(lowerLoA, 'r--', 'LineWidth', 1.2);
    xlabel('Mean of True and Predicted SBP (mmHg)', 'FontSize', 11);
    ylabel('Difference (Predicted - True) (mmHg)', 'FontSize', 11);
    title(sprintf('Bland-Altman Plot (Combination %d)', i), 'FontSize', 12);
    grid on;
    legend('Data', 'Mean Diff', 'LoA (Mean±1.96SD)', 'Location', 'best');
    axis square; % 强制坐标轴也为正方形

    xlim([min(meanVal)-2, max(meanVal)+2]);
    maxAbsDiff = max(abs(diff - meanDiff));
    ylim([meanDiff - maxAbsDiff*1.1, meanDiff + maxAbsDiff*1.1]);

    % 300 DPI 高清保存
    ba_filename = fullfile("result\SBP\", mat2str(numberofSort), mat2str(i), "BlandAltman.png");
    print(gcf, ba_filename, '-dpng', '-r300');
    close(gcf);

        %% 绘制相关性分析图
    figure('Visible', 'off', 'Position', [100, 100, 500, 500]);
    minVal = min([TestTrueSBP; TestPredSBP]);
    maxVal = max([TestTrueSBP; TestPredSBP]);
    rangeMin = floor(minVal/10)*10;
    rangeMax = ceil(maxVal/10)*10;
    if rangeMax - rangeMin < 50
        rangeMax = rangeMin + 50;
    end

    scatter(TestTrueSBP, TestPredSBP, 20, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    plot([rangeMin, rangeMax], [rangeMin, rangeMax], 'k--', 'LineWidth', 1.2);
    coeffs = polyfit(TestTrueSBP, TestPredSBP, 1);
    xFit = linspace(rangeMin, rangeMax, 100);
    yFit = polyval(coeffs, xFit);
    plot(xFit, yFit, 'r-', 'LineWidth', 1.5);

    xlabel('True SBP (mmHg)', 'FontSize', 11);
    ylabel('Predicted SBP (mmHg)', 'FontSize', 11);
    title(sprintf('Correlation Plot (Combination %d)', i), 'FontSize', 12);
    grid on;
    axis equal;
    axis square;
    xlim([rangeMin, rangeMax]);
    ylim([rangeMin, rangeMax]);
    set(gca, 'XTick', rangeMin:10:rangeMax, 'YTick', rangeMin:10:rangeMax);

    R = corrcoef(TestTrueSBP, TestPredSBP);
    r = R(1,2);
    text(0.05, 0.9, sprintf('r = %.3f', r), 'Units', 'normalized', ...
        'FontSize', 12, 'BackgroundColor', 'white', 'EdgeColor', 'k');

    % 300 DPI 高清保存
    corr_filename = fullfile("result\SBP\", mat2str(numberofSort), mat2str(i), "Correlation.png");
    print(gcf, corr_filename, '-dpng', '-r300');
    close(gcf);
    
    fprintf("第%d种特征组合处理完成\n", i);
end