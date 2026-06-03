%% 结果分析
% 文件总目录（使用脚本所在目录下的result子目录）
[scriptDir, ~, ~] = fileparts(mfilename('fullpath'));
dataPathRoot = fullfile(scriptDir, "result", "DBP", "6");
% 文件总数
number = 2;
% 文件名目录
nameList = [58, 60];
% 平均绝对误差与标准差列表
List = zeros(2, number);
% 循环中获取误差文件
for i = 1: number
    dataPath = fullfile(dataPathRoot, mat2str(nameList(i)), "TestPredDBP");
    load(dataPath);
    List(1, i) = DBP_MAE;
    List(2, i) = DBP_STD;
end
List = List';