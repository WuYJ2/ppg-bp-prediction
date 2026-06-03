% 批量获得特征值
clc;clear;
% 获取脚本所在目录
[scriptDir, ~, ~] = fileparts(mfilename('fullpath'));
% ppg路径
FromPlace = fullfile(scriptDir, "dataset", "PPG");
% 保存路径
ToPlace = fullfile(scriptDir, "output");
% 确保输出目录存在
mkdir(ToPlace);
% 获取文件夹中所有.mat文件的列表
Files = dir(fullfile(FromPlace, '*.mat'));
% 循环中提取特征
ParameterVerge = table2cell(readtable(fullfile(scriptDir, "ParameterVerge.xlsx"),'VariableNamingRule' , 'preserve' ))';
% 使用XLSX获得选择后的特征值表
Parameter = ParameterVerge(1, :);
for i = 1: length(Files)
    % 提取数据
    FileName = Files(i).name;
    SignalName = strsplit(FileName, '.');
    Signal = load(fullfile(FromPlace, FileName));
    Data = Signal.(SignalName{1});
    % 提取特征
    DataParameter = [];
    for j = 1: size(Data, 1)
        Data_j = Data(j, :);
        [~, DataPara_j] = find_parameter_amend(Parameter,Data_j);
        DataParameter = [DataParameter; DataPara_j];
    end
    % 保存特征
    ToPlace_i = fullfile(ToPlace, SignalName{1} + "_parameter.mat");
    save(ToPlace_i, "DataParameter");
    disp(SignalName{1} + " is over.");
end