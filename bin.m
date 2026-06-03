% 设置.bin文件路径（使用脚本所在目录）
[scriptDir, ~, ~] = fileparts(mfilename('fullpath'));
binFilePath = fullfile(scriptDir, 'data_10.bin');
% 假设数据是单精度浮点数（float），如果数据类型不同，需要修改 'float' 为对应类型，如 'double' 'int16' 等
data = fread(fopen(binFilePath,'rb'), '*float');
% 设置.mat文件路径
matFilePath = fullfile(scriptDir, 'data_10.mat');
save(matFilePath, 'data');