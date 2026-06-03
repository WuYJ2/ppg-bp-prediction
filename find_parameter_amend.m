%% 主输出函数
function varargout = find_parameter_amend(Parameter, data)
% 输入：Paramter:特征表，包含特征值名称的cell; data:ppg信号(2048)
% 输出：varargout:特征值按顺序排列构成的数组，使用varargout输出任意数量的特征值
% find_parameter将通过给定的PPG信号数组，未必按顺序地计算特征表中的特征值
TimeList = find_time(data);
AreaList = find_area(data);
Diff1List = find_diff1(data);
FreqList = find_freq(data);
WTList = find_WT(data);
% 标号
List = zeros(1, 78);
TimeListLabel = [1: 8, 12, 13, 23, 24, 36: 42, 45, 46];
AreaListLabel = [9, 10, 11, 43, 44];
Diff1ListLabel = 14: 22;
FreqListLabel = [25: 35, 64: 78];
WTListLabel = 47: 63;
% 填充
List = get_list(TimeListLabel, TimeList, List);
List = get_list(AreaListLabel, AreaList, List);
List = get_list(Diff1ListLabel, Diff1List, List);
List = get_list(FreqListLabel, FreqList, List);
List = get_list(WTListLabel, WTList, List);

varargout{1} = Parameter;
varargout{2} = List;
end
%% 时间特征
function TimeList = find_time(data)
% 输入：data:ppg信号(2048)
% 输出：TimeList:时间特征列表，包括特征1-8,12,13,23,24,36-42,45,46
% 采样频率Fs
Fs = 125;
TimeList = [];
test = cell2mat(data);

% 寻找所有峰值（cell不能直接寻峰）
[HighLabel, HighLabelPlace] = findpeaks(test);
% 大致寻找所有波谷，用于选择正式的寻峰方法
PreLowLabel = -findpeaks(-test);
% 峰值差异太小时，使用老方法，否则使用新方法
% 简单认为峰值范围在总差值的10%以内为太小
if max(HighLabel) - min(HighLabel) <= (max(HighLabel) - min(PreLowLabel)) / 5
    % 波峰,根据波形适当调整幅值和距离限制
    [HighLabelHigh, HighLabelHighPlace] = findpeaks(test,'MINPEAKHEIGHT',0.17,'MINPEAKDISTANCE',ceil(0.35*Fs));
    % 去除首尾峰值
    HighLabelHigh = HighLabelHigh(2: end - 1);
    HighLabelHighPlace = HighLabelHighPlace(2: end - 1);
    % 获得波谷
    LowLabelLow = []; LowLabelLowPlace = [];
    for i = 1: length(HighLabelHighPlace) - 1
        [LowLabelLow_i, LowLabelLowPlaceDelta] = min(test(HighLabelHighPlace(i): HighLabelHighPlace(i + 1)));
        LowLabelLowPlace_i = HighLabelHighPlace(i) + LowLabelLowPlaceDelta - 1;
        LowLabelLow = [LowLabelLow, LowLabelLow_i];
        LowLabelLowPlace = [LowLabelLowPlace, LowLabelLowPlace_i];
    end
else
    % 通过峰值均值，对波峰和重搏波进行分割（波峰比均值大）
    HighLabelMean = mean(HighLabel);
    % 波峰HighLabelHigh;对应地址HighLabelHighPlace
    HighLabelHigh = HighLabel(HighLabel > HighLabelMean);
    HighLabelHighPlace = HighLabelPlace(find(HighLabel > HighLabelMean));
    % 去除首尾波峰
    HighLabelHigh = HighLabelHigh(2: end - 1);
    HighLabelHighPlace = HighLabelHighPlace(2: end - 1);
    % 去除原始首尾波峰后，后续对数据的寻找均在新的首尾波峰间进行，相当于图像形状大致确定，便于后续处理
    % HighLabelHighPlace:波峰横坐标；HighLabelHigh:波峰纵坐标
    % 在波峰之间寻找重搏波（重搏波比均值小）
    HighLabelLow = []; HighLabelLowPlace = [];
    for i = 1: length(HighLabel)
        if HighLabelPlace(i) > HighLabelHighPlace(1) && HighLabelPlace(i) < HighLabelHighPlace(end)
            if HighLabel(i) <= HighLabelMean
                HighLabelLow = [HighLabelLow, HighLabel(i)];
                HighLabelLowPlace = [HighLabelLowPlace, HighLabelPlace(i)];
            end
        end
    end
    % 波谷在波峰之间，且靠近靠后的波峰，循环中通过取反找到波峰间的所有波谷，通过索引找到真正的波谷
    LowLabelLow = []; LowLabelLowPlace = [];
    for i = 1: (length(HighLabelHighPlace) - 1)
        % 波谷在List中
        List = cell2mat(data(HighLabelHighPlace(i): HighLabelHighPlace(i + 1)));
        ListPlace = HighLabelHighPlace(i): HighLabelHighPlace(i + 1);
        % 取反寻峰找所有波谷
        [LowLabel, LowLabelPlace] = findpeaks(-List);
        LowLabel = -LowLabel;
        LowLabelLow = [LowLabelLow, LowLabel(end)];
        LowLabelLowPlace = [LowLabelLowPlace, ListPlace(LowLabelPlace(end))];
    end
end

% 特征2:计算DT（单个脉搏波周期内，脉搏波波峰到波谷的时间间隔）
DTList = [];
for i = 1: length(HighLabelHighPlace) - 1
    DT_i = abs(HighLabelHighPlace(i) - LowLabelLowPlace(i)) / Fs;
    DTList = [DTList, DT_i];
end
DT = mean(DTList);

% 特征1:计算ST（单个脉搏波周期内，脉搏波波谷到波峰的时间间隔）
STList = [];
for i = 1: length(LowLabelLowPlace)
    ST_i = abs(HighLabelHighPlace(i + 1) - LowLabelLowPlace(i)) / Fs;
    STList = [STList, ST_i];
end
ST = mean(STList);

% 特征3:计算T（每次心搏所用时间，T=ST+DT，为相邻波谷间的时间间隔）
TList = DTList + STList;
T = mean(TList);

% 特征4:计算p_p（两个波峰之间的时间差）
p_pList = [];
for i = 1: (length(HighLabelHighPlace) - 1)
    p_p_i = abs(HighLabelHighPlace(i + 1) - HighLabelHighPlace(i)) / Fs;
    p_pList = [p_pList, p_p_i];
end
p_p = mean(p_pList);

% 特征5:计算RSD（RSD =ST/DT）
RSDLength = length(TList);
RSDList = STList ./ DTList;
RSD = mean(RSDList);

% 特征6:计算p_t（波峰波谷幅值之差，上升支高度）
p_tList = [];
for i = 1: length(LowLabelLowPlace)
    p_t_i = HighLabelHigh(i + 1) - LowLabelLow(i);
    p_tList = [p_tList, p_t_i];
end
p_t = mean(p_tList);

% 特征7:计算width1（单个脉搏波周期内幅值分别为2/3幅值对应点的时间间隔）
% width1List1:降支；width1List2:升支
width1List = [];
for i = 1: length(LowLabelLowPlace)
    % 降支获取
    Label_i1 = HighLabelHighPlace(i): LowLabelLowPlace(i);
    [~, width1_i1] = min(abs(cell2mat(data(Label_i1)) - LowLabelLow(i) - ...
        (HighLabelHigh(i) - LowLabelLow(i)) * 2 / 3));
    width1_i1 = Label_i1(width1_i1);
    % 升支获取
    Label_i2 = LowLabelLowPlace(i): HighLabelHighPlace(i + 1);
    [~, width1_i2] = min(abs(cell2mat(data(Label_i2)) - LowLabelLow(i) - ...
        (HighLabelHigh(i + 1) - LowLabelLow(i)) * 2 / 3));
    width1_i2 = Label_i2(width1_i2);
    % 相减获得单个脉宽
    width1_i = (width1_i2 - width1_i1) / Fs;
    width1List = [width1List, width1_i];
end
width1 = mean(width1List);

% 特征8:计算width2（单个脉搏波周期内幅值分别为1/2幅值对应点的时间间隔）
% width2List1:降支；width2List2:升支
width2List = [];
for i = 1: length(LowLabelLowPlace)
    % 降支获取
    Label_i1 = HighLabelHighPlace(i): LowLabelLowPlace(i);
    [~, width2_i1] = min(abs(cell2mat(data(Label_i1)) - LowLabelLow(i) - ...
        (HighLabelHigh(i) - LowLabelLow(i)) * 1 / 2));
    width2_i1 = Label_i1(width2_i1);
    % 升支获取
    Label_i2 = LowLabelLowPlace(i): HighLabelHighPlace(i + 1);
    [~, width2_i2] = min(abs(cell2mat(data(Label_i2)) - LowLabelLow(i) - ...
        (HighLabelHigh(i + 1) - LowLabelLow(i)) * 1 / 2));
    width2_i2 = Label_i2(width2_i2);
    % 相减获得单个脉宽
    width2_i = (width2_i2 - width2_i1) / Fs;
    width2List = [width1List, width2_i];
end
width2 = mean(width2List);

% 特征12:计算Pm（周期平均值:周期函数在一个周期内的均值）
PmList = [];
for i = 1: length(LowLabelLowPlace)
    Pm_i = mean(cell2mat(data(HighLabelHighPlace(i): HighLabelHighPlace(i + 1))));
    PmList = [PmList, Pm_i];
end
Pm = mean(PmList);

% 特征13:计算K（K=(Pm-Pd)/(Ps-Pd)，Ps、Pd 分别为周期内的波峰、波谷值）
KList = [];
for i = 1: length(LowLabelLowPlace)
    Pm_i = mean(cell2mat(data(HighLabelHighPlace(i): HighLabelHighPlace(i + 1))));
    Ps_i = HighLabelHigh(i);
    Pd_i = LowLabelLow(i);
    K_i = (Pm_i - Pd_i) / (Ps_i - Pd_i);
    KList = [KList, K_i];
end
K = mean(KList);

% 特征23:计算SV（心搏输出量，SV=0.283/K^2*T*Ps*Pd）
SVList = [];
for i = 1: length(LowLabelLowPlace)
    SV_i = 0.283 / (KList(i)) ^ 2 * TList(i) * HighLabelHigh(i) * LowLabelLow(i);
    SVList = [SVList, SV_i];
end
SV = mean(SVList);

% 特征24:计算TPR（外周阻力，TPR=Pm/(SV*60/T)）
TPRList = [];
for i = 1: length(LowLabelLowPlace)
    Pm_i = mean(cell2mat(data(HighLabelHighPlace(i): HighLabelHighPlace(i + 1))));
    TPR_i = Pm_i / (SVList(i) * 60 / TList(i));
    TPRList = [TPRList, TPR_i];
end
TPR = mean(TPRList);

% 特征36:计算pks（每个周期脉搏波波峰幅值）
pks = mean(HighLabelHigh);

% 特征37:计算dtpks（脉搏波波峰幅值变化量，即波峰峰值差分）
dtpksList = diff(HighLabelHigh);
dtpks = mean(dtpksList);

% 特征38:计算Td（脉率，每分钟脉搏波数，为脉动周期倒数）
TdList = 1 ./ TList;
Td = mean(TList);

% 特征39:计算dt_T（周期的差分序列，反映心搏周期的微小变化）
dt_TList = diff(TList);
dt_T = mean(dt_TList);

% 特征40:计算RST（收缩时间占脉动周期时间比）
RSTList = STList ./ TList;
RST = mean(RSTList);

% 特征41:计算RDT（舒张时间占脉动周期时间比）
RDTList = DTList ./ TList;
RDT = mean(RDTList);

% 特征42:计算Z（Z=H/(1+ST/DT)，反映每搏心输出量的大小）
ZList = p_tList ./ (1 + STList / DTList);
Z = mean(ZList);

% 特征45:计算k（上升支上升速率，脉搏波上升支单位时间上升幅度）
kList = p_tList ./ STList;
k = mean(kList);

% 特征46:计算p_pd（峰峰间隔时间的倒数）
p_pdList = 1 ./ p_pList;
p_pd = mean(p_pdList);

% 时间特征输出
TimeList = [TimeList, ST, DT, T, p_p, RSD, p_t, width1, width2, Pm, K, ...
    SV, TPR, pks, dtpks, Td, dt_T, RST, RDT, Z, k, p_pd];
end
%% 面积特征
function AreaList = find_area(data)
% 输入：data:ppg信号(2048)
% 输出：AreaList:面积特征列表，包括特征9,10,11,43,44
% 采样频率Fs
Fs = 125;
AreaList = [];
test = cell2mat(data);

% 寻找所有峰值（cell不能直接寻峰）
[HighLabel, HighLabelPlace] = findpeaks(test);
% 大致寻找所有波谷，用于选择正式的寻峰方法
PreLowLabel = -findpeaks(-test);
% 峰值差异太小时，使用老方法，否则使用新方法
% 简单认为峰值范围在总差值的10%以内为太小
if max(HighLabel) - min(HighLabel) <= (max(HighLabel) - min(PreLowLabel)) / 5
    % 波峰,根据波形适当调整幅值和距离限制
    [HighLabelHigh, HighLabelHighPlace] = findpeaks(test,'MINPEAKHEIGHT',0.17,'MINPEAKDISTANCE',ceil(0.35*Fs));
    % 去除首尾峰值
    HighLabelHigh = HighLabelHigh(2: end - 1);
    HighLabelHighPlace = HighLabelHighPlace(2: end - 1);
    % 获得波谷
    LowLabelLow = []; LowLabelLowPlace = [];
    for i = 1: length(HighLabelHighPlace) - 1
        [LowLabelLow_i, LowLabelLowPlaceDelta] = min(test(HighLabelHighPlace(i): HighLabelHighPlace(i + 1)));
        LowLabelLowPlace_i = HighLabelHighPlace(i) + LowLabelLowPlaceDelta - 1;
        LowLabelLow = [LowLabelLow, LowLabelLow_i];
        LowLabelLowPlace = [LowLabelLowPlace, LowLabelLowPlace_i];
    end
else
    % 通过峰值均值，对波峰和重搏波进行分割（波峰比均值大）
    HighLabelMean = mean(HighLabel);
    % 波峰HighLabelHigh;对应地址HighLabelHighPlace
    HighLabelHigh = HighLabel(HighLabel > HighLabelMean);
    HighLabelHighPlace = HighLabelPlace(find(HighLabel > HighLabelMean));
    % 去除首尾波峰
    HighLabelHigh = HighLabelHigh(2: end - 1);
    HighLabelHighPlace = HighLabelHighPlace(2: end - 1);
    % 去除原始首尾波峰后，后续对数据的寻找均在新的首尾波峰间进行，相当于图像形状大致确定，便于后续处理
    % HighLabelHighPlace:波峰横坐标；HighLabelHigh:波峰纵坐标
    % 在波峰之间寻找重搏波（重搏波比均值小）
    HighLabelLow = []; HighLabelLowPlace = [];
    for i = 1: length(HighLabel)
        if HighLabelPlace(i) > HighLabelHighPlace(1) && HighLabelPlace(i) < HighLabelHighPlace(end)
            if HighLabel(i) <= HighLabelMean
                HighLabelLow = [HighLabelLow, HighLabel(i)];
                HighLabelLowPlace = [HighLabelLowPlace, HighLabelPlace(i)];
            end
        end
    end
    % 波谷在波峰之间，且靠近靠后的波峰，循环中通过取反找到波峰间的所有波谷，通过索引找到真正的波谷
    LowLabelLow = []; LowLabelLowPlace = [];
    for i = 1: (length(HighLabelHighPlace) - 1)
        % 波谷在List中
        List = cell2mat(data(HighLabelHighPlace(i): HighLabelHighPlace(i + 1)));
        ListPlace = HighLabelHighPlace(i): HighLabelHighPlace(i + 1);
        % 取反寻峰找所有波谷
        [LowLabel, LowLabelPlace] = findpeaks(-List);
        LowLabel = -LowLabel;
        LowLabelLow = [LowLabelLow, LowLabel(end)];
        LowLabelLowPlace = [LowLabelLowPlace, ListPlace(LowLabelPlace(end))];
    end
end

% 使用f(x)-a的积分作为面积，其中f(x)为ppg信号，a为波谷
% 特征9:计算S_S（升支面积）
S_SList = [];
% 特征10:计算S_D（降支面积）
S_DList = [];
% 特征11:计算AreaR（升支面积与降支面积比值）
AreaRList = [];

for i = 1: length(LowLabelLowPlace)
    % 升支与降支范围
    UpLable = LowLabelLowPlace(i): HighLabelHighPlace(i + 1);
    DownLable = HighLabelHighPlace(i): LowLabelLowPlace(i);
    % 升支与降支取值
    UpData = cell2mat(data(UpLable)) - LowLabelLow(i);
    DownData = cell2mat(data(DownLable)) - LowLabelLow(i);
    % 计算升支面积与降支面积
    S_S_i = trapz(UpData);
    S_D_i = trapz(DownData);
    S_SList = [S_SList, S_S_i];
    S_DList = [S_DList, S_D_i];
    % 计算比值
    AreaR_i = S_S_i / S_D_i;
    AreaRList = [AreaRList, AreaR_i];
end
% 特征9
S_S = mean(S_SList) / Fs;
% 特征10
S_D = mean(S_DList) / Fs;
% 特征11
AreaR = mean(AreaRList);

% 特征43:计算SR_DA（下降面积占比）
SR_DAList = S_DList ./ (S_DList + S_SList);
SR_DA = mean(SR_DAList);

% 特征44:计算SR_SA（上升面积占比）
SR_SAList = S_SList ./ (S_DList + S_SList);
SR_SA = mean(SR_SAList);

% 面积特征输出
AreaList = [AreaList, S_S, S_D, AreaR, SR_DA, SR_SA];
end
%% 一阶微分特征
function Diff1List = find_diff1(data)
% 输入：data:ppg信号(2048)
% 输出：Diff1List:一阶微分特征列表，包括特征14-22
% 采样频率Fs
Fs = 125;
Diff1List = [];
test = cell2mat(data);

% 对数据求一阶导
DataDiffFirst = diff(cell2mat(data), 1);

% 特征14:计算pks_d1（微分序列中的极大值）
% 通过寻峰得到极大值
[HighDiffLabel, HighDiffLabelPlace] = findpeaks(DataDiffFirst);
PreLowLabelLow = -findpeaks(-DataDiffFirst);
% 根据峰值差异确定寻峰方式
if max(HighDiffLabel) - min(HighDiffLabel) <= (max(HighDiffLabel) - min(PreLowLabelLow)) / 5
    % 波峰,根据波形适当调整幅值和距离限制
    [HighLabelHigh, HighLabelHighPlace] = findpeaks(test,'MINPEAKHEIGHT',0.17,'MINPEAKDISTANCE',ceil(0.35*Fs));
    % 去除首尾峰值
    HighLabelHigh = HighLabelHigh(2: end - 1);
    HighLabelHighPlace = HighLabelHighPlace(2: end - 1);
else
    % 通过峰值均值进行分割
    HighDiffLabelMean = mean(HighDiffLabel);
    HighLabelHigh = HighDiffLabel(HighDiffLabel > HighDiffLabelMean);
    HighLabelHighPlace = HighDiffLabelPlace(find(HighDiffLabel > HighDiffLabelMean));
    % 去除首尾波峰
    HighLabelHigh = HighLabelHigh(2: end - 1);
    HighLabelHighPlace = HighLabelHighPlace(2: end - 1);
end
% 计算均值
pks_d1 = mean(HighLabelHigh);

% 特征15:计算T1_diff（一阶微分极小值点距离一阶微分极大值点时间间隔）
% 直接寻找峰值之间的最小值即可
LowLabelLow = []; LowLabelLowPlace = [];
for i = 1: (length(HighLabelHighPlace) - 1)
    % 波谷在List中
    List = DataDiffFirst(HighLabelHighPlace(i): HighLabelHighPlace(i + 1));
    ListPlace = HighLabelHighPlace(i): HighLabelHighPlace(i + 1);
    % 最小值为波谷
    [LowLabelLow_i, LowLabelLowPlace_i] = min(List);
    LowLabelLow = [LowLabelLow, LowLabelLow_i];
    LowLabelLowPlace = [LowLabelLowPlace, ListPlace(LowLabelLowPlace_i)];
end
T1_diffList = [];
for i = 1: length(LowLabelLowPlace)
    T1_diff_i = abs(HighLabelHighPlace(i + 1) - LowLabelLowPlace(i)) / Fs;
    T1_diffList = [T1_diffList, T1_diff_i];
end
T1_diff = mean(T1_diffList);

% 特征16:计算T2_diff（一阶微分极大值点距离脉搏波波峰时间间隔）
% 寻找所有峰值（cell不能直接寻峰）
[HighLabel0, HighLabelPlace0] = findpeaks(test);
% 大致寻找所有波谷，用于选择正式的寻峰方法
PreLowLabel0 = -findpeaks(-test);
% 峰值差异太小时，使用老方法，否则使用新方法
% 简单认为峰值范围在总差值的10%以内为太小
if max(HighLabel0) - min(HighLabel0) <= (max(HighLabel0) - min(PreLowLabel0)) / 5
    % 波峰,根据波形适当调整幅值和距离限制
    [HighLabelHigh0, HighLabelHighPlace0] = findpeaks(test,'MINPEAKHEIGHT',0.17,'MINPEAKDISTANCE',ceil(0.35*Fs));
    % 去除首尾峰值
    HighLabelHigh0 = HighLabelHigh0(2: end - 1);
    HighLabelHighPlace0 = HighLabelHighPlace0(2: end - 1);
    % 获得波谷
    LowLabelLow0 = []; LowLabelLowPlace0 = [];
    for i = 1: length(HighLabelHighPlace0) - 1
        [LowLabelLow_i0, LowLabelLowPlaceDelta0] = min(test(HighLabelHighPlace0(i): HighLabelHighPlace0(i + 1)));
        LowLabelLowPlace_i0 = HighLabelHighPlace0(i) + LowLabelLowPlaceDelta0 - 1;
        LowLabelLow0 = [LowLabelLow0, LowLabelLow_i0];
        LowLabelLowPlace0 = [LowLabelLowPlace0, LowLabelLowPlace_i0];
    end
else
    % 通过峰值均值，对波峰和重搏波进行分割（波峰比均值大）
    HighLabelMean0 = mean(HighLabel0);
    % 波峰HighLabelHigh;对应地址HighLabelHighPlace
    HighLabelHigh0 = HighLabel0(HighLabel0 > HighLabelMean0);
    HighLabelHighPlace0 = HighLabelPlace0(find(HighLabel0 > HighLabelMean0));
    % 去除首尾波峰
    HighLabelHigh0 = HighLabelHigh0(2: end - 1);
    HighLabelHighPlace0 = HighLabelHighPlace0(2: end - 1);
    % 去除原始首尾波峰后，后续对数据的寻找均在新的首尾波峰间进行，相当于图像形状大致确定，便于后续处理
    % HighLabelHighPlace:波峰横坐标；HighLabelHigh:波峰纵坐标
    % 在波峰之间寻找重搏波（重搏波比均值小）
    HighLabelLow0 = []; HighLabelLowPlace0 = [];
    for i = 1: length(HighLabel0)
        if HighLabelPlace0(i) > HighLabelHighPlace0(1) && HighLabelPlace0(i) < HighLabelHighPlace0(end)
            if HighLabel0(i) <= HighLabelMean0
                HighLabelLow0 = [HighLabelLow0, HighLabel0(i)];
                HighLabelLowPlace0 = [HighLabelLowPlace0, HighLabelPlace0(i)];
            end
        end
    end
    % 波谷在波峰之间，且靠近靠后的波峰，循环中通过取反找到波峰间的所有波谷，通过索引找到真正的波谷
    LowLabelLow0 = []; LowLabelLowPlace0 = [];
    for i = 1: (length(HighLabelHighPlace0) - 1)
        % 波谷在List中
        List = cell2mat(data(HighLabelHighPlace0(i): HighLabelHighPlace0(i + 1)));
        ListPlace0 = HighLabelHighPlace0(i): HighLabelHighPlace0(i + 1);
        % 取反寻峰找所有波谷
        [LowLabel0, LowLabelPlace0] = findpeaks(-List);
        LowLabel0 = -LowLabel0;
        LowLabelLow0 = [LowLabelLow0, LowLabel0(end)];
        LowLabelLowPlace0 = [LowLabelLowPlace0, ListPlace0(LowLabelPlace0(end))];
    end
end
% 计算差值
T2_diffList = [];
for i = 1: min(length(HighLabelHighPlace), length(HighLabelHighPlace0))
    T2_diff_i = abs(HighLabelHighPlace(i) - HighLabelHighPlace0(i)) / Fs;
    T2_diffList = [T2_diffList, T2_diff_i];
end
T2_diff = mean(T2_diffList);

% 特征17:计算p1（波峰与变化最快处幅值之差）
p1List = [];
% 特征18:计算p2（变化最快处与脉搏波波谷间幅值的单位时间变化率）
p2List = [];
% 特征19:计算p3（变化最快处与波峰幅值差/波峰波谷幅值差）
p3List = [];
% 特征20:计算t1（变化最快点与波谷时间间隔）
t1List = [];
% 特征21:计算t2（变化最快处与波谷时间间隔/收缩上升时间）
t2List = [];
% 特征22:计算t3（一阶微分极大值幅值（脉搏波幅值变化率）在单位时间变化率）
t3List = [];
% 特征23:计算SV（心搏输出量，SV=0.283/K^2*T*Ps*Pd）
SVList = [];

for i = 1: min(length(HighLabelHighPlace), length(LowLabelLowPlace0))
    p1_i = HighLabelHigh0(i) - cell2mat(data(HighLabelHighPlace(i)));
    p1List = [p1List, p1_i];

    p2_iUp = abs(cell2mat(data(HighLabelHighPlace(i))) - LowLabelLow0(i));
    p2_iDown = abs(HighLabelHighPlace(i) - LowLabelLowPlace0(i));
    p2_i = p2_iUp / p2_iDown;
    p2List = [p2List, p2_i];

    p3_iUp = HighLabelHigh0(i) - cell2mat(data(HighLabelHighPlace(i)));
    p3_iDown = HighLabelHigh0(i) - LowLabelLow0(i);
    p3_i = p3_iUp / p3_iDown;
    p3List = [p3List, p3_i];

    t1_i = abs(HighLabelHighPlace(i) - LowLabelLowPlace0(i)) / Fs;
    t1List = [t1List, t1_i];

    t2_iUp = t1_i;
    t2_iDown = abs(HighLabelHighPlace0(i + 1) - LowLabelLowPlace0(i)) / Fs;
    t2_i = t2_iUp / t2_iDown;
    t2List = [t2List, t2_i];

    t3_i = HighLabelHigh(i) / t1_i;
    t3List = [t3List, t3_i];

end

% 特征17
p1 = mean(p1List);
% 特征18
p2 = mean(p2List);
% 特征19
p3 = mean(p3List);
% 特征20
t1 = mean(t1List);
% 特征21
t2 = mean(t2List);
% 特征22
t3 = mean(t3List);

% 一阶微分特征输出
Diff1List = [Diff1List, pks_d1, T1_diff, T2_diff, p1, p2, p3, t1, t2, t3];
end
%% 频域特征
function FreqList = find_freq(data)
% 输入：data:ppg信号(2048)
% 输出：FreqList:频域特征列表，包括特征25-35,64-78
% 采样频率Fs
Fs = 125;
FreqList = [];
test = cell2mat(data);

N = length(test);% 计算信号长度
Segfft = fft(test);% 进行快速傅里叶变换

Segfft = abs(Segfft(1:N/2+1));% 计算单边幅度谱
Segfft_scaled = Segfft / N;% 考虑采样点数的缩放
Segfft_scaled(2:end-1) = 2 * Segfft_scaled(2:end-1);% 对于单边频谱，直流分量（频率为 0）和最高频率分量不加倍，其他分量加倍
freqs = (0:N/2)*(Fs/N);% 计算频率轴

% 找到 0.3Hz 至 2.1Hz 范围内的频率索引
idx_03 = find(freqs >= 0.3, 1, 'first');
idx_21 = find(freqs <= 2.1, 1, 'last');

% 提取 0.3Hz 至 2.1Hz 范围内的频率和幅值
freqs_range = freqs(idx_03:idx_21);
Segfft_scaled_range = Segfft_scaled(idx_03:idx_21);

% 使用 findpeaks 函数寻找峰值，应用高度和距离条件
height_threshold = mean(Segfft_scaled_range) * 1.3;
distance_threshold = min(round(Fs * 0.1), length(Segfft_scaled_range) - 1) - 1;
[peak_values, peak_indices] = findpeaks(Segfft_scaled_range, 'MinPeakHeight', height_threshold, 'MinPeakDistance', distance_threshold);

% 找到最大峰值的索引
if ~isempty(peak_values)
    [max_peak_value, max_peak_index] = max(peak_values);

    % 找到最大峰值对应的频率作为原始基频
    original_fundamental_frequency = freqs_range(peak_indices(max_peak_index));
    original_max_peak_value = max_peak_value;

    % 进行三次样条采样
    new_freqs = linspace(freqs_range(1), freqs_range(end), 10);
    new_Segfft_scaled = spline(freqs_range, Segfft_scaled_range, new_freqs);

    % 寻找样条采样后的最大幅值及其对应的频率
    [new_max_peak_value, new_max_peak_index] = max(new_Segfft_scaled);
    new_fundamental_frequency = new_freqs(new_max_peak_index);

    % 对比两个基频的幅值，取幅值最大的作为最终的基频
    if new_max_peak_value > original_max_peak_value
        f1_n = new_fundamental_frequency;
        a = new_max_peak_value;
    else
        f1_n = original_fundamental_frequency;
        a = original_max_peak_value;
    end

    % 特征25:计算f1（基频）
    f1 = f1_n;
    % 特征27:计算fp1（基波频谱幅值）
    fp1 = a;
else
    fprintf('在 0.3Hz 至 2.1Hz 范围内未找到符合条件的峰值。\n');
    f1 = Inf;
    fp1 = Inf;
    return;
end

% 确定二次谐波搜索范围
lower_bound = f1_n*1.5;
upper_bound = f1_n*2.5;

% 找到二次谐波搜索范围对应的索引
idx_lower = find(freqs >= lower_bound, 1, 'first');
idx_upper = find(freqs <= upper_bound, 1, 'last');

% 确保索引有效
if ~isempty(idx_lower) && ~isempty(idx_upper)
    % 提取二次谐波搜索范围内的频率和幅值
    freqs_harmonic_range = freqs(idx_lower:idx_upper);

    Segfft_scaled_harmonic_range = Segfft_scaled(idx_lower:idx_upper);

    % 使用 findpeaks 函数寻找二次谐波范围内的峰值，应用高度和距离条件
    height_threshold_harmonic = mean(Segfft_scaled_harmonic_range) * 1.3;
    distance_threshold_harmonic = min(round(Fs * 0.1), length(Segfft_scaled_harmonic_range) - 1) - 1;
    [harmonic_peak_values, harmonic_peak_indices] = findpeaks(Segfft_scaled_harmonic_range, 'MinPeakHeight', height_threshold_harmonic, 'MinPeakDistance', distance_threshold_harmonic);

    % 若有峰值存在
    if ~isempty(harmonic_peak_values)
        % 找到最大峰值的索引
        [max_harmonic_peak_value, max_harmonic_peak_index] = max(harmonic_peak_values);

        % 找到最大峰值对应的频率作为二次谐波
        f2_n = freqs_harmonic_range(harmonic_peak_indices(max_harmonic_peak_index));
        d = max_harmonic_peak_value;

        % 特征26:计算f2（二次谐波频率）
        f2 = f2_n;
        % 特征28:计算fp2（二次谐波频谱幅值）
        fp2 = d;
    else
        fprintf('在二次谐波搜索范围内未找到符合条件的峰值。\n');
        f2 = Inf;
        f2_n = []; % 若未找到二次谐波，将其设为空
        d = [];
    end
else
    fprintf('二次谐波搜索范围无效。\n');
    f2_n = []; % 若搜索范围无效，将其设为空
    d = [];
end

% 特征29:计算fs1（基频左右 0.3Hz 能量）
if ~isempty(f1_n)
    lower_fundamental = f1_n - 0.3;
    upper_fundamental = f1_n + 0.3;
    idx_lower_fundamental = find(freqs >= lower_fundamental, 1, 'first');
    idx_upper_fundamental = find(freqs <= upper_fundamental, 1, 'last');
    if ~isempty(idx_lower_fundamental) && ~isempty(idx_upper_fundamental)
        fs1_n = sum(Segfft_scaled(idx_lower_fundamental:idx_upper_fundamental).^2);
    else
        fs1_n = 0;
    end
else
    fs1_n = 0;
end
fs1 = fs1_n;

% 特征30:计算fs2（二次谐波频率左右 0.3Hz 能量）
if ~isempty(f2_n)
    lower_harmonic = f2_n - 0.3;
    upper_harmonic = f2_n + 0.3;
    idx_lower_harmonic = find(freqs >= lower_harmonic, 1, 'first');
    idx_upper_harmonic = find(freqs <= upper_harmonic, 1, 'last');
    if ~isempty(idx_lower_harmonic) && ~isempty(idx_upper_harmonic)
        fs2_n = sum(Segfft_scaled(idx_lower_harmonic:idx_upper_harmonic).^2);
    else
        fs2_n = 0;
    end
else
    fs2_n = 0;
end
fs2 = fs2_n;

% 特征31:计算fs3（总能量（0.3 - 10Hz））
idx_03_total = find(freqs >= 0.3, 1, 'first');
idx_10_total = find(freqs <= 10, 1, 'last');
if ~isempty(idx_03_total) && ~isempty(idx_10_total)
    fs3_n = sum(Segfft_scaled(idx_03_total:idx_10_total).^2);
else
    fs3_n = 0;
end
fs3 = fs3_n;

% 特征32:计算fs4（基波能量百分比,基频左右各0.3Hz）
fs4 = fs1 / fs3;

% 特征33:计算fs5（二次谐波能量百分比）
fs5 = fs2 / fs3;

% 特征34:计算fps1（基波能量/基波频谱幅值）
fps1 = fs1 / a;

% 特征35:计算fps2（二次谐波能量/二次谐波频谱幅值）
fps2 = fs2 / d;


% 特征64-70:计算pser_a0-pser_a6（不同频率范围内的能量）
% 根据0.25Hz, 1.5Hz, 3.5Hz, 7Hz, 10Hz, 20Hz分段
wsp = abs(Segfft / Fs);
k1=floor(0.25*length(wsp)/125);%频谱中，0.25Hz对应下标，频率*NFS/FPS
k2=floor(1.5*length(wsp)/125);%1.5Hz下标
k3=floor(3.5*length(wsp)/125);%3.5Hz下标
k4=floor(7*length(wsp)/125);%7Hz下标
k5=floor(10*length(wsp)/125);%10Hz下标
k6=floor(20*length(wsp)/125);%20Hz下标
% 特征64:10-20Hz
pser_a0 = sum(wsp(k5: k6) .^ 2);
% 特征65:7-10Hz
pser_a1 = sum(wsp(k4: k5) .^ 2);
% 特征66:3.5-7Hz
pser_a2 = sum(wsp(k3: k4) .^ 2);
% 特征67:1.5-3.5Hz
pser_a3 = sum(wsp(k2: k3) .^ 2);
% 特征68:0.25-1.5Hz
pser_a4 = sum(wsp(k1: k2) .^ 2);
% 特征69:0.25-3.5Hz
pser_a5 = sum(wsp(k1: k3) .^ 2);
% 特征70:0.25-10Hz
pser_a6 = sum(wsp(k1: k5) .^ 2);
% 0.25-20Hz
pser_all = sum(wsp(k1: k6) .^ 2);

% 特征71-78:计算r_a0-r_a7（不同频率范围内的能量比例）
% 特征71:10-20Hz/0.25-20Hz
r_a0 = pser_a0 / pser_all;
% 特征72:7-10Hz/0.25-20Hz
r_a1 = pser_a1 / pser_all;
% 特征73:3.5-7Hz/0.25-20Hz
r_a2 = pser_a2 / pser_all;
% 特征74:1.5-3.5Hz/0.25-20Hz
r_a3 = pser_a3 / pser_all;
% 特征75:0.25-1.5Hz/0.25-20Hz
r_a4 = pser_a4 / pser_all;
% 特征76:0.25-3.5Hz/0.25-20Hz
r_a5 = pser_a5 / pser_all;
% 特征77:0.25-10Hz/0.25-20Hz
r_a6 = pser_a6 / pser_all;
% 特征78:0.25-3.5Hz/0.25-10Hz
r_a7 = pser_a5 / pser_a6;

FreqList = [FreqList, f1, f2, fp1, fp2, fs1, fs2, fs3, fs4, fs5, fps1, fps2, ...
    pser_a0, pser_a1, pser_a2, pser_a3, pser_a4, pser_a5, pser_a6, ...
    r_a0, r_a1, r_a2, r_a3, r_a4, r_a5, r_a6, r_a7];
end
%% 小波变换特征
function WTList = find_WT(data)
% 输入：data:ppg信号(2048)
% 输出：WTList:小波变换特征列表，包括特征47-63
% 采样频率Fs
Fs = 125;
WTList = [];
test = cell2mat(data);

% 使用db6小波基分解
[C, L] = wavedec(test, 6, 'db6');
% 获得单支重构信号
a1 = wrcoef('a', C, L, 'db6', 1);
a2 = wrcoef('a', C, L, 'db6', 2);
a3 = wrcoef('a', C, L, 'db6', 3);
a4 = wrcoef('a', C, L, 'db6', 4);
a5 = wrcoef('a', C, L, 'db6', 5);
a6 = wrcoef('a', C, L, 'db6', 6);
aiList = [a1; a2; a3; a4; a5; a6];
% 获得重构细节系数
d1 = wrcoef('d', C, L, 'db6', 1);
d2 = wrcoef('d', C, L, 'db6', 2);
d3 = wrcoef('d', C, L, 'db6', 3);
d4 = wrcoef('d', C, L, 'db6', 4);
d5 = wrcoef('d', C, L, 'db6', 5);
d6 = wrcoef('d', C, L, 'db6', 6);
diList = [d1; d2; d3; d4; d5; d6];
% 第3-6层小波细节系数能量百分比
EdList = [];
for i = 3: 6
    % 各层细节系数求和
    Ed_i = sum(diList(i, :) .^ 2);
    EdList = [EdList, Ed_i];
end
EdSum = sum(EdList);
% 特征47-50:计算Ed1-Ed4（第3-6层小波细节系数能量百分比）
EdratioList = [];
for i = 1: 4
    Edratio_i = EdList(i) / EdSum;
    EdratioList = [EdratioList, Edratio_i];
end

% 特征51-59:计算E1-E9（9个IMF分量的能量矩）
% 获取IMF分量，第1列是原始数据，最后1列是残差，中间为IMF
% 由于输入数据长度为2048，获得的IMF数量为10个（n = fix(log2(2048))-1）
IMFList = eemd(test,0.1,4);
EpowerUpList = [];
for i = 2: size(IMFList, 2) - 1
    EpowerUp_i = 0;
    for j = 1: size(IMFList, 1)
        EpowerUp_i = EpowerUp_i + (j / Fs) * (IMFList(j, i) ^ 2);
    end
    EpowerUpList = [EpowerUpList, EpowerUp_i];
end
EpowerSum = sum(EpowerUpList);
EpowerList = [];
for i = 1: size(EpowerUpList, 2)
    Epower_i = EpowerUpList(i) / EpowerSum;
    EpowerList =[EpowerList, Epower_i];
end

% 特征60-63:计算Eb1-Eb4（第3-6层IMF的边际谱能量）
EbjpList = [];
timeList = (1: length(test)) / Fs;
c = IMFList';
HilbertTest = hilbert(test);
% 区分实部虚部
HilbertTestReal = real(HilbertTest);
HilbertTestImag = imag(HilbertTest);
% 计算瞬时振幅（实部虚部平方和开方）
ShunshiZhenfu = power(HilbertTestReal .^ 2 + HilbertTestImag .^ 2, 0.5);
% 计算瞬时相位
ShunshiXiangwei = angle(HilbertTest);
% 计算瞬时频率
dx = diff(ShunshiXiangwei);
dt = diff(timeList);
ShunshiPinlv = dx ./ dt;
for j = 3: 6
    % 1. 取出IMF并平滑 (使用Savitzky-Golay滤波器)
    imf_raw = IMFList(:, j);
    imf_smooth = sgolayfilt(imf_raw, 3, 11); % 3阶多项式，窗口大小11
    
    % 2. 生成解析信号
    an = hilbert(imf_smooth);
    
    % 3. 尝试调用 hhspectrum，包裹 try-catch 防止崩溃
    try
        [A, fa, tt] = hhspectrum(an, [], Fs); % 注意这里传入采样频率
        [Em, tt1, ff1] = toimage(A, fa, tt);
        
        % 4. 查找 0.5-3.5Hz 索引
        idx_low = find(ff1 >= 0.5, 1);
        idx_high = find(ff1 <= 3.5, 1, 'last');
        
        if ~isempty(idx_low) && ~isempty(idx_high)
            % 对边际谱 Em 积分
            bjp = sum(Em(idx_low:idx_high, :), 2) / Fs;
            EbjpList_i = sum(bjp .^ 2);
        else
            EbjpList_i = 0;
        end
    catch
        EbjpList_i = 0; % 如果报错，赋予默认值
    end
    
    EbjpList = [EbjpList, EbjpList_i];
end

WTList = [WTList, EdratioList(1), EdratioList(2), EdratioList(3), EdratioList(4), ...
    EpowerList(1), EpowerList(2), EpowerList(3), EpowerList(4), EpowerList(5), ...
    EpowerList(6), EpowerList(7), EpowerList(8), EpowerList(9), ...
    EbjpList(1), EbjpList(2), EbjpList(3), EbjpList(4)];
end
%% 填充函数
function List = get_list(ListLabel, sthList, List)
for i = 1: length(ListLabel)
    Label = ListLabel(i);
    List(Label) = sthList(i);
end
end