function signal=fdenoise_x(signal)
% N=size(signal,1);%信号长度
% k1=2;k2=1;%k1脉搏波，k2血压
% y=signal(1:N,k1);

%N=length(signal);
%y=;
% fileID = fopen('ch2_data4.txt','r');
% %fileID = fopen('real_data_filtered2.txt','r');
% formatSpec = '%d ';
[scriptDir, ~, ~] = fileparts(mfilename('fullpath'));
lpth = scriptDir;
%spth = fullfile(scriptDir, 'data_r');
load(fullfile(lpth, 'base.mat'));%%需要资源包！！！
load(fullfile(lpth, 'lowpass.mat'));
raw_A =signal;% fscanf(fileID,formatSpec);
A = raw_A;%(251 : 2298);%舍弃前2S的上升数据,凑成2048个数据
M_A = A(1 : 2048);
MAX_A = max(A);
B = mean(A) - A;
%A = max(A) - A;
%A = B ; %% 反转波形形成ppg信号
%% 需要反转波形
fs = 125;
Ts = 1/fs;
N = 2048;
nt = 0 : N-1;
t = nt * Ts;
nf = -N /2 : N/2-1;
f = nf * fs/N;
%Hlowpass = F_lowpass;
y = filter(Hlowpass,A);
%%延迟等于滤波器阶数的一半528
filter_lvl = 80/2;
y1 = filter(Hbase,y);
kk1 = y1;
y1 = y1(filter_lvl+1:2048);
y1(2048-filter_lvl-1+1:2048)= y1(2048-filter_lvl+1-1);
y2 = y - y1 ;
signal=y2;

% %% 高频噪声、基线漂移
% %选用基波函数
% %脉搏波信号的分解及合成选择了双正交样条小波作为小波基函数,具体为bio3.5、
% %双正交样条小波是一种双正交对称小波,频率特性好,分频能力强,具有线性相位特点。
% % wname='sym8'; %比较, Sym8小波函数和原始脉搏波信号更为相似,
% %db6、db8、coif5
% wname='coif5';
% %分解级数
% lev=7;
% %小波分解
% [C,L]=wavedec(y,lev,wname);
% C(1:L(1))=0;                  % 消除极低频范围（与基线波相关）   
% %%！！！这个L（1）与信号总长度有关，长度过短时易导致脉搏波信号丢失，是否要考虑及时检测的信号长度设置？
% C(end-sum(L(7:8))+1:end)=0;   %消除超高频率（与电力线谐波和肌肉活动伪像相关）的部分
% %% 重构
% y_1= waverec(C,L,wname);
% %去除高频干扰：使用软Rigrsure阈值策略再进行传统的小波去噪
% signal_den=wden(y_1,'rigrsure','s','sln',lev,wname);
% signal=signal_den;
% figure%%原波形，重构波形与小波去噪后波形的片段对比
% subplot(3,1,1);
% plot(y);xlim([0,1000]);title('原波形');
% subplot(3,1,2);
% plot(y_1);xlim([0,1000]);title('重构波形');
% subplot(3,1,3);
% plot(signal_den);xlim([0,1000]);title('小波去噪波形');
% 


