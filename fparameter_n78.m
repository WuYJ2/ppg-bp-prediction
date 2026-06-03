function varargout=fparameter_n78(signal,paramter,n0)%%
%paramter读取特征表，即78个特征对应名称
close all;
Fs=125;
test=signal;
%%
[pks,locs] = findpeaks(test,'MINPEAKHEIGHT',0.17,'MINPEAKDISTANCE',ceil(0.35*Fs));
%波峰,根据波形适当调整幅值和距离限制，幅值限制是不是最好改成与具体信号相关？距离限制0.2s是否太短？
for i=1:length(locs)-1%假如此时有1:n个峰值位置数据，n为定值
    [a,b]=min(test(locs(i):locs(i+1)));
    tough(i)=locs(i)+b-1;%波谷，获得1:n-1个谷值位置数据
end
pks_1=pks(2:length(pks)-1);
pks_1_sorted = sort(pks_1);
pks_1_trimmed = pks_1_sorted(2:end-1); % 去除最大值和最小值
pks_n = mean(pks_1_trimmed);
%%特征36，每个周期脉搏波波峰幅值，掐头去尾，此时留有2:n-1个峰值幅值数据
locs=locs(2:length(locs)-1);%留有2:n-1个峰值位置数据
% tough_n=mean(tough);
% 
% max_n=max(test);
% min_n=min(test);
% xmax=(max_n-tough_n)/(pks_n-tough_n);
% xmin=(min_n-tough_n)/(pks_n-tough_n);
% 
% tsx_test=test;
% %tsx_test=tsx_test';
% [xmap_test,xmap_set] = mapminmax(tsx_test,xmin,xmax);%使用训练集归一化范围
% test=xmap_test;
%%尝试提取特征过程中使用波峰波谷值归一化，结果一般，可以考虑使用更好的方法

dtpks_1=diff(pks_1);
dtpks_1_sorted = sort(dtpks_1);
dtpks_1_trimmed = dtpks_1_sorted(2:end-1); % 去除最大值和最小值
dtpks_n = mean(dtpks_1_trimmed);
%%特征37，脉搏波波峰幅值变化量，即波峰峰值差分，共n-3个数据
for j=1:length(pks_1)%1:n-2
    ST_1(j)=(locs(j)-tough(j))/Fs;
    DT_1(j)=(tough(j+1)-locs(j))/Fs;
end
ST_1_sorted = sort(ST_1);
ST_1_trimmed = ST_1_sorted(2:end-1);
ST_n = mean(ST_1_trimmed);
%%特征1，收缩时间，单个脉搏波周期内，脉搏波波谷到波峰的时间间隔。这里共求得n-2个数据
DT_1_sorted = sort(DT_1);
DT_1_trimmed = DT_1_sorted(2:end-1);
DT_n = mean(DT_1_trimmed);
%%特征2，舒张时间，单个脉搏波周期内，脉搏波波峰到波谷的时间间隔。这里共求得n-2个数据
%%
T_1 = ST_1+DT_1;
T_1_sorted = sort(T_1);
T_1_trimmed = T_1_sorted(2:end-1);
T_n = mean(T_1_trimmed);
%%特征3，脉动周期，每次心搏所用时间，T=ST+DT，为相邻波谷间的时间间隔。这里共求得n-2个数据
Td_1 = 1./T_1;
Td_n = mean(Td_1);
%%特征38，脉率，每分钟脉搏波数，为脉动周期倒数，n-2个数据。不等同于心率，但如需从脉搏波获取心率只能以此为标准
dt_T_1=diff(T_1);
dt_T_1_sorted = sort(dt_T_1);
dt_T_1_trimmed = dt_T_1_sorted(2:end-1); % 去除最大值和最小值
dt_T_n = mean(dt_T_1_trimmed);
%%特征39，周期的差分序列，反映心搏周期的微小变化n-3个数据
RST_1=ST_1./T_1;
RST_1_sorted = sort(RST_1);
RST_1_trimmed = RST_1_sorted(2:end-1); % 去除最大值和最小值
RST_n = mean(RST_1_trimmed);
%%特征40，收缩时间占脉动周期时间比，n-2个数据
RDT_1=DT_1./T_1;
RDT_1_sorted = sort(RDT_1);
RDT_1_trimmed = RDT_1_sorted(2:end-1); % 去除最大值和最小值
RDT_n = mean(RDT_1_trimmed);
%%特征41，舒张时间占脉动周期时间比，n-2个数据
RSD_1=ST_1./DT_1;
RSD_1_sorted = sort(RSD_1);
RSD_1_trimmed = RSD_1_sorted(2:end-1); % 去除最大值和最小值
RSD_n = mean(RSD_1_trimmed);
%%特征5，收缩时间与舒张时间比值，n-2个数据
%%
p_p_1=diff(locs)/Fs;
p_p_1_sorted = sort(p_p_1);
p_p_1_trimmed = p_p_1_sorted(2:end-1); % 去除最大值和最小值
p_p_n = mean(p_p_1_trimmed);
%%特征4，峰值位置数据差分的时长，即峰峰间隔时间，n-3个数据 % ！！！！！这个和T不一样。T是先升再降（谷到谷），p_p是先降再升（峰到峰）
p_pd_1=1./p_p_1;
p_pd_1_sorted = sort(p_pd_1);
p_pd_1_trimmed = p_pd_1_sorted(2:end-1); % 去除最大值和最小值
p_pd_n = mean(p_pd_1_trimmed);
%%特征46，峰峰间隔时间的倒数，n-3个数据
p_t_1=pks_1-test(tough(1:length(tough)-1));
p_t_1_sorted = sort(p_t_1);
p_t_1_trimmed = p_t_1_sorted(2:end-1); % 去除最大值和最小值
p_t_n = mean(p_t_1_trimmed);
%%特征6，波峰波谷幅值之差，上升支高度，2:n-1，n-2个数据
k_1=p_t_1./ST_1;
k_1_sorted = sort(k_1);
k_1_trimmed = k_1_sorted(2:end-1); % 去除最大值和最小值
k_n = mean(k_1_trimmed);
%%特征45，上升支上升速率，脉搏波上升支单位时间上升幅度，n-2个数据
Z_1=p_t_1/(1+ST_1/DT_1);
Z_1_sorted = sort(Z_1);
Z_1_trimmed = Z_1_sorted(2:end-1); % 去除最大值和最小值
Z_n = mean(Z_1_trimmed);
%%特征42，Z=H/(1+ST/DT)，反映每搏心输出量的大小，n-2个数据
%%
for i=1:length(pks_1)%n-2
    %距离波谷2/3倍的波峰波谷幅值差的点的位置
    [~,II1]=min(abs(test(tough(i):locs(i))-(test(tough(i))+p_t_1(i)/3*2)));
    %上升支最小值点（绝对值（上升支整体幅值-谷值-2/3峰谷幅值差）））
    [~,II2]=min(abs(test(locs(i):tough(i+1))-(test(tough(i))+p_t_1(i)/3*2)));
    %下降支最小值点（绝对值（下降支整体幅值-谷值-2/3峰谷幅值差）））
    width1_1(i)=(locs(i)+II2-(tough(i)+II1))/Fs;
    %%特征7，2/3脉宽，n-2个值
    [ww1,II1]=min(abs(test(tough(i):locs(i))-(test(tough(i))+p_t_1(i)/2)));
    [ww2,II2]=min(abs(test(locs(i):tough(i+1))-(test(tough(i))+p_t_1(i)/2)));
    width2_1(i)=(locs(i)+II2-(tough(i)+II1))/Fs;
    %%特征8，1/2脉宽，n-2个值
end
width1_1_sorted = sort(width1_1);
width1_1_trimmed = width1_1_sorted(2:end-1); % 去除最大值和最小值
width1_n = mean(width1_1_trimmed);
%%特征7，2/3脉宽
width2_1_sorted = sort(width2_1);
width2_1_trimmed = width2_1_sorted(2:end-1); % 去除最大值和最小值
width2_n = mean(width2_1_trimmed);
%%特征8，1/2脉宽
%% 面积
for i=1:length(pks_1)%n-2
    x1=(tough(i):locs(i))/Fs;%上升支时间点
    y1=test(tough(i):locs(i));%上升支幅值
    x2=(locs(i):tough(i+1))/Fs;%下降支时间点
    y2=test(locs(i):tough(i+1));%下降支幅值
    S_S_1(i)=trapz(x1,y1-repmat(test(tough(i)),1,length(y1)));
    %%特征9，上升面积，把波谷当基线，即将幅值均减去谷值计算，n-2
    S_D_1(i)=trapz(x2,abs(y2-repmat(test(tough(i)),1,length(y2))));
    %%特征10，下降面积（减基线漂移），取绝对值，n-2     S_DD(i)=trapz(x2,y2'-repmat(test(tough(i)),1,length(y2)));%下降面积，未取绝对值
    AreaR_1(i)=S_S_1(i)/S_D_1(i);
    %%特征11，上升面积与下降面积的比值，n-2，但文件中：AreaR=(SS+DS)/T
    Pm_1(i)=mean(test(tough(1):(tough(i+1))));%原计算式：(S_S_1(i)+S_D_1(i))/((tough(i+1)-tough(i)/Fs));%单位时间面积变化
    %%特征12，Pm心动周期内的时域平均值，n-2
    K_1(i)=(Pm_1(i)-test(tough(i)))/p_t_1(i);
    %%特征13，K=(Pm-Pd)/(Ps-Pd)，Ps是pks(i+1)波峰幅值数据，Pd是test(tough(i))波谷幅值数据，n-2
end
S_S_1_sorted = sort(S_S_1);
S_S_1_trimmed = S_S_1_sorted(2:end-1); % 去除最大值和最小值
S_S_n = mean(S_S_1_trimmed);
%%特征9，上升面积
S_D_1_sorted = sort(S_D_1);
S_D_1_trimmed = S_D_1_sorted(2:end-1); % 去除最大值和最小值
S_D_n = mean(S_D_1_trimmed);
%%特征10，下降面积
AreaR_1_sorted = sort(AreaR_1);
AreaR_1_trimmed = AreaR_1_sorted(2:end-1); % 去除最大值和最小值
AreaR_n = mean(AreaR_1_trimmed);
%%特征11，上升面积与下降面积的比值
Pm_1_sorted = sort(Pm_1);
Pm_1_trimmed = Pm_1_sorted(2:end-1); % 去除最大值和最小值
Pm_n = mean(Pm_1_trimmed);
%%特征12，Pm心动周期内的时域平均值
K_1_sorted = sort(K_1);
K_1_trimmed = K_1_sorted(2:end-1); % 去除最大值和最小值
K_n = mean(K_1_trimmed);
%%特征13，K=(Pm-Pd)/(Ps-Pd)
SR_SA_1=S_S_1./(S_S_1+S_D_1);
SR_SA_1_sorted = sort(SR_SA_1);
SR_SA_1_trimmed = SR_SA_1_sorted(2:end-1); % 去除最大值和最小值
SR_SA_n = mean(SR_SA_1_trimmed);
%%特征44，上升面积占比，n-2
SR_DA_1=S_D_1./(S_S_1+S_D_1);
SR_DA_1_sorted = sort(SR_DA_1);
SR_DA_1_trimmed = SR_DA_1_sorted(2:end-1); % 去除最大值和最小值
SR_DA_n = mean(SR_DA_1_trimmed);
%%特征43，下降面积占比，n-2
%% 一阶微分
test_diff=diff(test);
[pks_d1_1,locs_d1] = findpeaks(test_diff(tough(1):tough(end)),'MINPEAKHEIGHT',min(0.35*max(test_diff),0.2*abs(min(test_diff))),'MINPEAKDISTANCE',ceil(0.35*Fs));

pks_d1_1_sorted = sort(pks_d1_1);
pks_d1_1_trimmed = pks_d1_1_sorted(2:end-1); 
pks_d1_n=mean(pks_d1_1_trimmed);
%%特征14，一阶微分极大值点幅值，从第一个波谷开始寻找，1:n-1
locs_d1=locs_d1'+repmat(tough(1),1,length(locs_d1));%确定一阶微分极大值点位置，1:n-1

for i=1:min(length(locs_d1),length(locs))-1
    [~,b]=min(test_diff(locs_d1(i):locs_d1(i+1)));%b返回索引，确定一阶微分极小值点位置，1:n-2
    %[a,b]=min(abs(test_diff(locs_d1(i):locs_d1(i)+b-1)));
    T1_diff_1(i)=(b-1)/Fs;
    %%特征15，一阶微分极小值点距离一阶微分极大值点时间间隔
    T2_diff_1(i)=(locs(i)-locs_d1(i))/Fs;
    %%特征16，一阶微分极大值点距离脉搏波波峰时间间隔
    %T3_diff_1(i)=(locs_d1(i)-tough(i))/Fs;%%一阶微分极大值点(i)距离脉搏波波谷(i)时间间隔
    xx(i)=b+locs_d1(i)-1;%%确定一阶微分极小值点位置，1:n-2
end
T1_diff_1_sorted = sort(T1_diff_1);
T1_diff_1_trimmed = T1_diff_1_sorted(2:end-1);
T1_diff_n=mean(T1_diff_1_trimmed);
%%特征15，一阶微分极小值点距离一阶微分极大值点时间间隔
T2_diff_1_sorted = sort(T2_diff_1);
T2_diff_1_trimmed = T2_diff_1_sorted(2:end-1);
T2_diff_n=mean(T2_diff_1_trimmed);
%%特征16，一阶微分极大值点距离脉搏波波峰时间间隔

%figure;
% plot((1:length(test_diff))/Fs,test_diff,'k');%一阶微分
%hold on;
%plot(locs_d1/Fs,pks_d1_1,'r*');%波峰
%hold on;
%plot(xx/Fs,test_diff(xx),'k^');%波谷
%%
n_length=min([length(ST_1) length(pks_1) length(locs_d1)]);%n-1
% for i=1:length(ST_1)
for i=1:n_length
    p1_1(i)=pks_1(i)-test(locs_d1(i));
    %%特征17，波峰与变化最快处幅值之差
    t1_1(i)=(locs_d1(i)-tough(i))/Fs;
    %%特征20，变化最快点与波谷时间间隔
    p2_1(i)=(test(locs_d1(i))-test(tough(i)))/((locs_d1(i)-tough(i))/Fs);
    %%特征18，脉搏波变化最快处与脉搏波波谷间幅值的单位时间变化率
    p3_1(i)=p1_1(i)/p_t_1(i);
    %%特征19，变化最快处与波峰幅值差/波峰波谷幅值差
    t2_1(i)=t1_1(i)/ST_1(i);
    %%特征21，变化最快处与波谷时间间隔/收缩上升时间
    t3_1(i)=pks_d1_1(i)/t1_1(i);
    %%特征22，一阶微分极大值幅值（脉搏波幅值变化率）在单位时间变化率
    SV_1(i)=0.283./power(K_1(i),2)*T_1(i)*(pks_1(i)-test(tough(i)));
    %%特征23，心搏输出量，SV=0.283/K^2*T*Ps*Pd
    TPR_1(i)=Pm_1(i)/(SV_1(i)*60/T_1(i));
    %%特征24，外周阻力，TPR=Pm/(SV*60/T)
end
p1_1_sorted = sort(p1_1);
p1_1_trimmed = p1_1_sorted(2:end-1);
p1_n=mean(p1_1_trimmed);
%%特征17，波峰与变化最快处幅值之差
t1_1_sorted = sort(t1_1);
t1_1_trimmed = t1_1_sorted(2:end-1);
t1_n=mean(t1_1_trimmed);
%%特征20，变化最快点与波谷时间间隔
p2_1_sorted = sort(p2_1);
p2_1_trimmed = p2_1_sorted(2:end-1);
p2_n=mean(p2_1_trimmed);
%%特征18，脉搏波变化最快处与脉搏波波谷间幅值的单位时间变化率
p3_1_sorted = sort(p3_1);
p3_1_trimmed = p3_1_sorted(2:end-1);
p3_n=mean(p3_1_trimmed);
%%特征19，变化最快处与波峰幅值差/波峰波谷幅值差
t2_1_sorted = sort(t2_1);
t2_1_trimmed = t2_1_sorted(2:end-1);
t2_n=mean(t2_1_trimmed);
%%特征21，变化最快处与波谷时间间隔/收缩上升时间
t3_1_sorted = sort(t3_1);
t3_1_trimmed = t3_1_sorted(2:end-1);
t3_n=mean(t3_1_trimmed);
%%特征22，一阶微分极大值幅值（脉搏波幅值变化率）在单位时间变化率
SV_1_sorted = sort(SV_1);
SV_1_trimmed = SV_1_sorted(2:end-1);
SV_n=mean(SV_1_trimmed);
%%特征23，心搏输出量，SV=0.283/K^2*T*Ps*Pd
TPR_1_sorted = sort(TPR_1);
TPR_1_trimmed = TPR_1_sorted(2:end-1);
TPR_n=mean(TPR_1_trimmed);
%%特征24，外周阻力，TPR=Pm/(SV*60/T)
%% 频域特征
zz=10;
wft=fft(test);%test(1:2048)
wsp=abs(wft/125);

N = length(test);
half_N = floor(N/2);
% 先计算单边频谱的频率轴和幅值
f_single_side = (0:half_N - 1)*(125/N);
wsp_single_side = wsp(1:half_N);

k1=floor(0.3*length(wsp)/125);%频谱中，0.3Hz对应下标，频率*NFS/FPS
k2=floor(10*length(wsp)/125);%10Hz下标
k3=floor(2.1*length(wsp)/125);%2.1Hz下标
k4=floor((2*Td_n-0.2)*length(wsp)/125);
k5=floor((2*Td_n+0.2)*length(wsp)/125);
[a,b]=max(wsp(k1:k3));%基频下标偏移
[c,d]=max(wsp(k4:k5));%二次谐波频率下标偏移
f1_n=(b+k1-1)*125/length(wsp);%!!
%%特征25，基频，下标转频率
f2_n=(d+k4-1)*125/length(wsp);
%%特征26，二次谐波，下标转频率
fp1_n=a;
%%特征27，基波频谱幅值
fp2_n=c;
%%特征28，二次谐波频谱幅值
fs1_n=sum(power(wsp(b+k1-k1:b+k1+k1),2));
%%特征29，基频左右0.3Hz能量
fs2_n=sum(power(wsp(d+k4-k1:d+k4+k1),2));
%%特征30，二次谐波频率左右0.3Hz能量
fs3_n=sum(power(wsp(k1:k2),2));
%%特征31，总能量（0.3-10Hz）
fs4_n=fs1_n/fs3_n;
%%特征32，基波能量百分比,基频左右各0.3Hz
fs5_n=fs2_n/fs3_n;
%%特征33，二次谐波能量百分比 P5
fps1_n=fs1_n/a;
%%特征34，基波能量/基波频谱幅值
fps2_n=fs2_n/d;
%%特征35，二次谐波能量/二次谐波频谱幅值

% 新增绘制单边频谱图部分
%figure; % 创建新的图形窗口
%plot(f_single_side, wsp_single_side); % 绘制单边频谱
%hold on;
% 标注基频和二次谐波相关信息
%plot([f1_n, f1_n], [0, fp1_n], 'r--', 'LineWidth', 1.5); % 标注基频
%text(f1_n, fp1_n, ['f1 = ', num2str(f1_n),'Hz']);
%plot([f2_n, f2_n], [0, fp2_n], 'g--', 'LineWidth', 1.5); % 标注二次谐波
%text(f2_n, fp2_n, ['f2 = ', num2str(f2_n),'Hz']);
%title('单边频谱图');
%xlabel('频率(Hz)');
%ylabel('幅值');
%grid on;
%hold off;
%% 小波频域特征：谱能比、小波熵、IMF能量矩、小波系数能量矩阵、基于HHT边际谱的特征能量 %??单个脉搏波周期延拓后的谱能比
k1=floor(0.25*length(wsp)/125);%频谱中，0.25Hz对应下标，频率*NFS/FPS
k2=floor(1.5*length(wsp)/125);%1.5Hz下标
k3=floor(3.5*length(wsp)/125);%3.5Hz下标
k4=floor(7*length(wsp)/125);%7Hz下标
k5=floor(10*length(wsp)/125);%10Hz下标
k6=floor(20*length(wsp)/125);%20Hz下标
pser_a0_n=sum(power(wsp(k5:k6),2));
%%特征64，10-20Hz能量
pser_a1_n=sum(power(wsp(k4:k5),2));
%%特征65，7-10Hz能量
pser_a2_n=sum(power(wsp(k3:k4),2));
%%特征66，3.5-7Hz能量
pser_a3_n=sum(power(wsp(k2:k3),2));
%%特征67，1.5-3.5Hz能量
pser_a4_n=sum(power(wsp(k1:k2),2));
%%特征68，0.25-1.5Hz能量,按照频率算得的能量
pser_a5_n=sum(power(wsp(k1:k3),2));
%%特征69，0.25-3.5Hz能量    a5
pser_a6_n=sum(power(wsp(k1:k5),2));
%%特征70，0.25-10Hz能量     a6
pser_all_n=sum(power(wsp(k1:k6),2));
%总能量（0.25-20Hz）
%% ratio
r_a0_n=pser_a0_n/pser_all_n;%10-20Hz/0.25-20Hz
r_a1_n=pser_a1_n/pser_all_n;%7-10Hz/0.25-20Hz
r_a2_n=pser_a2_n/pser_all_n;%3.5-7Hz/0.25-20Hz
r_a3_n=pser_a3_n/pser_all_n;%1.5-3.5Hz/0.25-20Hz
r_a4_n=pser_a4_n/pser_all_n;%0.25-1.5Hz/0.25-20Hz
r_a5_n=pser_a5_n/pser_all_n;
%%特征76，0.25-3.5Hz/0.25-20Hz Ra5
r_a6_n=pser_a6_n/pser_all_n;
%%特征77，0.25-10Hz/0.25-20Hz  Ra6
r_a7_n=pser_a5_n/pser_a6_n;
%%特征78，0.25-3.5Hz/0.25-10Hz
%% 小波变换
[C, L] = wavedec(test,6,'db6');%采用db6小波基分解
a1=wrcoef('a',C,L,'db6',1);%从系数得到近似系数，近似系数即低频轮廓信息，概貌信息
a2=wrcoef('a',C,L,'db6',2);%从系数得到近似系数,wrcoef对一维小波系数进行单支重构
a3=wrcoef('a',C,L,'db6',3);
a4=wrcoef('a',C,L,'db6',4);
a5=wrcoef('a',C,L,'db6',5);
a6=wrcoef('a',C,L,'db6',6);
d1=wrcoef('d',C,L,'db6',1);%细节系数高频信息
d2=wrcoef('d',C,L,'db6',2);
d3=wrcoef('d',C,L,'db6',3);
d4=wrcoef('d',C,L,'db6',4);
d5=wrcoef('d',C,L,'db6',5);
d6=wrcoef('d',C,L,'db6',6);
dd=[d1;d2;d3;d4;d5;d6];
for j = 3:6 % 6层分解，只看与脉搏波频率相关，且噪声较小的
    Ed(j-2) = sum(power(dd(j,:),2)); %各层系数能量求和
end
sum_Ed=sum(Ed);
for j = 1:4
    Ed_1(j) = Ed(j)/sum_Ed; %小波系数能量百分比
    eval(['Ed',num2str(j),'_n=mean(Ed_1(:,j));']);
end
nn=[d1;d2;d3;d4;d5;d6;a6];
for j = 1:7 % 4层分解，16组系数
    E(j) = sum(power(nn(j,:),2));
end
E1 = sum(E);% 能量求和
dim = length(E);
for j= 1:dim
    p(j)= E(j)./E1;
end
% %小波熵
Entropy_1(i-zz)= -sum(p.*log(p));
Entropy_n=mean(Entropy_1);
%IMF矩
allmode=eemd(test,0.1,4);%c的第1列是原始数据，最后1列是残差，中间为IMF。IMF的个数为fix(log2(N))-1，N为输入数据长度
for k=2:size(allmode,2)-1
    E(k)=0;
    for j=1:size(allmode,1)
        E(k)=E(k)+j/Fs*power(allmode(j,k),2);
    end
end
tot=sum(E);
for j=1:size(E,2)
    E_1(j)=E(j)/tot;
    eval(['E',num2str(j),'_n=mean(E_1(:,j));']);
end
%HHT边际谱
t=(1:length(test))/Fs;
c=allmode';
hx=hilbert(test);
xr=real(hx);
xi=imag(hx);
%计算瞬时振幅
sz=sqrt(xr.^2+xi.^2);
%计算瞬时相位
sx=angle(hx);
%计算瞬时频率
dt=diff(t);
dx=diff(sx);
sp=dx./dt;
for j=3:6%只看第3-6 IMF
    bjp=0;
    %计算HHT时频谱和边际谱
    [A,fa,tt]=hhspectrum(c(j,:));%%工具包！！！tftb和emd
    [A,fa,tt]=hhspectrum(c(j,k1:k5));%0.25-10HZ
    [Em,tt1]=toimage(A,fa,tt,length(tt));
    Em=flipud(Em);%二维频谱图，Em(i-zz,j)，i表示频率索引，j表示时间索引
    for k=1:size(Em,1)
        bjp(k)=sum(Em(k,:))*1/Fs;%相当于按照时间求和得到的边际谱
    end
    Eb_1(j-2)=sum(power(bjp(floor(0.5*length(test)/Fs):floor(3.5*length(test)/Fs)),2));%feature180831存储的边际谱能量为0.8-3.5Hz的，可以修改为0.5Hz
    eval(['Eb',num2str(j-2),'_n=mean(Eb_1(:,j-2));']);
end
%% 各特征参量的最小维数
num=length(paramter);%特征数量
for j=1:num
    eval([cell2mat(paramter(j)),'=',cell2mat(paramter(j)),'_n.'';']);
end
for j=1:num
    eval(['wine(:,',num2str(j),')=',cell2mat(paramter(j)),'.'';']);
end
varargout{1}=paramter;
varargout{2}=wine;
%varargout{3}=wine_labels;
