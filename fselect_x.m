function signal=fselect_x(signal)
%用于去除波形中的异常段（尖峰）
mean_sig=mean(abs(signal));
for i=1:length(signal)
    if(abs(signal)>4*mean_sig)
        %检测数据中绝对值大于4倍平均值的异常数据（可靠嘛？）
        flag(i)=1;
    else
        flag(i)=0;
    end
end
[~,n]=find(flag==1);%返回异常数据位置,n为异常数据位置集合
if(n>0)%异常数据处理
    num=1;%记录正常数据段数
    for i=1:length(n)
        if(i==1)%如果i等于1，则它基于n（1）不等于1时创建信号sig_num
            if(n(i)~=1)
                eval(['sig_',num2str(num),'=signal(1:',num2str(n(i)-1),');']);%第num段正常数据：1到n（1）-1
                num=num+1;
            end
        end
        if(i==length(n))%如果i等于n的长度，则它基于n的先前值和当前值创建信号
            eval(['sig_',num2str(num),'=signal(',num2str(n(i-1)+1),':',num2str(n(i)-1),');']);%第num段正常数据：前一个异常值与此异常值之间
            num=num+1;
            if(n(i)~=length(flag))%如果异常数据未在数据结尾处
                eval(['sig_',num2str(num),'=signal(',num2str(n(i)+1),':end,:);']);%第num段正常数据：最后一个异常值与数据结尾之间
            end
        end
        if(i~=1&&i~=length(n))%对于其他情况（当i不是1或n的长度时），它基于n的连续元素之间的差来创建信号
            if(n(i)-n(i-1)>2)%要求不为连续异常数据（2个数据其实可能也太短了？）
                eval(['sig_',num2str(num),'=signal(',num2str(n(i-1)+1),':',num2str(n(i)-1),');']);%第num段正常数据：前一个异常值与此异常值之间
                num=num+1;
            end
        end
    end
    for i=1:num
        sig_length(i)=length(eval(['sig_',num2str(i)]));%计算每一段正常数据长度
    end
    [~,p]=max(sig_length);
    eval(['signal=sig_',num2str(p),';']);%选取最长的正常信号 
end