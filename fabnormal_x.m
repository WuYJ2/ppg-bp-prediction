function [wine]=fabnormal_x(wine)
mp=size(wine_labels,1);%mp是样本数，即wine_labels的行数。这里是舒张压
mq=size(wine,2);%mq=78，mq是wine特征数据的列数

for i=1:mq
    mean_f1(i)=mean(wine(:,i));%对于wine矩阵中的每个特征列，计算该列的均值mean_f1(i)
    for j=1:mp
        if(wine(j,i)>mean_f1(i)*1.5||wine(j,i)<mean_f1(i)*0.5)
            flag1(j,i)=1;
        else
            flag1(j,i)=0;
        end
    end
end
%然后对每个样本中的特征值进行检查。
% 如果特征值高于均值的1.5倍或低于均值的0.5倍，则将对应位置的flag1标记为1（表示异常），否则标记为0。

for k=1:size(wine_labels,2)%增加一列，将收缩压也作为一个特征
    for j=2:mp
        if(wine_labels(j,k)-wine_labels(j-1,k)>5||wine_labels(j,k)-wine_labels(j-1,k)<-5)
            flag1(j,mq+k)=1;
        else
            flag1(j,mq+k)=0;
        end
    end
end
%对wine_labels矩阵的每一列，检查相邻样本的差值变化是否大于5或小于-5。
%如果相邻样本的差值变化较大，则将flag1的对应位置标记为1（表示异常），并将收缩压视作额外特征，追加到flag1矩阵的列数中。

for i=1:mp
    num1(i)=0;
    for j=1:mq+size(wine_labels,2)
        if(flag1(i,j)==1)
            num1(i)=num1(i)+1;
        end
    end
end
%遍历每个样本，统计该样本的所有特征标记中异常的数量num1(i)。

k=1;%当前检查被试
c=size(wine_labels,1);%当前总被试数
j=1;%当前检查flag标识
while(k<=mp)
    if(k>c)
        break;
    end
    if(num1(j)>ceil(mq+size(wine_labels,2)))
        train_wine(k,:)=[];
        train_wine_labels(k,:)=[];
        c=c-1;
    else
        k=k+1;
    end
    j=j+1;
end
%遍历每个样本。如果该样本的异常特征数量num1(j)超过特定阈值（mq+size(wine_labels,2)的一半），则认为该样本为异常样本，将其从训练数据train_wine和标签train_wine_labels中删除。
end
%主要功能是检查数据集中每个样本的特征值是否异常，并根据特定条件标记和过滤异常样本。