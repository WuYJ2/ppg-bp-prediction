function [wine_m]=fmoveaverage_x(wine)
    kk=5;%定义了窗口的大小，这里窗口大小是5
    w=hamming(kk);%成一个长度为 kk 的 Hamming 窗口
    wine=wine';
    for i=1:floor(size(wine,1)./2) 
        cc=wine(i,:);% 取第 i 行的数据
        cc=conv(w,cc); % 使用卷积进行滤波
        wine_m(i,:)=cc(kk:length(cc)-(kk-1))/sum(w); % 对卷积结果进行规范化
    end
    for i=ceil(size(wine,1)./2):size(wine,1)
        wine_m(i,:)=wine(i,floor(kk/2)+1:size(wine,2)-floor(kk/2));
    end
    wine_m=wine_m';
    
    % for i=1:size(wine_labels,2)
    % cc=wine_labels(:,i);
    % cc=conv(w,cc);
    % wine_labels_m(:,i)=cc(kk:length(cc)-(kk-1))/sum(w);
    % end
end
%移动平均滤波