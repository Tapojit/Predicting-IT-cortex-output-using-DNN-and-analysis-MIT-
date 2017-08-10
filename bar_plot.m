function bar_plot()
%Plots bar graph of explained explainable variance

path_suff='_model.mat';
data_arr=categorical({'HMAX','HMO','Kr','V4','Ze'});
data_arr_2=categorical({'HMAX','HMO','Krizhevsky','V4','Zeiler'});
exp_var_arr=zeros(1,5);
var=zeros(1,5);
matrix=zeros(5,168);
for i=1:size(exp_var_arr,2);
    m=load([char(data_arr(i)) path_suff]);
    exp_var_arr(i)=m.exp_var;
    var(i)=m.var;
    matrix(i,:)=m.arr;
end
figure
hold on
bar(data_arr_2,exp_var_arr, 'r')
errorbar(exp_var_arr,var,'.','Color','black')
ylabel('Explained variance (%)')
xlabel('DNNS & Neural models')
ax=gca;
ax.YLim=[0 70];
saveas(gcf,'barplot.png');
close
lineplot(matrix);
end
function lineplot(matrix)
figure 
hold on
ze=matrix(end-1,:);
v4=matrix(end,:);
scatter(v4,ze,'filled')
ax=gca;

keyboard 

ax.YLim=[0 100];
ax.XLim=[0 100];
xlabel('V4 Cortex Multi-Unit Sample (Explained Variance, %)')
ylabel('Zeiler & Fergus 2013 (Explained Variance, %)')
lsline
saveas(gcf,'lineplot.png')
close
r=corrcoef(ze,v4);
r(1,2)
end
