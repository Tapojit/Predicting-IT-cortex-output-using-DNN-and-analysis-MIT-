function RDM=RDM_calc(type,model)
%Saves a heatmap of calculated RDM.

%Inputs:
%type=keyword representing data model type. They are: 'NM' and 'DNN'
%model=keyword representing model.

%If model is a DNN, two RDM heatmaps are saved. One calculated using DNN model features and the other calculated
%using IT model features predicted from same DNN features.


%Calculating 49x49 matrix
[a,b]=matrix_summer(type,model);
RDM=zeros(49,49);
for i=1:49
    for j=1:49
        r1=cov(a(i,:),a(j,:));
        r1=r1(1,end);
        r2=sqrt(var(a(i,:))*var(a(j,:)));
        RDM(i,j)=1-(r1/r2);
        
    end
end

if strcmp(model, 'Kr')
    title='Krizhevsky et al. 2012';
elseif strcmp(model, 'Ze')
    title='Zeiler & Fergus 2013';
else 
    title=model;
end
if ~isnan(b)
    RDM2=zeros(49,49);
    for k=1:49
        for l=1:49
            r1=cov(b(k,:),b(l,:));
            r1=r1(1,end);
            r2=sqrt(var(b(k,:))*var(b(l,:)));
            RDM2(k,l)=1-(r1/r2);

        end
    end
%     subplot(1,1,1);
    color_map2=jet;
    h2=heatmap(RDM2,'ColorLimits',[0 1],'Colormap',color_map2);
    h2.Title=title;
    h2.YLabel='Model Representations + IT-fit';
    saveas(gcf,[h2.Title '_' h2.YLabel '.png']);
    close;
end
% subplot(1,1,2);
color_map=jet;
h=heatmap(RDM,'ColorLimits',[0 1],'Colormap',color_map);
h.Title=title;
if strcmp(type,'DNN')
    h.YLabel='Model Representations';
else
    h.YLabel='Neural Representations';
end
saveas(gcf,[h.Title '_' h.YLabel '.png']);
close;
end
