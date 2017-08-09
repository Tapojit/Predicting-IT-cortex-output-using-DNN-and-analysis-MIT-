function [s,s2] = matrix_summer( pathtype,model )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if strcmp(pathtype,'NM')
    path='/cbcl/cbcl01/tdebnath/Work/Expt-cadieu-replicate/PLoSCB2014_data_20141216/PLoSCB2014_data_20141216/';
else
    path='/cbcl/cbcl01/tdebnath/Work/Expt-cadieu-replicate/PLoSCB2014_models_20150218/20150218/';
end

if strcmp(model,'IT')
    path=[path 'NeuralData_IT_multiunits.mat'];
elseif strcmp(model,'V4')
    path=[path 'NeuralData_V4_multiunits.mat'];
elseif strcmp(model,'V1')
    path=[path 'Models_V1like.mat'];
elseif strcmp(model,'V2')
    path=[path 'Models_V2like.mat'];
elseif strcmp(model,'HMO')
    path=[path 'Models_HMO.mat'];
    if exist('HMO_IT.mat','file')==2
        path2='HMO_IT.mat';
    else
        IT_predictor('HMO');
        path2='HMO_IT.mat';
    end
elseif strcmp(model,'HMAX')
    path=[path 'Models_HMAX.mat'];
elseif strcmp(model,'Kr')
    path=[path 'Models_Krizhevsky2012.mat'];
    if exist('Kr_IT.mat','file')==2
        path2='Kr_IT.mat';
    else
        IT_predictor('Kr');
        path2='Kr_IT.mat';
    end
elseif strcmp(model,'Ze')
    path=[path 'Models_ZeilerFergus2013.mat'];
    if exist('Ze_IT.mat','file')==2
        path2='Ze_IT.mat';
    else
        IT_predictor('HMO');
        path2='Ze_IT.mat';
    end
end

data=load(path);
matrix=data.features;
meta=data.meta;
matrix2=zeros(size(matrix));
features={'Animals','Cars','Chairs','Faces','Fruits','Planes','Tables'};
a=1;
if strcmp(model,'Ze') || strcmp(model,'HMO') || strcmp(model,'Kr')
    data2=load(path2);
    matrix3=data2.IT_matrix_pred;
    matrix4=zeros(size(matrix3));
    b=1;
    for n=features
        for j=1:1960
            if strcmp(meta(j,(53:size(cell2mat(n),2)+52)),cell2mat(n))
                matrix4(b,:)=matrix3(j,:);
                b=b+1;
            end
        end
    end
    s2=zeros(49,size(matrix3,2));
    for k=1:49
        s2(k,:)=sum(matrix4((((k-1)*40)+1:k*40),:));
    end
else
    s2=NaN;
end
for m=features
    
    for i=1:1960
        if strcmp(meta(i,(53:size(cell2mat(m),2)+52)),cell2mat(m))
            matrix2(a,:)=matrix(i,:);
            a=a+1;
        end
    end
end
s=zeros(49,size(matrix,2));
for i=1:49
    s(i,:)=sum(matrix2((((i-1)*40)+1:i*40),:));
end


end

