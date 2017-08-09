function [ IT_matrix_pred ] = IT_predictor( model )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
IT_data_path='PLoSCB2014_data_20141216/PLoSCB2014_data_20141216/NeuralData_IT_multiunits';
IT_data=load(IT_data_path);
IT_data_features=IT_data.features;

model_path='PLoSCB2014_models_20150218/20150218/Models_';
if strcmp(model,'HMO')
    model_path=[model_path 'HMO.mat'];
elseif strcmp(model,'Kr')
    model_path=[model_path 'Krizhevsky2012.mat'];
elseif strcmp(model,'Ze')
    model_path=[model_path 'ZeilerFergus2013.mat'];
elseif strcmp(model,'V1')
    model_path=[model_path 'V1like.mat'];
elseif strcmp(model,'V2')
    model_path=[model_path 'V2like.mat'];
end
rng(0);
model_load=load(model_path);
model_f=model_load.features;
model_f_shuff=model_f(randperm(size(model_f,1)),:);
IT_shuff=IT_data_features(randperm(size(IT_data_features,1)),:);

IT_matrix_pred=zeros(size(IT_data_features));
for m=1:size(IT_data_features,2)
    train_subsam_size=ceil(size(IT_shuff,1)/3);
    val_r_sq=zeros(1,10);
    for a=1:10
        r_sq_val=zeros(1,3);


        % 3-fold cross validation
        for i=1:3


            if i==1
                training_sub_f=model_f_shuff(1:train_subsam_size*2,:);
                training_sub_l=IT_shuff((1:train_subsam_size*2),m);
                val_sub_f=model_f_shuff((train_subsam_size*2)+1:end,:);
                val_sub_l=IT_shuff((train_subsam_size*2)+1:end,m);
            elseif i==2
                training_sub_f=model_f_shuff([1:train_subsam_size,(train_subsam_size*2)+1:end],:);
                training_sub_l=IT_shuff([1:train_subsam_size,(train_subsam_size*2)+1:end],m);
                val_sub_f=model_f_shuff(train_subsam_size+1:(train_subsam_size*2),:);
                val_sub_l=IT_shuff(train_subsam_size+1:(train_subsam_size*2),m);
            else
                training_sub_f=model_f_shuff(train_subsam_size+1:end,:);
                training_sub_l=IT_shuff(train_subsam_size+1:end,m);
                val_sub_f=model_f_shuff(1:train_subsam_size,:);
                val_sub_l=IT_shuff(1:train_subsam_size,m);
            end
            b=ridge_r(training_sub_l,training_sub_f,'alpha', a);

            p_hat=mtimes(val_sub_f,b);
            r_sq_val(i)=100*(corr(p_hat,val_sub_l,'type','Pearson'))^2;


        end
        val_r_sq(a)=median(r_sq_val);

    end
    [err_min,ind]=max(val_r_sq);

    b2=ridge_r(IT_data_features(:,m),model_f,'alpha',ind);
    IT_matrix_pred(:,m)=mtimes(model_f,b2);
    disp(['lap ' num2str(m) ' of ' num2str(size(IT_data_features,2)) ' complete' ])

    
    
    
end
filepath=[model '_IT.mat'];
save(filepath,'IT_matrix_pred');

end

