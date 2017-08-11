function IT_multi(model,cores)
%Carries out ridge regression to predict individual channels of IT multiunit
%arrays using given Neural data/DNN final layer output matrix as features.

%Returns a file with name corresponding to feature matrix used followed by '_model.mat'. Contains values:
%Explained explainable variance between corresponding predicted channel values and actual channel values. 
%Mean explained explainable variance
%Explained variance in results obtained

%Inputs:
%model=keyword representing one of 7 data matrices; 'HMO', 'HMAX', 'V4', 'V1', 'V2', 'Kr', 'Ze'
%cores=number of cores to be made available for parapooling.


%IT data path
IT_data_path='PLoSCB2014_data_20141216/PLoSCB2014_data_20141216/NeuralData_IT_multiunits';
IT_data=load(IT_data_path);
IT_data_features=IT_data.features;

IT_data_cases=size(IT_data_features);
exp_var_arr=zeros(1,IT_data_cases(2));
std_dev_arr=zeros(1,IT_data_cases(2));

%Starting parallel pooling
parpool(cores);
%looping 168 times to predict individual IT channels
parfor f=1:IT_data_cases(2)
    

    meta_tr_test=cell(IT_data_cases(1),10);
    for i=1:IT_data_cases(1)
        str=IT_data.meta(i,:);
        str=strsplit(str,' ');
        meta_tr_test(i,:)=str(3:12);
    %     disp([num2str(i) ' of ' '1960']);
    end


    test_rmse=zeros(1,10);
    exp_var=zeros(1,10);
   
    for m=1:10
        IT_train_lab=zeros(1,0.8*IT_data_cases(1));
        IT_test_lab=zeros(1,0.2*IT_data_cases(1));
        a=1;
        b=1;
        for i=1:IT_data_cases(1)
            if cell2mat(meta_tr_test(i,m))=='1'
                IT_train_lab(a)=i;
                a=a+1;
            else
                IT_test_lab(b)=i;
                b=b+1;
            end
        end

        rng(0);
        IT_train=IT_data_features(IT_train_lab,f);
        IT_train=IT_train(randperm(numel(IT_train(:,1))),:);
        IT_test=IT_data_features(IT_test_lab,f);
        tr_path='PLoSCB2014_models_20150218/20150218/Models_';
        if strcmp(model,'HMAX')
            tr_path=[tr_path 'HMAX.mat'];
        elseif strcmp(model,'HMO')
            tr_path=[tr_path 'HMO.mat'];
        elseif strcmp(model,'Kr')
            tr_path=[tr_path 'Krizhevsky2012.mat'];
        elseif strcmp(model,'V1')
            tr_path=[tr_path 'V1like.mat'];
        elseif strcmp(model,'V2')
            tr_path=[tr_path 'V2like.mat'];
        elseif strcmp(model,'Ze')
            tr_path=[tr_path 'ZeilerFergus2013.mat'];
        elseif strcmp(model,'V4')
            tr_path='PLoSCB2014_data_20141216/PLoSCB2014_data_20141216/NeuralData_V4_multiunits';
        end

        feature=load(tr_path);
        feature_data=feature.features;
        rng(0);

        feature_data_train=feature_data(IT_train_lab,:);
        feature_data_train=feature_data_train(randperm(numel(feature_data_train(:,1))),:);
        feature_data_test=feature_data(IT_test_lab,:);


        train_subsam_size=ceil(size(IT_train)/3);
        train_subsam_size=train_subsam_size(1);
        val_error=zeros(1,10);
        %10 train test splits for individual channels
        for a=1:10;
            rmse_val=zeros(1,3);


            % 3-fold cross validation
            for i=1:3
                
                
                if i==1
                    training_sub_f=feature_data_train(1:train_subsam_size*2,:);
                    training_sub_l=IT_train(1:train_subsam_size*2);
                    val_sub_f=feature_data_train((train_subsam_size*2)+1:end,:);
                    val_sub_l=IT_train((train_subsam_size*2)+1:end);
                elseif i==2
                    training_sub_f=feature_data_train([1:train_subsam_size,(train_subsam_size*2)+1:end],:);
                    training_sub_l=IT_train([1:train_subsam_size,(train_subsam_size*2)+1:end]);
                    val_sub_f=feature_data_train(train_subsam_size+1:(train_subsam_size*2),:);
                    val_sub_l=IT_train(train_subsam_size+1:(train_subsam_size*2));
                else
                    training_sub_f=feature_data_train(train_subsam_size+1:end,:);
                    training_sub_l=IT_train(train_subsam_size+1:end);
                    val_sub_f=feature_data_train(1:train_subsam_size,:);
                    val_sub_l=IT_train(1:train_subsam_size);
                end
                %calculating encoding model
                b=ridge_r(training_sub_l,training_sub_f, a);
                p_hat=mtimes(val_sub_f,b);
                rmse_val(i)=sqrt(immse(p_hat,val_sub_l));


            end
            val_error(a)=mean(rmse_val);

        end


        [err_min,ind]=min(val_error);
        %Calculating encoding model
        b2=ridge_r(IT_train,feature_data_train,ind);
        %making predictions on test features
        p_hat_te=mtimes(feature_data_test,b2);
        test_rmse(m)=sqrt(immse(p_hat_te,IT_test));
        split_size=size(feature_data_test)/2;
        split_size=split_size(1);
        c1=gather(corr(p_hat_te(1:split_size),IT_test(1:split_size),'type','Pearson'));
        c2=gather(corr(p_hat_te(split_size+1:end),IT_test(split_size+1:end),'type','Pearson'));
        mean_r=(c1+c2)/2;
        %calculating spearman split-half correlation
        sp_br=(2*mean_r)/(1+mean_r);
%         corr_exp=corr(p_hat_te,IT_test,'type','Pearson');
        %Explained explainable variance
        exp_var(m)=sp_br*100;
        disp(['lap ' num2str(m) ' of 10 complete' ])
    end
    disp(['unit ' num2str(f) ' of ' num2str(IT_data_cases(2)) ' done'])
    %Taking median explained explainable variance from ten train-test split
    exp_var_arr(f)=median(exp_var);
    %Taking standard deviation from ten train-test splits
    std_dev_arr(f)=std(exp_var);
    disp(exp_var_arr(f))
end

IT.arr=exp_var_arr;
%Mean explained explainable variance from 168 IT channels
IT.exp_var=mean(exp_var_arr);
%Mean standard deviation over 168 channels
IT.std_dev=mean(std_dev_arr);
filepath=[model '_model' '.mat'];
save(filepath,'-struct','IT');
end


