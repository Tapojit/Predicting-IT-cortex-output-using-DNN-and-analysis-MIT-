%Determines how similar individual channels of response matrix are to respective channels predicted using covariate matrix.
%This is done by calculating explained explainable variance between individual response channels and predicted channels.


classdef reliability_test < handle
    properties
        beta_trained
        train_test_split
        
        resp_test
        reps
        cores
    end
    
    properties (Access=private)

        hyp_arr
        cov_data_shape
        resp_data_shape
    end
    
    methods
        %%%%Arguments - Require first two arguments or all%%%%%
        %cov_data = covariate data matrix
        %resp_data = response data matrix
        %hyp_arr = array of hyperparameter over which crossvalidation is carried out
        %reps = integer representing number of individual train-test splits carried out
        %train_test_split = float representing proportion of train data cases for each split
        %cores = cores used for parapooling
        %%%%%%Returns%%%%%%%
        %File with name "reliability.mat" which contains in a structure form:
        %%arr=array of explained explainable variance
        %%exp_var=mean explained explainable variance
        %%std_dev=standard deviation of 'arr'
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function obj=reliability_test(cov_data, resp_data, hyp_arr, reps,train_test_split, cores)
            obj = obj.init();
            obj.cov_data_shape = size(cov_data);
            obj.resp_data_shape = size(resp_data);
            
            %Shuffling covariate and response matrix
            rng(0);
            cov_data = cov_data(randperm(obj.cov_data_shape(1)), :);
            resp_data = resp_data(randperm(obj.resp_data_shape(1)), :);
            
            if nargin > 2 
                obj.reps = reps;
                obj.cores = cores;
                obj.train_test_split = train_test_split;
                obj.hyp_arr = hyp_arr;
            else
                hyp_arr = obj.hyp_arr;
                reps = obj.reps;
                train_test_split = obj.train_test_split;
            end
            
            betas = zeros(obj.cov_data_shape(2),1, obj.reps, obj.resp_data_shape(2));
            
            exp_var_arr = zeros(1, obj.resp_data_shape(2));
            
            std_dev_arr = zeros(1, obj.resp_data_shape(2));
            parpool(obj.cores);
            
            %Starting loop over response data channels
            parfor i = 1:obj.resp_data_shape(2)
                
                resp_vec = resp_data(:,i);
                resp_tr = zeros((train_test_split*size(cov_data,1)),1);
                resp_te = zeros(((1-train_test_split)*size(cov_data,1)),1);
                cov_tr = zeros((train_test_split*size(cov_data,1)),size(cov_data,2));
                cov_te = zeros(((1-train_test_split)*size(cov_data,1)),size(cov_data,2));
                
                exp_var = zeros(1, reps);
                %Starting loop over train-test reps
                for a = 1:reps
                    split_arr = [zeros(1, (size(cov_data,1)*train_test_split)) ones(1, (cobj.ov_data_shape(1)*(1-train_test_split)))];
                    
                    %Generating deterministically training data and test data for current split
                    rng(i);
                    split_arr = split_arr(randperm(numel(split_arr)));
                    
                    train=1;
                    test=1;
                    for m = 1:size(split_arr,2)
                        if split_arr(m) == 0
                            cov_tr(train,:) = cov_data(m,:);
                            resp_tr(train,:) = resp_vec(m,:);
                            train = train + 1;
                        else
                            cov_te(test,:) = cov_data(m,:);
                            resp_te(test,:) = resp_vec(m,:);
                            test = test + 1;
                        end
                    end
                    
                    %Carrying out regression
                    reg = k_fold_cv(cov_tr, resp_tr, hyp_arr, 3, reg_ridge, 'RMSE');
                    
                    %Calculating and storing betas
                    reg_2 = reg_ridge.fit(cov_tr, resp_tr, reg.opt_hyp);
                    betas(:,:,a,i)=reg_2.beta;
                    %Making predictions
                    pred_resp = reg_2.predict(cov_te);
                    
                    split_half = ceil(size(pred_resp,1)/2);
                    
                    r1 = corr(pred_resp(1:split_half,:), resp_te(1:split_half,:), 'type', 'Pearson');
                    r2 = corr(pred_resp((split_half+1:end),:), resp_te((split_half+1:end),:), 'type', 'Pearson');
                    
                    mean_r = (r1 + r2)/2;
                    
                    %Calculating Spearman split-half correlation
                    sp_corr = (2*mean_r)/(1+mean_r);
                    %Explained explainable variance for current split/rep
                    exp_var(a) = sp_corr * 100;
                    disp(['Rep ' num2str(a) ' of ' num2str(reps)])
                end
                %Storing median explained explainable variance over all splits/reps for current channel
                exp_var_arr(i) = median(exp_var);
                %Storing standard deviation of explained explainable variance over all splits/reps for current channel
                std_dev_arr(i) = std(exp_var);
                disp(['Channel ' num2str(i) ' of ' num2str(obj.resp_data_shape(2)) ' done'])
                
            end
            
            %Array of explained explainable variance
            data.arr = exp_var_arr;
            %Mean explained explainable variance over all response matrix channels
            data.exp_var = mean(exp_var_arr);
            %Mean standard deviation over all response matrix channels
            data.std_dev = mean(std_dev_arr);
            %Name of .mat file where data will be saved
            filepath = ['reliability.mat'];
            save(filepath,'-struct','data');
            
            
        end
        %Initializes variables and default values
        function obj = init(obj)
            obj.cores = 1;
            obj.reps = 10;
            obj.train_test_split = 0.8;
            obj.hyp_arr = [1:10];
            obj.beta_trained = zeros(obj.cov_data_shape(2),1, obj.reps, obj.resp_data_shape(2));
        end
        
    end
    
end
