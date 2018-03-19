%Carries out k-fold-cross-validation using given covariate matrix and response vector

classdef k_fold_cv < handle
    
    properties
        opt_hyp
        min_loss
        beta_trained
    end
    
    properties (Access = private)
        hyp_arr
        folds
        cov_matrix_shape
        loss
    end
    
    methods
        %%%%%Arguments - Requires first two arguments or all%%%%%%%%
        %cov_matrix=covariate data matrix
        %resp_vec=response vetor
        %hyp_arr=array of hyperparameters over which crossvalidation is carried out
        %folds=number of folds in crossvalidation
        %reg_alg=Object representing algorithm used for regression. Eg: 'reg_ridge' represents ridge regression
        %loss=String representing loss function. Eg: 'RMSE' represents Root mean squared error. 'MSE' represents Mean squared error.
        %%%%%Returns%%%%%%%%
        %object
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function obj = k_fold_cv(cov_matrix, resp_vec, reg_alg, hyp_arr, folds, loss)
            obj = obj.init();
            obj.cov_matrix_shape = size(cov_matrix);
            if nargin > 3 
                obj.hyp_arr = hyp_arr;
                obj.folds = folds;
                obj.loss = loss;
            else
                folds=obj.folds;
            end
            
            fold_shape_arr=zeros(1,folds);
            data_case_no = size(cov_matrix, 1);
            
            %Calculating proportion of data cases for each fold
            for i = 1:folds
                if i == folds
                    fold_shape_arr(i) = data_case_no - sum(fold_shape_arr(1:folds-1));
                else
                    fold_shape_arr(i) = floor(data_case_no/folds);
                end
            end
            
            
            
            cv_hyp_loss = zeros(1, size(obj.hyp_arr, 2));
            %Looping over hyperparameter array to determine optimum hyperparameter value
            for m = 1:size(obj.hyp_arr, 2)
                hyp_loss_arr = zeros(1, folds);
                %Looping over folds
                for a = 1:folds
                    if a == 1
                        cv_cov_tr = cov_matrix((fold_shape_arr(a)+1:end),:);
                        cv_resp_tr = resp_vec((fold_shape_arr(a)+1:end),:);
                        cv_cov_val = cov_matrix((1:fold_shape_arr(a)),:);
                        cv_resp_val = resp_vec((1:fold_shape_arr(a)),:);
                    elseif a == folds
                        cv_cov_tr = cov_matrix((1:sum(fold_shape_arr(1:end-1))),:);
                        cv_resp_tr = resp_vec((1:sum(fold_shape_arr(1:end-1))),:);
                        cv_cov_val = cov_matrix((sum(fold_shape_arr(1:end-1))+1:end),:);
                        cv_resp_val = resp_vec((sum(fold_shape_arr(1:end-1))+1:end),:);
                    else
                        part_1_sum = sum(fold_shape_arr(1:a-1));
                        part_2_sum = sum(fold_shape_arr(1:a));
                        cv_cov_tr = [cov_matrix((1:part_1_sum),:); cov_matrix((part_2_sum+1:end),:)];
                        cv_resp_tr = [resp_vec((1:part_1_sum),:); resp_vec((part_2_sum+1:end),:)];
                        cv_cov_val = cov_matrix((part_1_sum+1:part_2_sum),:);
                        cv_resp_val = resp_vec((part_1_sum+1:part_2_sum),:);
                    end
                    %Validation beta
                    beta = reg_alg(cv_cov_tr, cv_resp_tr, obj.hyp_arr(m));
                    pred_vector = mtimes(cv_cov_val, beta);

                    %%%Calculating validation loss%%%%
                    
                    %For RMSE loss:
                    if strcmp(obj.loss, 'RMSE')
                        loss_obj = RMSE_LM;
                        hyp_loss_arr(a) = loss_obj.calculate_loss(pred_vector, cv_resp_val);
                    %For MSE loss:
                    else
                        loss_obj = MSE_LM;
                        hyp_loss_arr(a) = loss_obj.calcualte_loss(pred_vector, cv_resp_val);

                    end

                end
                %Calculating and storing mean validation error over folds for current hyperparameter value
                cv_hyp_loss(m) = mean(hyp_loss_arr);

            end
            
            %Determining minimum mean validation error and index of optimum hyperparameter  
            [err_min, indx] = min(cv_hyp_loss);
            
            %Optimum hyperparameter
            obj.opt_hyp = obj.hyp_arr(indx);
            
            %Smallest mean validation loss
            obj.min_loss = err_min;
            
            %Beta trained using optimum hyperparameter
            obj.beta_trained = reg_alg(cov_matrix, resp_vec, obj.opt_hyp);
                        
        end
        
        %Initializing default values
        function obj = init(obj)
            
            obj.hyp_arr = [1:10];
            obj.folds = 3;
            obj.loss = 'RMSE';
            
        end
        
        


    end
    
end
