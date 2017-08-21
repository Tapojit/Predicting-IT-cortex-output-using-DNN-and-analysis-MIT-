classdef k_fold_cv < handle
    
    properties
        reg_alg
        betas_trained
    end
    
    properties (Access = private)
        hyp_arr
        folds
        cov_matrix_shape
        resp_matrix_shape
        loss
    end
    
    methods
        
        function obj = k_fold_cv(cov_matrix, resp_matrix, hyp_arr, folds, reg_alg, loss)
            obj = obj.init();
            
            obj.cov_matrix_shape = size(cov_matrix);
            obj.resp_matrix_shape = size(resp_matrix);
            if nargin > 2 
                obj.hyp_arr = hyp_arr;
                obj.folds = folds;
                obj.reg_alg = reg_alg;
                obj.loss = loss;
            end
            
            fold_shape_arr=zeros(1,folds);
            data_case_no = size(cov_matrix, 1);
            
            for i = 1:folds
                if i == folds
                    fold_shape_arr(i) = data_case_no - sum(fold_shape_arr(1:folds-1));
                else
                    fold_shape_arr(i) = floor(data_case_no/folds);
                end
            end
            
            for i = 1:obj.resp_matrix_shape(2)
                labels = resp_matrix(:,i);
                for m = 1:size(obj.hyp_arr, 2)
                    for a = 1:folds
                        if a == 1
                            cv_cov_tr = cov_matrix((fold_shape_arr(a)+1:end),:);
                            cv_resp_tr = labels((fold_shape_arr(a)+1:end),:);
                            cv_cov_val = cov_matrix((1:fold_shape_arr(a)),:);
                            cv_resp_val = labels((1:fold_shape_arr(a)),:);
                        elseif a == fold
                            cv_cov_tr = cov_matrix((1:sum(fold_shape_arr(1:end-1))),:);
                            cv_resp_tr = labels((1:sum(fold_shape_arr(1:end-1))),:);
                            cv_cov_val = cov_matrix((sum(fold_shape_arr(1:end-1))+1:end),:);
                            cv_resp_val = labels((sum(fold_shape_arr(1:end-1))+1:end),:);
                        else
                            part_1_sum = sum(fold_shape_arr(1:a-1));
                            part_2_sum = sum(fold_shape_arr(1:a));
                            cv_cov_tr = [cov_matrix((1:part_1_sum),:) cov_matrix((part_2_sum+1:end),:)];
                            cv_resp_tr = [labels((1:part_1_sum),:) labels((part_2_sum+1:end),:)];
                            cv_cov_val = cov_matrix((part_1_sum+1:part_2_sum),:);
                            cv_resp_val = labels((part_1_sum+1:part_2_sum),:);
                        end

                        if strcmp(obj.reg_alg, 'Ridge_reg')
                           
                            beta = ridge_reg(cv_resp_tr, cv_cov_tr, obj.hyp_arr(m));
                            
                        end
                        if strcmp(obj.loss, 'RMSE')
                            
                        end

                    end
                end
            end
            
            
        end
        
        function obj = init(obj)
            obj.hyp_arr = [1:10];
            obj.folds = 3;
            obj.reg_alg = 'Ridge_reg';
            obj.loss = 'RMSE';
            obj.betas_trained = zeros(obj.cov_matrix_shape(2), 1, obj.resp_matrix_shape(2));
            
        end
        
        %Carries out ridge regression and returns decision rule (beta)
        function beta = ridge_reg( resp_matrix,cov_matrix,alpha )

            x_f = size(cov_matrix, 2);

            G = (alpha^2)*eye(x_f);

            G(1,1) = 0;

            beta = mtimes(inv(mtimes(cov_matrix.',cov_matrix)+G),mtimes(cov_matrix.',resp_matrix));
        end
        
        %RMSE (Root Mean Squared Error) loss function
        function loss=rmse_loss(obj,predicted,actual)
            loss=zeros(1,size(predicted,2));
            for i=1:size(predicted,2)
                loss(i)=sqrt(mean((predicted(:,i)-actual(:,i)).^2));
            end
        end
        %MSE (Mean Squared Error) loss function
        function loss=mse_loss(obj,predicted,actual)
            loss=zeros(1,size(predicted,2));
            for i=1:size(predicted,2)
                loss(i)=mean((predicted(:,i)-actual(:,i)).^2);
            end
        end
        
        %Makes predictions on test covariates. Returns predicted response
        %matrix.
        function predicted_figures=predict(obj,test_features)
            if nargin~=2
                test_features=obj.cov_test;
            end
            predicted_figures=zeros(size(test_features,1),obj.resp_data_shape(2));
            for i=1:obj.resp_data_shape(2)
                predicted_figures(:,i)=mtimes(test_features,obj.beta_trained(:,:,i));
            end
        end


    end
    
end