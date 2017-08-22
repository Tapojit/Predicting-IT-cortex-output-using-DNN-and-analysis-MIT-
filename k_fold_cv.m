classdef k_fold_cv < handle
    
    properties
        reg_alg
        betas_trained
        opt_hyp
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
            else
                folds=obj.folds;
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
            
            best_hyp_indx = zeros(1, obj.resp_matrix_shape(2));
            
            for i = 1:obj.resp_matrix_shape(2)
                labels = resp_matrix(:,i);
                cv_hyp_loss = zeros(1, size(obj.hyp_arr, 2));
                
                for m = 1:size(obj.hyp_arr, 2)
                    hyp_loss_arr = zeros(1, folds);
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
                            pred_vector = predict(beta, cv_cov_val);
                            
                        end
                        if strcmp(obj.loss, 'RMSE')
                            
                            hyp_loss_arr(a) = rmse_loss(pred_vector, cv_resp_val);
                        else
                            
                            hyp_loss_arr(a) = mse_loss(pred_vector, cv_resp_val);
                            
                        end

                    end
                    
                    cv_hyp_loss(m) = mean(hyp_loss_arr);
                    
                end
                
                [err_min, indx] = min(cv_hyp_loss);
                
                best_hyp_indx(i) = indx;
                
            end
            
            obj.opt_hyp = obj.hyp_arr(mode(best_hyp_indx));
            
            for i = 1:obj.resp_matrix_shape(2)
                if strcmp(obj.reg_alg, 'Ridge_reg')
                    obj.betas_trained(:,:,i) = ridge_reg(resp_matrix(:,i), cov_matrix, obj.opt_hyp);
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
        function beta = ridge_reg( resp_vector, cov_matrix, alpha )

            x_f = size(cov_matrix, 2);

            G = (alpha^2)*eye(x_f);

            G(1,1) = 0;

            beta = mtimes(inv(mtimes(cov_matrix.',cov_matrix)+G),mtimes(cov_matrix.',resp_vector));
        
        end
        
        %RMSE (Root Mean Squared Error) loss function
        function loss = rmse_loss(obj, predicted, actual)
            
            loss = sqrt(mean((predicted-actual).^2));
        
        end
        
        %MSE (Mean Squared Error) loss function
        function loss = mse_loss(obj, predicted, actual)
            
            loss = mean((predicted-actual).^2);
            
        end
        
        %Makes predictions on test covariates. Returns predicted response
        %vector.
        function predicted_figures=predict(obj, beta, test_features)
            
            predicted_figures = mtimes(test_features, beta);

        end


    end
    
end