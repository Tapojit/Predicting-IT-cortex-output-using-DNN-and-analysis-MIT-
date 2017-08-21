% The purpose of this function is to find a set of linear regression weights (betas) 
% that can predict a response value based on a set of predictor values. To find the beta values, a 
% cross-validation loop is used and 

i% s object does cross-validation over an input alpha array and generates beta based on the optimum alpha. 

classdef crossvalidation < handle
    
    properties
        cores
        cv_reps
        alpha_array
        loss
        min_val_error
        optimum_alpha
        beta_trained
    end
    
    properties (Access=private)
        cov
        resp
    end
    
    methods
    
        %Carries out crossvalidation using ridge regression. Requires at least first two arguments.
        function obj = crossvalidation(cov_data,resp_data,cores,alpha_array,loss,cv_reps)
            
            obj.cov=cov_data;
            obj.resp=resp_data;
            
            if nargin < 3
                obj = obj.init();
            else
                obj = obj.init();
                obj.cores = cores;
                obj.cv_reps = cv_reps;
                obj.alpha_array = alpha_array;
                obj.loss = loss;
            end
        
            alpha_loss_arr=zeros(size(obj.alpha_array,2),size(resp_data,2));
            
            %Starting parallel pooling
            parpool(obj.cores);
            
            %Iterating over alpha array to determine optimum alpha
            parfor i = 1:size(obj.alpha_array,2)
                
                loss_arr = zeros(cv_reps,size(resp_data,2));
                
                %Iterating over cross-validation range for individual alpha
                for a= 1:obj.cv_reps
                    reg=ridge_regression(cov_data,resp_data,obj.alpha_array(i),a);
                    resp_predicted=reg.predict();
                    true_resp=reg.resp_test;
                    if strcmp('RMSE',obj.loss)
                        loss_arr(a,:)=reg.rmse_loss(resp_predicted,true_resp);
                    else
                        loss_arr(a,:)=reg.mse_loss(resp_predicted,true_resp);
                    end
                end
                
                alpha_loss_arr(i,:)=mean(loss_arr);

            end
            
            
            [min_err,ind]=min(mean(alpha_loss_arr,2));
            obj.min_val_error=min_err;
            obj.optimum_alpha=obj.alpha_array(ind);
            delete(gcp);
            
            
        end
        
        %Initializing default values
        function obj=init(obj)
            obj.cores=1;
            obj.alpha_array=[1:10];
            obj.cv_reps=10;
            obj.loss='RMSE';
            
        end
        
        %Calculating betas over response matrix channels using optimum alpha 
        function obj=retrainer(obj,cov,resp)
            if nargin==1
                cov=obj.cov;
                resp=obj.resp;
            end
            reg=ridge_regression(cov,resp,obj.optimum_alpha,datasample([1:obj.cv_reps],1));
            obj.beta_trained=reg.beta_trained;
        end
            
    end
    
end
