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
        function obj = crossvalidation(cov_data,resp_data,cores,alpha_array,loss,cv_reps)
            obj.cov=cov_data;
            obj.resp=resp_data;
            if nargin<3
                obj=obj.init();
            else
                obj=obj.init();
                obj.cores=cores;
                obj.cv_reps=cv_reps;
                obj.alpha_array=alpha_array;
                obj.loss=loss;
            end
            alpha_loss_arr=zeros(1,size(obj.alpha_array));
            parpool(obj.cores);
            parfor i=1:size(obj.alpha_array)
                loss_arr=zeros(1,cv_reps);
                for a=1:obj.cv_reps
                    reg=ridge_regression(cov_data,resp_data,obj.alpha_array(i),a);
                    resp_predicted=reg.predict();
                    true_resp=reg.resp_test;
                    if strcmp('RMSE',obj.loss)
                        loss_arr(a)=rmse_loss(resp_predicted,true_resp);
                    else
                        loss_arr(a)=mse_loss(resp_predicted,true_resp);
                    end
                end
                alpha_loss_arr(i)=mean(loss_arr);

            end
            [min_err,ind]=min(alpha_loss_arr);
            obj.min_val_error=min_err;
            obj.optimum_alpha=obj.alpha_array(ind);
            
            
        end
        function obj=init(obj)
            obj.cores=1;
            obj.alpha_array=[1:10];
            obj.cv_reps=10;
            obj.loss='RMSE';
            
        end
        function loss=rmse_loss(predicted,actual)
            loss=zeros(1,size(predicted,2));
            for i=1:size(predicted,2)
                loss(i)=sqrt(mean((predicted(:,i)-actual(:,i)).^2));
            end
        end
        
        function loss=mse_loss(predicted,actual)
            loss=zeros(1,size(predicted,2));
            for i=1:size(predicted,2)
                loss(i)=mean((predicted(:,i)-actual(:,i)).^2);
            end
        end
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