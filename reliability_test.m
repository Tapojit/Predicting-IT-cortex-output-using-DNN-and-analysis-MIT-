classdef reliability_test < handle
    properties
        beta_trained
        train_test_split
        cov_data_shape
        resp_data_shape
        resp_test
        reps
    end
    
    properties (Access=private)
        cov_train
        cov_test
        resp_train
        
    end
    
    methods
        %Takes in as input-covariate data matrix, response data matrix,
        %alpha and seed. Works if seed argument not provided
        function obj=reliability_test(cov_data, resp_data,alpha_arr, reps,train_test_split)
            obj.cov_data_shape = size(cov_data);
            obj.resp_data_shape = size(resp_data);
            
            if nargin < 3
                obj.cov_data_shape = size(cov_data);
                obj.resp_data_shape = size(resp_data);
                obj = obj.init();
                obj.cov_data_shape=size(cov_data);

            end
            train=1;
            test=1;
            for i=1:size(obj.train_test_split,2)
                if obj.train_test_split(i)==1
                    obj.cov_train(train,:)=cov_data(i,:);
                    obj.resp_train(train,:)=resp_data(i,:);
                    train=train+1;
                else
                    obj.cov_test(test,:)=cov_data(i,:);
                    obj.resp_test(test,:)=resp_data(i,:);
                    test=test+1;
                end
            end
            %Carrying out ridge regression.
            for m=1:obj.resp_data_shape(2)
                obj.beta_trained(:,:,m)=ridge_r(obj.resp_train(:,m),obj.cov_train,alpha);
            end
            disp('training done')
            
            
        end
        %Initializes variables and default values
        function obj = init(obj)
            obj.reps = 10;
            obj.train_test_split = 0.8;
            obj.cov_train=zeros((1960*0.8),obj.cov_data_shape(2));
            obj.cov_test=zeros((1960*0.2),obj.cov_data_shape(2));
            obj.resp_train=zeros((1960*0.8),obj.resp_data_shape(2));
            obj.resp_test=zeros((1960*0.2),obj.resp_data_shape(2));
            obj.beta_trained=zeros(obj.cov_data_shape(2),1,obj.resp_data_shape(2));
        end
        
    end
    
end