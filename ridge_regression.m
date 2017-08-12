classdef ridge_regression < handle
    properties
        beta_trained
        train_test_split
        train_test_seed
        cov_data_shape
        resp_data_shape
        resp_test
    end
    properties (Access=private)
        cov_train
        cov_test
        resp_train
        
    end
    methods
        function obj=ridge_regression(cov_data, resp_data,alpha, seed)
            obj.cov_data_shape=size(cov_data);
            obj.resp_data_shape=size(resp_data);
            if nargin<4
                obj.cov_data_shape=size(cov_data);
                obj.resp_data_shape=size(resp_data);
                obj.train_test_seed=1;
                obj=obj.init();
                obj.cov_data_shape=size(cov_data);
            else
                obj.train_test_seed=seed;
                obj=obj.init();
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
            for m=1:obj.resp_data_shape(2)
                obj.beta_trained(:,:,m)=ridge_r(obj.resp_train(:,m),obj.cov_train,alpha);
            end
            disp('training done')
            
            
        end
        function obj=init(obj)
            rng(obj.train_test_seed);
            arr=[zeros(1,(1960*0.2)),ones(1,(1960*0.8))];
            obj.train_test_split=arr(randperm(numel(arr)));
            obj.cov_train=zeros((1960*0.8),obj.cov_data_shape(2));
            obj.cov_test=zeros((1960*0.2),obj.cov_data_shape(2));
            obj.resp_train=zeros((1960*0.8),obj.resp_data_shape(2));
            obj.resp_test=zeros((1960*0.2),obj.resp_data_shape(2));
            obj.beta_trained=zeros(obj.cov_data_shape(2),1,obj.resp_data_shape(2));
        end
        function predicted_figures=predict(obj,test_features)
            if nargin~=2
                test_features=obj.cov_test;
            end
            predicted_figures=zeros((1960*0.2),obj.resp_data_shape(2));
            for i=1:obj.resp_data_shape(2)
                predicted_figures(:,i)=mtimes(test_features,obj.beta_trained(:,:,i));
            end
        end
        
    end
    
end