%Ridge regression object
classdef reg_ridge < handle
    properties
        beta
    end
    
    methods
        
        %Carries out ridge regression and returns decision rule (beta)
        %%%%%%%Arguments%%%%%
        %obj=object
        %resp_vec=response vector
        %cov_matrix=covariate matrix
        %alpha=hyperparameter value
        %%%%%Returns%%%%%
        
        %obj.beta=decision rule
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function obj = fit(obj, cov_matrix, resp_vec, alpha)
            x_f = size(cov_matrix, 2);

            G = (alpha^2)*eye(x_f);

            G(1,1) = 0;

            obj.beta = mtimes(inv(mtimes(cov_matrix.',cov_matrix)+G),mtimes(cov_matrix.',resp_vec));

        end
        
        
        %Makes predictions on test covariates. Returns predicted response
        %vector.
        %%%%%%Arguments%%%%%%
        %obj=object
        %test_features=Test covariate matrix
        
        %%%%%%%Returns%%%%%%%%
        %predicted_figures=predicted response vector
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function predicted_figures=predict(obj, test_features)
            
            predicted_figures = mtimes(test_features, obj.beta);

        end
        
        %RMSE (Root Mean Squared Error) loss function
        %%%%%%Arguments%%%%%%
        %obj=object
        %predicted=predicted response vector
        %actual=actual response vector
        
        %%%%%%%Returns%%%%%%%%
        %loss=RMSE loss value
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function loss = rmse_loss(obj, predicted, actual)
            
            loss = sqrt(mean((predicted-actual).^2));
        
        end
        
        %MSE (Mean Squared Error) loss function
        %%%%%%Arguments%%%%%%
        %obj=object
        %predicted=predicted response vector
        %actual=actual response vector
        
        %%%%%%%Returns%%%%%%%%
        %loss=MSE loss value
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function loss = mse_loss(obj, predicted, actual)
            
            loss = mean((predicted-actual).^2);
            
        end
    end
end