classdef Ridge_RG < handle

    properties
        
        beta
        
    end
    
    methods
        

        
        function obj = train(obj, matrix_covariate, vector_response, cross_validation)
            
            %Ridge regression equation. Will be used as an anonymous function
            %%%%%%%Arguments%%%%%
            %vector_response=response vector
            %matrix_covariate=covariate matrix
            %alpha=hyperparameter value
            %%%%%Returns%%%%%

            %beta_vector=decision rule
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            function beta_vector = training_function(matrix_covariate, vector_response, alpha)

                 x_f = size(matrix_covariate, 2);
 
                 G = (alpha^2)*eye(x_f);
 
                 G(1,1) = 0;
 
                 beta_vector = mtimes(inv(mtimes(matrix_covariate.',matrix_covariate)+G),mtimes(matrix_covariate.',vector_response));

            end
            
            if nargin < 4
                obj.beta = training_function(matrix_covariate, vector_response, randi([1 10]));
                
            elseif cross_validation == true
                validated = k_fold_cv(matrix_covariate, vector_response, @training_function);
                obj.beta = validated.beta_trained;
                
            end
            
        end
        
        %Makes predictions on test covariates. Returns predicted response
        %vector.
        %%%%%%Arguments%%%%%%
        %obj=object
        %matrix_covariate=Test covariate matrix
        
        %%%%%%%Returns%%%%%%%%
        %predicted_figures=predicted response vector
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function predicted_vector = test(obj, matrix_covariate)
            
            predicted_vector = mtimes(matrix_covariate, obj.beta);
            
        end
        
        

        
    end

end
