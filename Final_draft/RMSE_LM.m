%Calculates Root Mean Squared Error loss
classdef RMSE_LM < handle
        
    methods
        
        %RMSE (Root Mean Squared Error) loss function
        %%%%%%Arguments%%%%%%
        %obj=object
        %predicted_vector=predicted response vector
        %actual_vector=actual response vector
        
        %%%%%%%Returns%%%%%%%%
        %loss=RMSE loss value
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function loss = calculate_loss(obj, predicted_vector, actual_vector)
            
            loss = sqrt(mean((predicted_vector - actual_vector).^2)); 
        
        end
    end
    
end