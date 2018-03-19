%Calculates Mean Squared Error loss
classdef MSE_LM < handle
        
    methods
        
        %%%%%%Arguments%%%%%%
        %obj=object
        %predicted_vector=predicted response vector
        %actual_vector=actual response vector
        
        %%%%%%%Returns%%%%%%%%
        %loss=MSE loss value
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function loss = calculate_loss(obj, predicted_vector, actual_vector)
            
            loss = mean((predicted_vector - actual_vector).^2); 
        
        end
    end
    
end
