%Calculates Explained Variance loss
classdef Explained_variance_LM < handle
        
    methods
        
        %%%%%%Arguments%%%%%%
        %obj=object
        %predicted_vector=predicted response vector
        %actual_vector=actual response vector
        
        %%%%%%%Returns%%%%%%%%
        %loss=Explained Variance loss value
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function loss = calculate_loss(obj, predicted_vector, actual_vector)
            total_sum_of_squares = sum((actual_vector-mean(actual_vector)).^2);
            residual_sum_of_squares = sum((predicted_vector-actual_vector).^2);
            loss = 1 - (residual_sum_of_squares/total_sum_of_squares); 
        
        end
    end
    
end