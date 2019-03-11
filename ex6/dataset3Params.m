function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

temp_c = 0;
temp_s = 0;
err = 1;
for i=1:size(vec,1)
    temp_c = vec(i);
    for j= 1:size(vec,1)
        temp_s = vec(j);
        model = svmTrain(X, y, temp_c, @(x1, x2)gaussianKernel(x1, x2, temp_s));
        predictions = svmPredict(model, Xval);
        temp_err = mean(double(predictions ~= yval));
        if err > temp_err
            err = temp_err;
            C = temp_c;
            sigma = temp_s;
        end
        %fprintf("C: %f, sigma: %f, error: %f\n", temp_c, temp_s, temp_err);
    end
end



%fprintf("C: %f, sigma: %f, error: %f\n", C, sigma, err);

% =========================================================================

end
