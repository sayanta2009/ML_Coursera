function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
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
C_val = [0.01 0.03 0.1 0.3 1.0 3.0 10.0 30.0];
sigma_val = [0.01 0.03 0.1 0.3 1.0 3.0 10.0 30.0];
leng = size(C_val,2);
results = zeros(leng^2, 3);
i = 1;

for c = 1:leng
    for s = 1:leng
       model = svmTrain(X, y, C_val(c), @(x1, x2) gaussianKernel(x1, x2, sigma_val(s)));
       predictions = svmPredict(model, Xval);
       res = mean(double(predictions ~= yval));
       results(i,:) = [res C_val(c) sigma_val(s)];
       i = i+1;
    end
end

[~, row_val] = min(results(:,1));
C = results(row_val,2);
sigma = results(row_val,3);
% =========================================================================

end
