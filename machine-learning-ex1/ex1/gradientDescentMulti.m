function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
col = size(X, 2);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    h = zeros(m, 1);
    for i = 1:m
        for k = 1:col
            h(i) = h(i) + theta(k) * X(i, k);
        end    
    end
    
    for k = 1:col
        derv_theta = 0;
        for i = 1:m
            derv_theta = derv_theta + (h(i) - y(i))*X(i, k);
        end
        theta(k,:) = theta(k,:) - (alpha/m) * derv_theta;
    end
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
end

end
