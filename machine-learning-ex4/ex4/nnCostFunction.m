function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
J_reg_theta1 = 0;
J_reg_theta2 = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% --------------------REGULARISED COST CALCULATION-----------------------------------------
X = [ones(m, 1) X];%5000*401
a_hidden = sigmoid(Theta1 * X');%(25*401)*(401*5000)=25*5000
a_hidden = [ones(m, 1)';a_hidden];%26*5000
h_theta = sigmoid(Theta2 * a_hidden);%(10*26)*(26*5000)=10*5000
% fprintf('Size h_theta: %d %d\n', size(h_theta));
for i = 1:m
    for k = 1:num_labels
        J = J + (-(log(h_theta(k,i))*(y(i)==k))) - log(1-h_theta(k,i))*(1-(y(i)==k));
    end
end
J = J/m;

for j = 1:size(Theta1, 1)
    for k = 2:size(Theta1, 2)
        J_reg_theta1 = J_reg_theta1 + Theta1(j,k)^2;
    end
end

for j = 1:size(Theta2, 1)
    for k = 2:size(Theta2, 2)
        J_reg_theta2 = J_reg_theta2 + Theta2(j,k)^2;
    end
end

J = J + ((J_reg_theta1+J_reg_theta2)*lambda/(2*m));
% --------------------BACKPROPAGATION ALGORITHM-----------------------------------------
a_1 = X;%5000*401
z_2 = a_1 * Theta1';%(5000*401)*(401*25)=5000*25
a_2 = sigmoid(z_2);%5000*25
a_2 = [ones(m, 1) a_2];%5000*26
z_3 = a_2 * Theta2';%(5000*26)*(26*10)=5000*10
a_3 = sigmoid(z_3);%5000*10
y_1 = zeros(size(a_3));%5000*10
for i = 1:m
    a = y(i);
    y_1(i,a) = 1;
end
delta_3 = a_3 - y_1; %5000*10
delta_2 = delta_3 * Theta2(:,2:end).* sigmoidGradient(z_2); %(5000*10)*(10*25).*(5000*25) = 5000*25

Theta1_grad = delta_2' * a_1; %25*5000*5000*401 = 25*401
Theta2_grad = delta_3' * a_2; %10*5000*5000*26 = 10*26

Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;

Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta1 = Theta1 * (lambda/m);
Theta2 = Theta2 * (lambda/m);

Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;
% =========================================================================
% fprintf('Size Theta1_grad: %d %d\n', size(Theta1_grad));
% fprintf('Size Theta2_grad: %d %d\n', size(Theta2_grad));
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
