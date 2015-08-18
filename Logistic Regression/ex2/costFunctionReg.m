function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Reusing costFunction.m
[J, grad] = costFunction(theta, X, y);

% taking care of theta(1)
clean_theta = [0;theta(2:end)];

% new cost function is old cost function plus an additional term
J = J + ((lambda/(2*m)) * (clean_theta' * clean_theta));

% new grad is old grad plus an additional term
grad = grad + ((lambda/m) * clean_theta);




% =============================================================

end
