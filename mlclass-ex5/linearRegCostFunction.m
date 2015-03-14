function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

n = length(theta); % number of features

a = X * theta - y;

% printf('Size of a: %d, %d\n', size(a));

J1 = 1/(2*m) * sum(a .^ 2);
% printf('J1: %.3f\n', J1);

theta2 = theta(2:end);

% J2 = lambda/(2*m) * (theta2 .^ 2);
J2 = lambda/(2*m) * (theta2' * theta2);
% printf('J2: %.3f\n', J2);

J = J1+J2;
% printf('J: %.3f\n', J);

% grad = ((X * theta - y)' * X)/m;
% printf('Size of X: %d, %d\n', size(X));

grad = (X' * (X * theta - y))/m;
grad(2:end) += (lambda * theta2)/m;


% for i=1:m,
% 	% or calling dot() to explicitly use dot product
% 	h = X(i,:) * theta;
% 	J += (h - y(i)) ^ 2;

% 	for j=1:n,
% 		grad(j) += (h - y(i)) * X(i, j);
% 	end;
% end;

% lambda_term = 0;
% % NOTE start from index 2 to avoid regularize theta(1)
% for j=2:n,
% 	lambda_term += lambda * (theta(j, :) ^ 2);
% end
% J += lambda_term;

% for j=2:n,
% 	% apply lambda term
% 	grad_lambda_term = lambda * theta(j, :);
% 	grad(j) += grad_lambda_term;
% end;

% % or using ./ operator to explicit use element wise divide operation
% J = J/(2*m);
% grad = grad/m;






% =========================================================================

grad = grad(:);

end
