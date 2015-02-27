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
n = length(theta);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta




for i=1:m,
	% or calling dot() to explicitly use dot product
	z = X(i,:) * theta;
	h = sigmoid(z);

	J += -y(i) * log(h) - (1-y(i)) * log(1-h);

	
	for j=1:n,
		grad(j) += (h - y(i)) * X(i, j);
	end;
end;

lambda_term = 0;
% NOTE start from index 2 to avoid regularize theta(1)
for j=2:n,
	lambda_term += lambda / 2 * (theta(j, :) ^ 2);
end
J += lambda_term;

for j=2:n,
	% apply lambda term
	grad_lambda_term = lambda * theta(j, :);
	grad(j) += grad_lambda_term;
end;

% or using ./ operator to explicit use element wise divide operation
J = J/m;
grad = grad/m;



% =============================================================

end
