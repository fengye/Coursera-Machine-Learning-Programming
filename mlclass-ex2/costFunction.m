function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%


for i=1:m,
	% or calling dot() to explicitly use dot product
	z = X(i,:) * theta;
	h = sigmoid(z);

	J += -y(i) * log(h) - (1-y(i)) * log(1-h);


	for j=1:n,
		grad(j) += (h - y(i)) * X(i, j);
	end;
end;

% or using ./ operator to explicit use element wise divide operation
J = J/m;
grad = grad/m;





% =============================================================

end