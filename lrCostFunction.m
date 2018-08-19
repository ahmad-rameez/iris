function [J, grad] = lrCostFunction(theta, X, y, lambda)


% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


% calculate hypothesis
h = sigmoid(X*theta);

% regularize theta by removing first value
theta_reg = [0;theta(2:end, :);];

J = (1/m)*(-y'* log(h) - (1 - y)'*log(1-h))+(lambda/(2*m))*theta_reg'*theta_reg;

grad = (1/m)*(X'*(h-y)+lambda*theta_reg);


grad = grad(:);

end
