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

predication = sigmoid(X * theta);
oneMinusPredication = 1 .- predication ; 
costFunctionForEachPredication = (-1 .* y .* log(predication) ) + (-1 .* (1 .- y) .* log(oneMinusPredication) );

modificationTheta  = theta;
modificationTheta(1) = 0 ; %zeroth theta not be impacted

J = ((1 ./ m ) .* sum(costFunctionForEachPredication)) + ( (lambda ./(2 .* m))  *  sum(modificationTheta .^ 2) ) ;

grad = 1/m .* sum( ((  predication - y)' .* X' ),2) .+ (lambda ./ m) .* modificationTheta ;


% =============================================================

end
