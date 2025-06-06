function [pred, precision] = predict(Theta1, Theta2, X, y)

% Useful values
m = size(X, 1);

% You need to return the following variables correctly 
pred = zeros(size(X, 1), 1);

h1 = activation([ones(m, 1) X] * Theta1');
h2 = activation([ones(m, 1) h1] * Theta2');
[~, pred] = max(h2, [], 2);

if isempty(y)
    precision = 0;
else
    precision = mean(double(pred == y));
end


end