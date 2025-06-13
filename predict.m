function [pred, precision] = predict(W1, W2, X, y)

m = size(X, 1);

pred = zeros(size(X, 1), 1);

h1 = sigmoidal([ones(m, 1) X] * W1');
h2 = sigmoidal([ones(m, 1) h1] * W2');
[~, pred] = max(h2, [], 2);

if ~exist('y') || isempty(y)
    precision = 0;
else
    % Si se pasa el valor de y, también devolvemos la precisión
    precision = mean(double(pred == y));
end


end