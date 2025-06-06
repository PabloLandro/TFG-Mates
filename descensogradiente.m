function [pesos, i] = descensogradiente(coste, pesos_iniciales, X, y, desenrollar, max_iter, eps)

learning_rate = 0.5;

num_rows = size(X, 1);
num_select = ceil(100 * num_rows / 100);
sel = randperm(size(X, 1));
sel = sel(1:num_select);

%X = X(sel, :);
%y = y(sel,:);

pesos = pesos_iniciales;

for i = 1:max_iter
    % Evaluate function value and gradient
    [fval, grad] = coste(pesos, X, y);
    fprintf('Iteración %d, coste: %.3f\n', i, fval);
    pesos = pesos-learning_rate.*grad;

    % Comprobamos la precisión en el conjunto de entrenamiento
    [W1, W2] = desenrollar(pesos);

    [~, precision] = predict(W1, W2, X, y);
    fprintf('Precisión %.3f\n', precision);
    if precision >= 1 - eps
        fprintf('Condición de precisión satisfecha. Terminando el entrenamiento en la iteración %d.\n', i);
        break; % Exit the loop
    end
end
