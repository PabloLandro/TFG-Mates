function [X_train, X_test, y_train, y_test] = particion_train_test(X, y, porcentaje_test)
% particion_train_test divide el conjunto de datos en subconjuntos de entrenamiento y prueba.
% Entradas:
%   X - Matriz de características (las filas son muestras, las columnas son características)
%   y - Vector de etiquetas (las filas son etiquetas correspondientes a X)
%   testPercentage - Porcentaje de los datos a utilizar para prueba (0-100)
% Salidas:
%   X_train - Subconjunto de entrenamiento de X
%   X_test - Subconjunto de prueba de X
%   y_train - Subconjunto de entrenamiento de y
%   y_test - Subconjunto de prueba de y

    % Check that X and y have compatible dimensions
    if size(X, 1) ~= length(y)
        error('Number of samples in X and y must match.');
    end

    % Shuffle the dataset
    rng('shuffle'); % Ensures randomness in every run
    numSamples = size(X, 1);
    indices = randperm(numSamples);

    % Determine the split point
    numTest = round(porcentaje_test * numSamples);
    testIndices = indices(1:numTest);
    trainIndices = indices(numTest + 1:end);

    % Split the dataset
    X_test = X(testIndices, :);
    X_train = X(trainIndices, :);
    y_test = y(testIndices);
    y_train = y(trainIndices);
end
