clear ; close all; clc

% Parametros para la red neuronal
num_entrada  = 784;         % 28x28 números por imagen
num_oculta = 25;            % 25 neuronas ocultas
num_etiquetas = 10;         % 10 etiquetas (el 10 se corresponde con el 0)   
lambda = 0.1;               % Parámetro de regularización

% Creamos abreviaturas de estas funciones para facilitar llamarlas desde el
% descenso de gradiente
costFunction = @(p, X_este, y_este) funcioncoste(p, num_entrada, num_oculta, num_etiquetas, X_este, y_este, lambda);
desenrollar = @(pesos) desenrollar_pesos(pesos, num_entrada, num_oculta, num_etiquetas);

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

%load('ex4data1.mat');
load('mnist-original.mat');
X = double(data');
X = double(X) / 255;
label(label == 0) = 10;
y = label';




% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;


%-------------------------Train test split--------------------------------%

% Vemos los tamaños de cada conjunto de datos
[X_train, X_test, y_train, y_test] = particion_train_test(X,y, 0.2);
fprintf('Tamaño X_train: %d x %d\n', size(X_train, 1), size(X_train, 2));
fprintf('Tamaño X_test: %d x %d\n', size(X_test, 1), size(X_test, 2));
fprintf('Tamaño y_train: %d x %d\n', size(y_train, 1), size(y_train, 2));
fprintf('Tamaño y_test: %d x %d\n', size(y_test, 1), size(y_test, 2));
displayData(X_train(1:100, :));
fprintf('Program paused. Press enter to continue.\n');
pause;


% Contamos las etiquetas en cada conjunto, para comprobar que no se está
% dejando ninguna fuera del entrenamiento
[labelCounts, ~] = histcounts(y_train, 1:11);
disp('Distribución de etiquetas en el conjunto de entrenamiento:');
disp(labelCounts);
[labelCounts, ~] = histcounts(y_test, 1:11);
disp('Distribución de etiquetas en el conjunto de test:');
disp(labelCounts);
pause

%--------------------------Inicialización NN------------------------------%

% Inicialización aleatoria de los pesos

W1_inicial = randInitializeWeights(num_entrada, num_oculta);
W2_inicial = randInitializeWeights(num_oculta, num_etiquetas);

% Desenrollamos los pesos
pesos_iniciales = [W1_inicial(:) ; W2_inicial(:)];

%----------------------------Pruebas regularización-----------------------%

%{

% Pruebas de parámetro regularización
max_iter = 100;
eps = 0.1;
lambdas = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20];

precision_train = zeros(size(lambdas));
precision_test = zeros(size(lambdas));


for i = 1:length(lambdas)
    aux_cost = @(p, X_este, y_este) funcioncoste(p, num_entrada, num_oculta, num_etiquetas, X_este, y_este, lambdas(i));
    [pesos] = descensogradiente(aux_cost, pesos_iniciales, X_train, y_train, desenrollar, max_iter, eps);
    [W1, W2] = desenrollar(pesos);
    [~, precision_train(i)] = predict(W1, W2, X_train, y_train);
    [~, precision_test(i)] = predict(W1, W2, X_test, y_test);
end

figure;
plot(lambdas, precision_train, 'r', 'LineWidth', 2);
hold on;
plot(lambdas, precision_test, 'b', 'LineWidth', 2);
hold off;
xlabel('Lambda');
ylabel('Error');
title('Curva de sobreajuste');
legend({'Error de entrenamiento', 'Error de test'}, 'Location', 'best'); % Leyenda
%ylim([0 1]);
grid on; % Mostrar una cuadrícula
pause;

%--------------------------------Figura Learning Curve--------------------%

% Pruebas de learning curve
max_iter = 100000;
eps = 0.1:0.05:0.95;
tiempos = zeros(size(eps));

aux = @(eps) descensogradiente(costFunction, pesos_iniciales, X_train, y_train, desenrollar, max_iter, eps);

for i = 1:length(eps)
    tic;
    aux(eps(i));
    tiempos(i) = toc;
end

plot(1-eps, log(tiempos), '-o');
xlabel('Precisión');
ylabel('Tiempo de entrenamiento (seconds)');
title('Curva de aprendizaje');
pause;
%}
%---------------------------------Entrenamiento---------------------------%
fprintf('\nEntrenando red neuronal... \n')
max_iter = 10000;    % Iteraciones máximas del descenso de gradiente
eps = 0.2;         % Condición de parada (error mínimo)
lambda = 0.1;       % Parámetro de regularización



[pesos] = descensogradiente(costFunction, pesos_iniciales, X_train, y_train, desenrollar, max_iter, eps);
[W1, W2] = desenrollar(pesos);

fprintf('Program paused. Press enter to continue.\n');
pause;


% Visualizamos los pesos aprendidos

fprintf('\nVisualizando los pesos de la red neuronal\n')

displayData(W1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Comprobamos la precisión en el conjunto de test

[pred, precision] = predict(W1, W2, X_test, y_test);
fprintf('Precisión en el conjunto de test: %f\n', precision * 100);



% RDE

% cogemos una imagen que haya sido bien clasificada
% Seleccionamos una imagen que ha sido bien clasificada
correct_indices = find(pred == y_test); % Índices de las predicciones correctas
selected_index = correct_indices(randi(length(correct_indices))); % Elegimos un índice aleatorio de los correctos
selected_image = reshape(X_test(selected_index, :), [28,28]) % Imagen seleccionada

% Visualizamos la imagen seleccionada
displaySingleImage(selected_image);
title('Imagen seleccionada que ha sido bien clasificada');
% Generamos una máscara aleatoria de números entre 0 y 1 del tamaño de la imagen
S = rand(size(selected_image));
% Llamamos a gradiente numérico para minimizar la función de distorsión

% Aplanamos la imagen y la máscara
x = selected_image(:);
S = S(:);

%Creamos una abreviatura de la distorsión para llamarla solo con la máscara
distortionFunction = @(S) distorsion(pesos, num_entrada, num_oculta, num_etiquetas, x, S);

for i = 1:1000
    fprintf("Iteración %d\n", i);
    S = S + gradientenumerico(distortionFunction, S, 0.001);
end

fprintf('distorsión final: %f\n', distortionFunction(S));


% Visualizamos la máscara
displaySingleImage(reshape(S_opt, size(selected_image))); 
title('Máscara generada para la imagen seleccionada');
