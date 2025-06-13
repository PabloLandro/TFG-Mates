function [disto] = distorsion(pesos, num_entrada, num_oculta, num_etiquetas, x, S)
 
% Suponemos S como vector columna

lambda = 0.1;

[W1, W2] = desenrollar_pesos(pesos, num_entrada, num_oculta, num_etiquetas);

% Obtenemos los momentos estadísticos por montecarlo

% Input distribution parameters
N = 100; % Number of samples
mu = zeros(1, num_entrada); % Mean vector
Sigma = eye(num_entrada); % Covariance matrix (identity for independent normal variables)

% Generate random samples
x_samples = mvnrnd(mu, Sigma, N); % N samples, each of size n

y_samples = x_samples .* S;

% Evaluate the function on the samples
phi_y_samples = predict(W1, W2, y_samples); % f(x) applied row-wise to x_samples

% Estimate expected value and variance
E_phi_y = mean(phi_y_samples); % Monte Carlo estimate of E[f(x)]
Var_phi_y = var(phi_y_samples); % Monte Carlo estimate of Var(f(x)]

phi_x = predict(W1, W2, x);

disto = phi_x - E_phi_y + Var_phi_y + lambda * norm(S, 1);

end