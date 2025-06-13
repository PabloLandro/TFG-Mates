function grad = gradientenumerico(f, x, h)
    
    % Computamos el gradiente numérico
    n = length(x);            % Número de variables
    grad = zeros(1,n);        % Inicializamos el gradiente
    for i = 1:n
        e = zeros(1,n);       % Vector Unidad
        e(i) = 1;             % Perturbación en la i-ésima dirección
        grad(i) = (f(x + h * e) - f(x)) / h; % Derivada numéricas
    end
end