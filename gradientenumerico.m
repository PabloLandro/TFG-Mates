function grad = gradientenumerico(f, x, h)
    % Ensure x is a column vector
    if size(x, 2) > 1
        x = x'; % Convert to column vector
    end
    
    % Compute numerical gradient of f at x
    n = length(x);            % Number of variables
    grad = zeros(n, 1);       % Initialize gradient vector
    for i = 1:n
        e = zeros(n, 1);      % Unit vector
        e(i) = 1;             % Perturb along the i-th direction
        grad(i) = (f(x + h * e) - f(x)) / h; % Numerical derivative
    end
end