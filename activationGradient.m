function g = activationGradient(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = activation(z).*(1-activation(z));

%g = double(z > 0);

end