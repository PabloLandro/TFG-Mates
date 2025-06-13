function g = sigmoidal_grad(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = sigmoidal(z).*(1-sigmoidal(z));

%g = double(z > 0);

end