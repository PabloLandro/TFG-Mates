function [J, grad] = funcioncoste(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X,1);

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m, 1) X];
a2=activation(Theta1*X');
a2 = [ones(m, 1) a2'];
a3=activation(a2*Theta2');
y_vect=zeros(num_labels,m);

%convert labels to vectors
for n=1:m
    y_vect(y(n),n) = 1;
end

%unregularized cost function
J = sum(sum(-y_vect.*log(a3') - (1-y_vect).*log(1-a3')))/m;

%bias terms are not regularized
Theta1_reg = Theta1;
Theta1_reg(:,1)=[];
Theta2_reg = Theta2;
Theta2_reg(:,1)=[];

%add regularization term
J = J + lambda*(sum(sum(Theta1_reg.^2)) + sum(sum(Theta2_reg.^2)))/(2*m);


z2=[ones(m, 1) (Theta1*X')'];
d3=a3'-y_vect;

d2=Theta2'*d3.*activationGradient(z2');

d2(1,:)=[];

Theta1_grad = (Theta1_grad + d2*X)/m;
Theta2_grad = (Theta2_grad + d3*a2)/m;

%regularize gradient
tmp=size(Theta1_grad);
Theta1_grad(:,2:tmp(2)) = Theta1_grad(:,2:tmp(2)) + lambda*Theta1(:,2:tmp(2))/m;
tmp2=size(Theta2_grad);
Theta2_grad(:,2:tmp2(2)) = Theta2_grad(:,2:tmp2(2)) + lambda*Theta2(:,2:tmp2(2))/m;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end