function [J, grad] = funcioncoste(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

W1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

W2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X,1);

W1_grad = zeros(size(W1));
W2_grad = zeros(size(W2));

X = [ones(m, 1) X];
a2=sigmoidal(W1*X');
a2 = [ones(m, 1) a2'];
a3=sigmoidal(a2*W2');
y_vect=zeros(num_labels,m);

for n=1:m
    y_vect(y(n),n) = 1;
end

J = sum(sum(-y_vect.*log(a3') - (1-y_vect).*log(1-a3')))/m;

W1_reg = W1;
W1_reg(:,1)=[];
W2_reg = W2;
W2_reg(:,1)=[];

%add regularization term
J = J + lambda*(sum(sum(W1_reg.^2)) + sum(sum(W2_reg.^2)))/(2*m);


z2=[ones(m, 1) (W1*X')'];
d3=a3'-y_vect;

d2=W2'*d3.*sigmoidal_grad(z2');

d2(1,:)=[];

W1_grad = (W1_grad + d2*X)/m;
W2_grad = (W2_grad + d3*a2)/m;

tmp=size(W1_grad);
W1_grad(:,2:tmp(2)) = W1_grad(:,2:tmp(2)) + lambda*W1(:,2:tmp(2))/m;
tmp2=size(W2_grad);
W2_grad(:,2:tmp2(2)) = W2_grad(:,2:tmp2(2)) + lambda*W2(:,2:tmp2(2))/m;

grad = [W1_grad(:) ; W2_grad(:)];


end