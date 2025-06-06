function [W1, W2] = desenrrollar_pesos(pesos, num_entrada, num_oculta, num_etiquetas)

W1 = reshape(pesos(1:num_oculta * (num_entrada + 1)), num_oculta, (num_entrada + 1));

W2 = reshape(pesos((1 + (num_oculta * (num_entrada + 1))):end), num_etiquetas, (num_oculta + 1));