function h = displaySingleImage(X, index, example_width)
%DISPLAY SINGLE IMAGE Display one 2D example from X given by index
%   h = DISPLAY SINGLE IMAGE(X, index, example_width) displays the image at 
%   row `index` in X. `example_width` is optional.

    if ~exist('example_width', 'var') || isempty(example_width)
        example_width = round(sqrt(size(X, 2)));
    end

    if ~exist('index') || isempty(index)
        img = X;
    else
        example_height = size(X, 2) / example_width;
    
        % Extract the example vector and reshape it into a 2D image
        %img = reshape(X(index, :), example_height, example_width);
    
        img = rot90(flipud(reshape(X(index, :), example_height, example_width)), -1);

    % Normalize image to [-1, 1] or [0,1]
    max_val = max(abs(img(:)));
    if max_val > 0
        img = img / max_val;
    end

    % Show the image
    colormap(gray);
    h = imagesc(img, [-1 1]);
    axis image off;
    drawnow;
end
