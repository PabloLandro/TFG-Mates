function [h, display_array] = displayData(X, example_width)
%DISPLAYDATA Display 2D data in a nice grid
%   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
%   stored in X in a nice grid. It returns the figure handle h and the 
%   displayed array if requested.

% Set example_width automatically if not passed in
if ~exist('example_width', 'var') || isempty(example_width) 
	example_width = round(sqrt(size(X, 2)));
end

% Gray Image
colormap(gray);

% Compute rows, cols
[m, n] = size(X);
example_height = round(n / example_width); % Ensure example_height is an integer

% Validate that the dimensions match
if example_height * example_width ~= n
    error('The number of features in X does not match example_width * example_height.');
end

% Compute number of items to display
display_rows = floor(sqrt(m));
display_cols = ceil(m / display_rows);

% Between images padding
pad = 1;

% Setup blank display
display_array = -ones(pad + display_rows * (example_height + pad), ...
                      pad + display_cols * (example_width + pad));

% Copy each example into a patch on the display array
curr_ex = 1;
for j = 1:display_rows
	for i = 1:display_cols
		if curr_ex > m, 
			break; 
		end
		% Copy the patch
		max_val = max(abs(X(curr_ex, :))); % Get the max value of the patch
		%display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		%              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
		%				reshape(X(curr_ex, :), example_height, example_width) / max_val;
        display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
                      pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
                      rot90(flipud(reshape(X(curr_ex, :), example_height, example_width)), -1) / max_val;
		curr_ex = curr_ex + 1;
	end
	if curr_ex > m, 
		break; 
	end
end

% Display Image
h = imagesc(display_array, [-1 1]);

% Do not show axis
axis image off

drawnow;

end
