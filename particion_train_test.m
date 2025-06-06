function [X_train, X_test, y_train, y_test] = particion_train_test(X, y, testPercentage)
    % particion_train_test splits the dataset into training and testing subsets.
    % Inputs:
    %   X - Feature matrix (rows are samples, columns are features)
    %   y - Label vector (rows are labels for corresponding X)
    %   testPercentage - Percentage of the data to use for testing (0-100)
    % Outputs:
    %   X_train - Training subset of X
    %   X_test - Testing subset of X
    %   y_train - Training subset of y
    %   y_test - Testing subset of y

    % Check that X and y have compatible dimensions
    if size(X, 1) ~= length(y)
        error('Number of samples in X and y must match.');
    end

    % Shuffle the dataset
    rng('shuffle'); % Ensures randomness in every run
    numSamples = size(X, 1);
    indices = randperm(numSamples);

    % Determine the split point
    numTest = round(testPercentage * numSamples);
    testIndices = indices(1:numTest);
    trainIndices = indices(numTest + 1:end);

    % Split the dataset
    X_test = X(testIndices, :);
    X_train = X(trainIndices, :);
    y_test = y(testIndices);
    y_train = y(trainIndices);
end
