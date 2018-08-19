clear; close all; clc

fprintf('Loading Data ...\n');
input_layer_size  = 4;  
num_labels = 3;  

%step 1: get the ddata set
data = load('irisdata.txt');

%load into matrices
[a,b] = size(data);
X = data(:, 1:b-1);
x = data(:,1:2);
y = data(:, b);
m = size(X, 1);
n = size(X, 2);
fprintf('Program paused. Press enter to continue.\n');
pause;

%Plotting
plotdata(x,y);
hold on;
xlabel('sepal length');
ylabel('sepal width');
legend('Iris-Setosa', 'Iris-Versicolor', 'Iris-Virginica');
hold off;
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Vectorize Logistic Regression
fprintf('For lambda = 0\n');

lambda = 0;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;

% Step 3: Predict for One-Vs-All 

pred = predictOneVsAll(all_theta, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
pause;

% Vectorize Logistic Regression

fprintf('\nTraining One-vs-All Logistic Regression...\n')
fprintf('For lambda = 2\n');

lambda = 2;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;
initial_theta = zeros(n + 1, 1);

options = optimset('GradObj', 'on', 'MaxIter', 150);

[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
% NOTE: that by using fminunc, you do not have to write any loops yourself,
% or set a learning rate like you did for gradient descent. You ONLY need
% to provide a function calculating the cost and the gradient.

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

plotDecisionBoundary(theta, X, y);

pred = predictOneVsAll(all_theta, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;
