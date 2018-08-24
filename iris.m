clear; close all; clc

fprintf('Loading Data ...\n');
input_layer_size  = 4;  
num_labels = 3;  

%step 1: get the data set
data = load('irisdata.txt');

%load into matrices
[a,b] = size(data);
X = data(:, 1:b-1);
x = data(:,1:2);
y = data(:, b);
y1 = [ones(50,1);zeros(100,1)];
y1 = [zeros(50,1);ones(50,1);zeros(50,1)];
y3 = [zeros(100,1);ones(50,1)];

fprintf('Program paused. Press enter to continue.\n');
pause;

%Plotting
plotData3(x,y);
hold on;
xlabel('sepal length');
ylabel('sepal width');
legend('Iris-Setosa', 'Iris-Versicolor', 'Iris-Virginica');
hold off;
fprintf('\nProgram paused. Press enter to continue.\n');
pause;


% Step 3: Compute Cost and gradient
[m,n] = size(X);

X = [ones(m,1) X];
init_theta = zeros(n+1,1);
[cost, grad] = costFunction(init_theta, X, y1);

fprintf('Cost at initial theta(0) for y1:%f\n', cost);
fprintf('Gradient at initial theta(zeroes) for y1: \n%f\n', grad);
fprintf('\nProgram paused press ENTER to continue\n');
pause;

% Step 4: Optimising using fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = ...
        fminunc(@(t)(costFunction(t, X, y1)), init_theta, options);
fprintf('Cost at theta found by fminunc y1: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

% Step 5: Plot Boundary
plotDecisionBoundary(theta, X, y1);
pause;







