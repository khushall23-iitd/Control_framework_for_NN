clc; clear; close all;

%% ============================================================
%  XOR 2-2-1 Neural Network: SGD, SGD-Momentum, Adam, CLF-GD
%  Loss: Binary Cross Entropy
%  Network: 2 inputs -> 2 hidden neurons -> 1 output
%% ============================================================

rng(10);   % Reproducibility

%% XOR Dataset
X = [0 0;
     0 1;
     1 0;
     1 1];

Y = [0; 1; 1; 0];

N = size(X,1);

%% Training Parameters
epochs = 1000;

% SGD
eta_sgd = 0.8;

% SGD with Momentum
eta_mom = 0.8;
beta_mom = 0.9;

% Adam
eta_adam = 0.05;
beta1 = 0.9;
beta2 = 0.999;
eps_adam = 1e-8;

% CLF-GD
alpha_clf = 0.4;

%% Same Initial Parameters for All Optimizers
% theta = [w11 w12 w21 w22 b1 b2 v1 v2 b3]^T
theta0 = 0.5 * randn(9,1);

theta_sgd  = theta0;
theta_mom  = theta0;
theta_adam = theta0;
theta_clf  = theta0;

% Momentum and Adam Variables
vel_mom = zeros(9,1);

m_adam = zeros(9,1);
v_adam = zeros(9,1);

%% Storage
loss_sgd  = zeros(epochs,1);
loss_mom  = zeros(epochs,1);
loss_adam = zeros(epochs,1);
loss_clf  = zeros(epochs,1);

acc_sgd  = zeros(epochs,1);
acc_mom  = zeros(epochs,1);
acc_adam = zeros(epochs,1);
acc_clf  = zeros(epochs,1);

lambda_clf_store = zeros(epochs,1);

%% Training Loop
for k = 1:epochs

    %% ---------------- SGD ----------------
    [L, g, yhat] = lossGrad_XOR_221(theta_sgd, X, Y);
    theta_sgd = theta_sgd - eta_sgd * g;

    loss_sgd(k) = L;
    acc_sgd(k) = classification_accuracy(yhat,Y);

    %% ---------------- SGD + Momentum ----------------
    [L, g, yhat] = lossGrad_XOR_221(theta_mom, X, Y);
    vel_mom = beta_mom * vel_mom + g;
    theta_mom = theta_mom - eta_mom * vel_mom;

    loss_mom(k) = L;
    acc_mom(k) = classification_accuracy(yhat,Y);

    %% ---------------- Adam ----------------
    [L, g, yhat] = lossGrad_XOR_221(theta_adam, X, Y);

    m_adam = beta1 * m_adam + (1-beta1) * g;
    v_adam = beta2 * v_adam + (1-beta2) * (g.^2);

    m_hat = m_adam / (1-beta1^k);
    v_hat = v_adam / (1-beta2^k);

    theta_adam = theta_adam - eta_adam * m_hat ./ (sqrt(v_hat) + eps_adam);

    loss_adam(k) = L;
    acc_adam(k) = classification_accuracy(yhat,Y);

    %% ---------------- CLF-Based GD ----------------
    [L, g, yhat] = lossGrad_XOR_221(theta_clf, X, Y);

    lambda_clf = L / (norm(g)^2 + alpha_clf * L + 1e-12);

    theta_clf = theta_clf - lambda_clf * g;

    loss_clf(k) = L;
    acc_clf(k) = classification_accuracy(yhat,Y);
    lambda_clf_store(k) = lambda_clf;

end

%% Final Predictions
fprintf('\nFinal XOR Predictions\n');
fprintf('======================\n');

print_predictions('SGD', theta_sgd, X, Y);
print_predictions('SGD-Momentum', theta_mom, X, Y);
print_predictions('Adam', theta_adam, X, Y);
print_predictions('CLF-GD', theta_clf, X, Y);

%% ============================================================
%  PLOT 1: Training Loss Comparison
%% ============================================================

figure('Color','w','Position',[100 100 950 650]);

plot(1:epochs, loss_sgd,  'LineWidth',3.5); hold on;
plot(1:epochs, loss_mom,  'LineWidth',3.5);
plot(1:epochs, loss_adam, 'LineWidth',3.5);
plot(1:epochs, loss_clf,  'LineWidth',4.2);

grid on; box on;

xlabel('Epoch','FontSize',26,'FontWeight','bold');
ylabel('Binary Cross-Entropy Loss','FontSize',26,'FontWeight','bold');
title('XOR 2-2-1 Network: Training Loss Comparison','FontSize',28,'FontWeight','bold');

legend({'SGD','SGD-Momentum','Adam','CLF-GD'}, ...
    'FontSize',20,'FontWeight','bold','Location','northeast');

set(gca,'FontSize',22,'FontWeight','bold','LineWidth',3);
xlim([1 epochs]);

%% ============================================================
%  PLOT 2: Accuracy Comparison
%% ============================================================

figure('Color','w','Position',[150 150 950 650]);

plot(1:epochs, acc_sgd,  'LineWidth',3.5); hold on;
plot(1:epochs, acc_mom,  'LineWidth',3.5);
plot(1:epochs, acc_adam, 'LineWidth',3.5);
plot(1:epochs, acc_clf,  'LineWidth',4.2);

grid on; box on;

xlabel('Epoch','FontSize',26,'FontWeight','bold');
ylabel('Classification Accuracy (%)','FontSize',26,'FontWeight','bold');
title('XOR 2-2-1 Network: Accuracy Comparison','FontSize',28,'FontWeight','bold');

legend({'SGD','SGD-Momentum','Adam','CLF-GD'}, ...
    'FontSize',20,'FontWeight','bold','Location','southeast');

set(gca,'FontSize',22,'FontWeight','bold','LineWidth',3);
ylim([0 105]);
xlim([1 epochs]);

%% ============================================================
%  PLOT 3: CLF Adaptive Step Size
%% ============================================================

figure('Color','w','Position',[200 200 950 650]);

plot(1:epochs, lambda_clf_store, 'k', 'LineWidth',4);

grid on; box on;

xlabel('Epoch','FontSize',26,'FontWeight','bold');
ylabel('CLF Step Size \lambda_k','FontSize',26,'FontWeight','bold');
title('Adaptive CLF-Based Step Size','FontSize',28,'FontWeight','bold');

set(gca,'FontSize',22,'FontWeight','bold','LineWidth',3);
xlim([1 epochs]);

%% ============================================================
%  PLOT 4: Decision Boundary of CLF-GD
%% ============================================================

plot_decision_boundary(theta_clf, X, Y);

%% ============================================================
%  Local Functions
%% ============================================================

function [L, g, yhat] = lossGrad_XOR_221(theta, X, Y)

    % Parameters
    w11 = theta(1); w12 = theta(2);
    w21 = theta(3); w22 = theta(4);
    b1  = theta(5); b2  = theta(6);
    v1  = theta(7); v2  = theta(8);
    b3  = theta(9);

    N = size(X,1);

    % Forward Pass
    x1 = X(:,1);
    x2 = X(:,2);

    z1 = w11*x1 + w12*x2 + b1;
    z2 = w21*x1 + w22*x2 + b2;

    h1 = sigmoid(z1);
    h2 = sigmoid(z2);

    s = v1*h1 + v2*h2 + b3;
    yhat = sigmoid(s);

    % Numerical protection
    yhat = min(max(yhat,1e-10),1-1e-10);

    % Binary Cross Entropy Loss
    L = -(1/N) * sum(Y.*log(yhat) + (1-Y).*log(1-yhat));

    % Backpropagation
    delta_o = yhat - Y;

    delta_h1 = delta_o .* v1 .* h1 .* (1-h1);
    delta_h2 = delta_o .* v2 .* h2 .* (1-h2);

    dL_dw11 = (1/N) * sum(delta_h1 .* x1);
    dL_dw12 = (1/N) * sum(delta_h1 .* x2);

    dL_dw21 = (1/N) * sum(delta_h2 .* x1);
    dL_dw22 = (1/N) * sum(delta_h2 .* x2);

    dL_db1 = (1/N) * sum(delta_h1);
    dL_db2 = (1/N) * sum(delta_h2);

    dL_dv1 = (1/N) * sum(delta_o .* h1);
    dL_dv2 = (1/N) * sum(delta_o .* h2);
    dL_db3 = (1/N) * sum(delta_o);

    g = [dL_dw11;
         dL_dw12;
         dL_dw21;
         dL_dw22;
         dL_db1;
         dL_db2;
         dL_dv1;
         dL_dv2;
         dL_db3];
end

function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

function acc = classification_accuracy(yhat,Y)
    pred = double(yhat >= 0.5);
    acc = 100 * mean(pred == Y);
end

function print_predictions(name, theta, X, Y)

    [L, ~, yhat] = lossGrad_XOR_221(theta, X, Y);
    pred = double(yhat >= 0.5);

    fprintf('\n%s\n',name);
    fprintf('Final Loss = %.6f\n',L);
    fprintf(' x1  x2   y   yhat     pred\n');
    fprintf('-----------------------------\n');

    for i = 1:size(X,1)
        fprintf(' %d   %d    %d   %.4f     %d\n', ...
            X(i,1), X(i,2), Y(i), yhat(i), pred(i));
    end
end

function plot_decision_boundary(theta, X, Y)

    [xg, yg] = meshgrid(linspace(-0.2,1.2,300), linspace(-0.2,1.2,300));
    grid_points = [xg(:), yg(:)];

    yhat_grid = predict_XOR_221(theta, grid_points);
    Z = reshape(yhat_grid, size(xg));

    figure('Color','w','Position',[250 250 850 700]);

    contourf(xg, yg, Z, 30, 'LineColor','none'); hold on;
    contour(xg, yg, Z, [0.5 0.5], 'k', 'LineWidth',5);

    scatter(X(Y==0,1), X(Y==0,2), 220, 'o', ...
        'filled', 'MarkerEdgeColor','k', 'LineWidth',3);

    scatter(X(Y==1,1), X(Y==1,2), 220, 's', ...
        'filled', 'MarkerEdgeColor','k', 'LineWidth',3);

    grid on; box on;

    xlabel('$x_1$','Interpreter','latex','FontSize',30,'FontWeight','bold');
    ylabel('$x_2$','Interpreter','latex','FontSize',30,'FontWeight','bold');
    title('CLF-GD Decision Boundary for XOR','FontSize',28,'FontWeight','bold');

    legend({'Decision Boundary','Class 0','Class 1'}, ...
        'FontSize',20,'FontWeight','bold','Location','best');

    set(gca,'FontSize',24,'FontWeight','bold','LineWidth',3);
    axis equal;
    xlim([-0.2 1.2]);
    ylim([-0.2 1.2]);
end

function yhat = predict_XOR_221(theta, X)

    w11 = theta(1); w12 = theta(2);
    w21 = theta(3); w22 = theta(4);
    b1  = theta(5); b2  = theta(6);
    v1  = theta(7); v2  = theta(8);
    b3  = theta(9);

    x1 = X(:,1);
    x2 = X(:,2);

    z1 = w11*x1 + w12*x2 + b1;
    z2 = w21*x1 + w22*x2 + b2;

    h1 = sigmoid(z1);
    h2 = sigmoid(z2);

    s = v1*h1 + v2*h2 + b3;
    yhat = sigmoid(s);
end