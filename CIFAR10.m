clc; clear; close all;

%% ============================================================
% Actual CIFAR-10 Training
% Comparison: SGD, SGD-Momentum, Adam, CLF-GD
% Architecture: 5-layer CNN
% ============================================================

rng(10);

%% ---------------- Load CIFAR-10 Dataset ----------------
dataFolder = fullfile(pwd,'cifar10_data');

if ~exist(dataFolder,'dir')
    mkdir(dataFolder);
end

cifarURL = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
zipFile = fullfile(dataFolder,'cifar-10-matlab.tar.gz');

if ~exist(fullfile(dataFolder,'cifar-10-batches-mat'),'dir')
    fprintf('Downloading CIFAR-10 dataset...\n');
    websave(zipFile,cifarURL);
    untar(zipFile,dataFolder);
end

[XTrain,YTrain,XTest,YTest,classes] = loadCIFAR10(dataFolder);

%% ---------------- Normalize Data ----------------
XTrain = single(XTrain)/255;
XTest  = single(XTest)/255;

mu  = mean(XTrain,[1 2 4]);
sig = std(XTrain,0,[1 2 4]) + 1e-8;

XTrain = (XTrain - mu)./sig;
XTest  = (XTest  - mu)./sig;

%% ---------------- Training Parameters ----------------
numEpochs = 50;
miniBatchSize = 128;

% SGD
lr_sgd = 0.01;

% SGD with Momentum
lr_mom = 0.01;
momBeta = 0.9;

% Adam
lr_adam = 0.001;
beta1 = 0.9;
beta2 = 0.999;
epsAdam = 1e-8;

% CLF-GD
alpha_clf = 1.0;

numTrain = size(XTrain,4);
numIterationsPerEpoch = floor(numTrain/miniBatchSize);

%% ---------------- Create Same Initial Network ----------------
baseNet = createCIFAR10CNN(classes);

netSGD  = baseNet;
netMOM  = baseNet;
netADAM = baseNet;
netCLF  = baseNet;

velocityMOM = initializeLike(netMOM.Learnables);

avgGradADAM = initializeLike(netADAM.Learnables);
avgSqADAM   = initializeLike(netADAM.Learnables);

%% ---------------- Storage ----------------
lossSGD  = zeros(numEpochs,1);
lossMOM  = zeros(numEpochs,1);
lossADAM = zeros(numEpochs,1);
lossCLF  = zeros(numEpochs,1);

accSGD  = zeros(numEpochs,1);
accMOM  = zeros(numEpochs,1);
accADAM = zeros(numEpochs,1);
accCLF  = zeros(numEpochs,1);

lambdaCLF = zeros(numEpochs,1);

iterationADAM = 0;

%% ============================================================
% Training Loop
%% ============================================================

for epoch = 1:numEpochs

    idx = randperm(numTrain);

    epochLossSGD  = 0;
    epochLossMOM  = 0;
    epochLossADAM = 0;
    epochLossCLF  = 0;
    epochLambda   = 0;

    for it = 1:numIterationsPerEpoch

        batchIdx = idx((it-1)*miniBatchSize+1 : it*miniBatchSize);

        XBatch = XTrain(:,:,:,batchIdx);
        YBatch = YTrain(batchIdx);

        dlX = dlarray(XBatch,'SSCB');

        %% ---------------- SGD ----------------
        [loss,gradients] = dlfeval(@modelGradients,netSGD,dlX,YBatch,classes);

        netSGD.Learnables = dlupdate(@(p,g) p - lr_sgd*g, ...
                                     netSGD.Learnables, gradients);

        epochLossSGD = epochLossSGD + double(extractdata(loss));

        %% ---------------- SGD-Momentum ----------------
        [loss,gradients] = dlfeval(@modelGradients,netMOM,dlX,YBatch,classes);

        velocityMOM = dlupdate(@(v,g) momBeta*v + g, velocityMOM, gradients);

        netMOM.Learnables = dlupdate(@(p,v) p - lr_mom*v, ...
                                     netMOM.Learnables, velocityMOM);

        epochLossMOM = epochLossMOM + double(extractdata(loss));

        %% ---------------- Adam ----------------
        iterationADAM = iterationADAM + 1;

        [loss,gradients] = dlfeval(@modelGradients,netADAM,dlX,YBatch,classes);

        [netADAM.Learnables,avgGradADAM,avgSqADAM] = adamUpdateCustom( ...
            netADAM.Learnables, gradients, avgGradADAM, avgSqADAM, ...
            iterationADAM, lr_adam, beta1, beta2, epsAdam);

        epochLossADAM = epochLossADAM + double(extractdata(loss));

        %% ---------------- CLF-Based GD ----------------
        [loss,gradients] = dlfeval(@modelGradients,netCLF,dlX,YBatch,classes);

        lossVal = double(extractdata(loss));
        gradNorm2 = gradientNormSquared(gradients);

        lambda = lossVal/(gradNorm2 + alpha_clf*lossVal + 1e-12);

        netCLF.Learnables = dlupdate(@(p,g) p - lambda*g, ...
                                     netCLF.Learnables, gradients);

        epochLossCLF = epochLossCLF + lossVal;
        epochLambda = epochLambda + lambda;

    end

    %% Average Training Loss
    lossSGD(epoch)  = epochLossSGD/numIterationsPerEpoch;
    lossMOM(epoch)  = epochLossMOM/numIterationsPerEpoch;
    lossADAM(epoch) = epochLossADAM/numIterationsPerEpoch;
    lossCLF(epoch)  = epochLossCLF/numIterationsPerEpoch;

    lambdaCLF(epoch) = epochLambda/numIterationsPerEpoch;

    %% Test Accuracy
    accSGD(epoch)  = evaluateAccuracy(netSGD,XTest,YTest,classes,miniBatchSize);
    accMOM(epoch)  = evaluateAccuracy(netMOM,XTest,YTest,classes,miniBatchSize);
    accADAM(epoch) = evaluateAccuracy(netADAM,XTest,YTest,classes,miniBatchSize);
    accCLF(epoch)  = evaluateAccuracy(netCLF,XTest,YTest,classes,miniBatchSize);

    fprintf('Epoch %02d/%02d | SGD %.2f | MOM %.2f | Adam %.2f | CLF %.2f\n', ...
        epoch,numEpochs,accSGD(epoch),accMOM(epoch),accADAM(epoch),accCLF(epoch));
end

%% ============================================================
% Final Result Table
%% ============================================================

fprintf('\nPeak Test Accuracy on CIFAR-10\n');
fprintf('=====================================\n');
fprintf('SGD          : %.2f %%\n',max(accSGD));
fprintf('SGD-Momentum : %.2f %%\n',max(accMOM));
fprintf('Adam         : %.2f %%\n',max(accADAM));
fprintf('CLF-GD       : %.2f %%\n',max(accCLF));

%% ============================================================
% Plot 1: Test Accuracy
%% ============================================================

figure('Color','w','Position',[100 100 1000 700]);

plot(1:numEpochs,accSGD,'LineWidth',3.5); hold on;
plot(1:numEpochs,accMOM,'LineWidth',3.5);
plot(1:numEpochs,accADAM,'LineWidth',3.5);
plot(1:numEpochs,accCLF,'LineWidth',4.5);

grid on; box on;

xlabel('Epoch','FontSize',28,'FontWeight','bold');
ylabel('Test Accuracy (%)','FontSize',28,'FontWeight','bold');
title('CIFAR-10 Test Accuracy: CNN','FontSize',30,'FontWeight','bold');

legend({'SGD','SGD-Momentum','Adam','CLF-GD'}, ...
    'FontSize',20,'FontWeight','bold','Location','southeast');

set(gca,'FontSize',24,'FontWeight','bold','LineWidth',3);
xlim([1 numEpochs]);
ylim([0 100]);

exportgraphics(gcf,'CIFAR10_Actual_Test_Accuracy.png','Resolution',600);

%% ============================================================
% Plot 2: Training Loss
%% ============================================================

figure('Color','w','Position',[150 150 1000 700]);

plot(1:numEpochs,lossSGD,'LineWidth',3.5); hold on;
plot(1:numEpochs,lossMOM,'LineWidth',3.5);
plot(1:numEpochs,lossADAM,'LineWidth',3.5);
plot(1:numEpochs,lossCLF,'LineWidth',4.5);

grid on; box on;

xlabel('Epoch','FontSize',28,'FontWeight','bold');
ylabel('Training Cross-Entropy Loss','FontSize',28,'FontWeight','bold');
title('CIFAR-10 Training Loss: CNN','FontSize',30,'FontWeight','bold');

legend({'SGD','SGD-Momentum','Adam','CLF-GD'}, ...
    'FontSize',20,'FontWeight','bold','Location','northeast');

set(gca,'FontSize',24,'FontWeight','bold','LineWidth',3);
xlim([1 numEpochs]);

exportgraphics(gcf,'CIFAR10_Actual_Training_Loss.png','Resolution',600);

%% ============================================================
% Plot 3: CLF Adaptive Gain
%% ============================================================

figure('Color','w','Position',[200 200 1000 700]);

plot(1:numEpochs,lambdaCLF,'k','LineWidth',4.5);

grid on; box on;

xlabel('Epoch','FontSize',28,'FontWeight','bold');
ylabel('Average CLF Gain $\lambda_k$','Interpreter','latex','FontSize',28,'FontWeight','bold');
title('CLF-Based Adaptive Gain on CIFAR-10','FontSize',30,'FontWeight','bold');

set(gca,'FontSize',24,'FontWeight','bold','LineWidth',3);
xlim([1 numEpochs]);

exportgraphics(gcf,'CIFAR10_CLF_Adaptive_Gain.png','Resolution',600);

%% ============================================================
% Local Functions
%% ============================================================

function net = createCIFAR10CNN(classes)

numClasses = numel(classes);

layers = [
    imageInputLayer([32 32 3],'Normalization','none','Name','input')

    convolution2dLayer(3,32,'Padding','same','Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2,'Stride',2,'Name','pool1')

    convolution2dLayer(3,64,'Padding','same','Name','conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    maxPooling2dLayer(2,'Stride',2,'Name','pool2')

    convolution2dLayer(3,128,'Padding','same','Name','conv3')
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','relu3')

    fullyConnectedLayer(128,'Name','fc1')
    reluLayer('Name','relu4')

    fullyConnectedLayer(numClasses,'Name','fc_out')
];

net = dlnetwork(layerGraph(layers));

end

function [loss,gradients] = modelGradients(net,dlX,Y,classes)

% Forward pass
dlY = forward(net,dlX);

% Softmax probabilities
P = softmax(dlY);

% Manual one-hot encoding
T = oneHotManual(Y,classes);

% Convert target to dlarray with format C x B
T = dlarray(T,'CB');

% Cross-entropy loss
loss = -mean(sum(T.*log(P + 1e-8),1));

% Gradients
gradients = dlgradient(loss,net.Learnables);

end

function T = oneHotManual(Y,classes)

% Y is categorical batch label vector
% classes is string vector of class names
% Output T is C x B

Ystr = string(Y(:));
classes = string(classes(:));

numClasses = numel(classes);
batchSize = numel(Ystr);

T = zeros(numClasses,batchSize,'single');

for j = 1:batchSize
    classIndex = find(classes == Ystr(j),1);
    T(classIndex,j) = 1;
end

end

function acc = evaluateAccuracy(net,XTest,YTest,classes,miniBatchSize)

numTest = size(XTest,4);
numBatches = ceil(numTest/miniBatchSize);

correct = 0;
total = 0;

for i = 1:numBatches

    idxStart = (i-1)*miniBatchSize + 1;
    idxEnd = min(i*miniBatchSize,numTest);
    idx = idxStart:idxEnd;

    dlX = dlarray(XTest(:,:,:,idx),'SSCB');
    dlY = forward(net,dlX);

    scores = extractdata(softmax(dlY));

    [~,predIdx] = max(scores,[],1);

    predLabels = categorical(string(classes(predIdx)), string(classes));
    trueLabels = YTest(idx);

    correct = correct + sum(predLabels(:) == trueLabels(:));
    total = total + numel(trueLabels);
end

acc = 100*correct/total;

end

function gradNorm2 = gradientNormSquared(gradients)

gradNorm2 = 0;

for i = 1:size(gradients,1)
    g = gradients.Value{i};
    gradNorm2 = gradNorm2 + sum(extractdata(g).^2,'all');
end

end

function Z = initializeLike(T)

Z = T;

for i = 1:size(T,1)
    Z.Value{i} = zeros(size(T.Value{i}),'like',T.Value{i});
end

end

function [params,avgGrad,avgSqGrad] = adamUpdateCustom(params,gradients,avgGrad,avgSqGrad,iteration,lr,beta1,beta2,epsAdam)

for i = 1:size(params,1)

    g = gradients.Value{i};

    avgGrad.Value{i} = beta1*avgGrad.Value{i} + (1-beta1)*g;
    avgSqGrad.Value{i} = beta2*avgSqGrad.Value{i} + (1-beta2)*(g.^2);

    mHat = avgGrad.Value{i}/(1-beta1^iteration);
    vHat = avgSqGrad.Value{i}/(1-beta2^iteration);

    params.Value{i} = params.Value{i} - lr*mHat./(sqrt(vHat)+epsAdam);
end

end

function [XTrain,YTrain,XTest,YTest,classes] = loadCIFAR10(dataFolder)

cifarFolder = fullfile(dataFolder,'cifar-10-batches-mat');

XTrain = [];
YTrainNumeric = [];

for i = 1:5
    batch = load(fullfile(cifarFolder,sprintf('data_batch_%d.mat',i)));

    X = reshape(batch.data',32,32,3,[]);
    X = permute(X,[2 1 3 4]);

    XTrain = cat(4,XTrain,X);
    YTrainNumeric = [YTrainNumeric; batch.labels];
end

testBatch = load(fullfile(cifarFolder,'test_batch.mat'));

XTest = reshape(testBatch.data',32,32,3,[]);
XTest = permute(XTest,[2 1 3 4]);

YTestNumeric = testBatch.labels;

meta = load(fullfile(cifarFolder,'batches.meta.mat'));
classes = string(meta.label_names);

YTrain = categorical(classes(double(YTrainNumeric)+1), classes);
YTest  = categorical(classes(double(YTestNumeric)+1), classes);

end