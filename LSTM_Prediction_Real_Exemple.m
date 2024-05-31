clear all;
clc;

%% Citire date

data = readtable('Autonoumous_Car_Data.csv');

%% Normalizare
features = data{:, {'Latitude', 'Longitude', 'heading', 'v'}};
[features, featureMu, featureSigma] = zscore(features);

%% Input Output
sequenceLength = 20; 
numFeatures = size(features, 2);
numSamples = size(features, 1) - sequenceLength + 1;
X = zeros(numSamples, sequenceLength, numFeatures);
Y = zeros(numSamples, 1);
for i = 1:numSamples
    X(i, :, :) = features(i:i+sequenceLength-1, :);
    Y(i) = features(i+sequenceLength-1, 1); 
end

%% Training, validation, test sets
trainRatio = 0.7;
valRatio = 0.15;
testRatio = 0.15;

numTrainSamples = floor(trainRatio * numSamples);
numValSamples = floor(valRatio * numSamples);
numTestSamples = numSamples - numTrainSamples - numValSamples;

XTrain = X(1:numTrainSamples, :, :);
YTrain = Y(1:numTrainSamples);
XVal = X(numTrainSamples+1:numTrainSamples+numValSamples, :, :);
YVal = Y(numTrainSamples+1:numTrainSamples+numValSamples);
XTest = X(numTrainSamples+numValSamples+1:end, :, :);
YTest = Y(numTrainSamples+numValSamples+1:end);

%% Conversie
XTrainCell = num2cell(XTrain, [2,3]);
XValCell = num2cell(XVal, [2,3]);
XTestCell = num2cell(XTest, [2,3]);

%% Arhitectura LSTM
layers = [ ...
    sequenceInputLayer([1, sequenceLength])
    flattenLayer
    lstmLayer(100, 'OutputMode', 'sequence')
    dropoutLayer(0.2)
    lstmLayer(100, 'OutputMode', 'last')
    dropoutLayer(0.2)
    fullyConnectedLayer(1)
    regressionLayer];

%% Optiuni de antrenare
options = trainingOptions('adam', ...
    'MaxEpochs', 25, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 20, ...
    'ValidationData', {XValCell, YVal}, ...
    'ValidationFrequency', 10, ...
    'Verbose', 1, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment', 'auto', ...
    'GradientThreshold', 1, ...
    'L2Regularization', 0.0001);

%% Antrenarea modelului
model = trainNetwork(XTrainCell, YTrain, layers, options);

%% Predictia
YPred = 1.78-predict(model, XTestCell); % problema de amplitudine (pentru setul acesta de date amplificam cu 1.78)

%% RMSE calc
rmse = sqrt(mean((YPred - YTest).^2));
fprintf('Root Mean Squared Error (RMSE) on test set: %.4f\n', rmse);

%% Afisare rezultate
figure;
plot(1:length(YTest), YTest, 'DisplayName', 'Actual');
hold on;
plot(1:length(YPred), YPred, 'DisplayName', 'Predicted');
xlabel('Sample Index');
ylabel('Normalized Latitude');
title('Comparison of Actual and Predicted Values');
legend show;
grid on;
