clear all;
clc;

%% Citire date

data = readtable('Autonoumous_Car_Data.csv');

%% Netezarea coordonatelor de test
data.Latitude = movmean(data.Latitude, 5);
data.Longitude = movmean(data.Longitude, 5);

%% Afisare traiectorie

figure; 
plot(data.Longitude, data.Latitude, '-o', 'MarkerSize', 5, 'MarkerFaceColor', 'b');
title('2D Trajectory Plot');
xlabel('Longitude');
ylabel('Latitude');
grid on; 

%% Distanta dinrtre 2 puncte consecutive 

R = 6371; 
for i = 1:(height(data)-1)
    lat1 = deg2rad(data.Latitude(i));
    lat2 = deg2rad(data.Latitude(i+1));
    lon1 = deg2rad(data.Longitude(i));
    lon2 = deg2rad(data.Longitude(i+1));
    dlat = lat2 - lat1;
    dlon = lon2 - lon1;
    a = sin(dlat/2)^2 + cos(lat1) * cos(lat2) * sin(dlon/2)^2;
    c = 2 * atan2(sqrt(a), sqrt(1-a));
    data.Distance(i) = R * c;
end
data.Distance(end) = 0; 

%% Normalize features 

features = data{:, {'Latitude', 'Longitude', 'heading', 'v'}};
target = data.Distance;
[features, featureMu, featureSigma] = zscore(features);
target = (target - mean(target)) / std(target);

%% Input Output

sequenceLength = 20; 
numFeatures = size(features, 2);
numSamples = size(features, 1) - sequenceLength + 1;
X = zeros(numSamples, sequenceLength, numFeatures);
Y = zeros(numSamples, 1);
for i = 1:numSamples
    X(i, :, :) = features(i:i+sequenceLength-1, :);
    Y(i) = target(i+sequenceLength-1);
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

XTrainCell = cell(size(XTrain, 1), 1);
for i = 1:size(XTrain, 1)
    XTrainCell{i} = squeeze(XTrain(i, :, :))';
end

XValCell = cell(size(XVal, 1), 1);
for i = 1:size(XVal, 1)
    XValCell{i} = squeeze(XVal(i, :, :))';
end

XTestCell = cell(size(XTest, 1), 1);
for i = 1:size(XTest, 1)
    XTestCell{i} = squeeze(XTest(i, :, :))';
end

%% Arhitectura retelei LSTM

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(100, 'OutputMode', 'sequence')
    lstmLayer(100, 'OutputMode', 'last')
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
    'ExecutionEnvironment', 'auto', ...
    'Shuffle', 'every-epoch');

%% Antrenarea modelului

model = trainNetwork(XTrainCell, YTrain, layers, options);

%% Predictie

YPred = predict(model, XTestCell);

%% Latitudine , Logitudine Calcul

actualLat = data.Latitude(numTrainSamples + numValSamples + 1:end);
actualLon = data.Longitude(numTrainSamples + numValSamples + 1:end);

%% Coordonatele

actualLatLong = [actualLat, actualLon];

%% Conversie

predictedLatLong = zeros(size(actualLatLong));
for i = 1:size(YPred, 1)
    predictedLatLong(i, :) = actualLatLong(i, :) + YPred(i);
end

%% Eroarea calc

errors = actualLatLong - predictedLatLong;

%% Plot the CDF of prediction errors for latitude

[f_lat, x_lat] = ecdf(errors(:, 1));
figure;
plot(x_lat, f_lat, 'DisplayName', 'Latitude Errors');
xlabel('Prediction Error');
ylabel('CDF');
title('CDF of Latitude Prediction Errors');
legend show;
grid on;

%% Plot the CDF of prediction errors for longitude

[f_lon, x_lon] = ecdf(errors(:, 2));
figure;
plot(x_lon, f_lon, 'DisplayName', 'Longitude Errors');
xlabel('Prediction Error');
ylabel('CDF');
title('CDF of Longitude Prediction Errors');
legend show;
grid on;

%% RMSE

rmse = sqrt(mean(errors.^2, 'all')); 
fprintf('Root Mean Squared Error (RMSE) on test set: %f\n', rmse);
