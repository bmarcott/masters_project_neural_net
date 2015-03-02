trainingImagePath = 'train-images.idx3-ubyte';
trainingLabelPath = 'train-labels.idx1-ubyte';
testingImagePath = 't10k-images.idx3-ubyte';
testingLabelPath = 't10k-labels.idx1-ubyte';
numInTrainFile = 60000;
numInTestFile = 10000;
offset = 0;
% total 2s and 3s = 3231
numToTrain = 1940;
numToValidate = 646;
numToTest = 645;

% Generate inputs and targets
[imgs, labels] = readMNIST(trainingImagePath, trainingLabelPath, numInTrainFile, offset);
[testImgs, testLabels] = readMNIST(testingImagePath, testingLabelPath, numInTestFile, offset);
imgs = cat(3, imgs, testImgs);
labels = cat(1, labels, testLabels);

imgs = squeeze(reshape(imgs, 400, 1, size(imgs,3)));
imgs = imgs(:, labels == 2 | labels == 3)';
labels = labels(labels == 2 | labels == 3);

outputs = zeros(numel(labels), 2);
for i = 1 : numel(labels)
    if (labels(i) == 2)
        outputs(i,:) = [1, 0];
    else
        outputs(i,:) = [0, 1];
    end
end

%datwrite('twoAndThreeData.dat', imgs, outputs);


% Set up network parameters and options



% Train
modelAndErrors = {};

for numHidden = 4 : 4 : 56
    for actFn = {'logistic'}
    net = mlp(400, numHidden, 2, actFn{1});
    net = mlpinit(net, 10);
    [net, error] = mlptrain(net, imgs(1:numToTrain,:), outputs(1:numToTrain,:), 100);
    modelAndErrors = cat(1, modelAndErrors, {net, error});
    end
end

% Validate
modelValidationErrors = {};
for i = 1 : size(modelAndErrors,1)
    net = modelAndErrors{i,1};
    x = imgs(numToTrain+1:numToTrain+numToValidate,:);
    t = outputs(numToTrain+1:numToTrain+numToValidate,:);
    error = mlperr(net, x, t);
    modelValidationErrors = cat(1, modelValidationErrors, {modelAndErrors{i,1}, error});
end
[~, idx] = min([modelValidationErrors{:,2}]);
bestModel = modelValidationErrors{idx,1};

% Test
x = imgs(numToTrain+numToValidate+1:numToTrain+numToValidate+numToTest,:);
t = outputs(numToTrain+numToValidate+1:numToTrain+numToValidate+numToTest,:);
error = mlperr(bestModel, x, t);

estimates = round(mlpfwd(bestModel, x));
errorRate = sum(estimates ~= t) / size(t, 1);
% type     nHid     errorRate
% linear   32       .0465
% linear   24       .0434
% linear   36       .0450

% logistic 4        .0434
% logistic 4        .0481
% logistic 8        .0434

% softmax  4        .0527
% softmax  4        .0388
% softmax  4        .0434






% Plot