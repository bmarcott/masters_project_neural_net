%                 
% neural_data format:
%   {
%     1: E_fit,
%     2  E_def, 
%     3  img,
%     4  control points, 
%     5  num_beads, 
%     6  bead variance, 
%     7  A
%   }
%
%
%
%
%
%

inputs = zeros(size(neural_data,1), 6);
load('neural_data');
for imageNdx = 1 : size(neural_data,1)
    
    A = neural_data{imageNdx}{7};
   
    % Calculate transformation amounts
    theta1 = atan(A(2,1)/A(1,1));
    s1 = A(1,1) / cos(theta1);
    theta2 = atan(-A(1,2)/A(2,2));
    s2 = A(2,2) / cos(theta2);
    
    % calculate E_w (whitespace energy)
    %   penalizes control points far from inked pixels
    I  = neural_data{imageNdx}{3};
    [inkedRows, inkedCols] = find(I == 1);
    inkedPixel_coords = [inkedRows, inkedCols];
    [all_beads_coords, ~] = compute_bead_locs(neural_data{imageNdx}{4}', neural_data{imageNdx}{5});
    bead_variance = neural_data{imageNdx}{6} * eye(2) * 10^4; % boost since model doesn't fit well
    E_w = 0;
    for b = 1 : neural_data{imageNdx}{5}
        bead_coords = all_beads_coords(:,b)';
        E_w = E_w + log(sum(mvnpdf(inkedPixel_coords, bead_coords, bead_variance)));
    end
    

    E_fit = neural_data{imageNdx}{1};
    E_def = neural_data{imageNdx}{2};
    E_w = -E_w;
    shear = sin(theta1 - theta2)^2;
    orientation = sin(theta2)^2;
    elongation = s2/s1;
    inputs(imageNdx,:) = [E_fit, E_def, E_w, shear, orientation, elongation];

end

% normalize each models E_fit by simply subracting the min
minVal = min(inputs(1:size(inputs,1)/2,1));
inputs(1:size(inputs,1)/2,1) = inputs(1:size(inputs,1)/2,1) - minVal;

minVal = min(inputs(size(inputs,1)/2+1:end,1));
inputs(size(inputs,1)/2+1:end,1) = inputs(size(inputs,1)/2+1:end,1) - minVal;
outputs = cat(1, repmat([1,0],floor(size(inputs,1)/2),1), repmat([0,1],floor(size(inputs,1)/2),1));

%% Neural Net Training, Validation, and Testing
numToTrain = 10;
numToValidate = 646;
numToTest = 645;

% Train
modelAndErrors = {};

for numHidden = 4 : 4 : 32
    for actFn = {'softmax'}
    net = mlp(6, numHidden, 2, actFn{1});
    net = mlpinit(net, 10);
    [net, error] = mlptrain(net, inputs(1:numToTrain,:), outputs(1:numToTrain,:), 100);
    modelAndErrors = cat(1, modelAndErrors, {net, error});
    end
end

% Validate
modelValidationErrors = {};
for i = 1 : size(modelAndErrors,1)
    net = modelAndErrors{i,1};
    x = inputs(numToTrain+1:numToTrain+numToValidate,:);
    t = outputs(numToTrain+1:numToTrain+numToValidate,:);
    error = mlperr(net, x, t);
    modelValidationErrors = cat(1, modelValidationErrors, {modelAndErrors{i,1}, error});
end
[~, idx] = min([modelValidationErrors{:,2}]);
bestModel = modelValidationErrors{idx,1};

% Test
x = inputs(numToTrain+numToValidate+1:numToTrain+numToValidate+numToTest,:);
t = outputs(numToTrain+numToValidate+1:numToTrain+numToValidate+numToTest,:);
error = mlperr(bestModel, x, t);

estimates = round(mlpfwd(bestModel, x));
errorRate = sum(estimates ~= t) / size(t, 1);





