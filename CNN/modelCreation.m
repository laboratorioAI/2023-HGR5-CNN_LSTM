% modelCreation trains and saves the HGR CNN model.
% In this file is stablished the neural network architecture.
% Models are generated in "Models/" folder.

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

laboratorio.ia@epn.edu.ec

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

Matlab 9.11.0.2022996 (R2021b) Update 4.
%}

% #################################################################
% #################################################################



%% SET DATASTORES PATHS
dataDirTraining = fullfile('Datastores', 'training');
dataDirValidation = fullfile('Datastores', 'validation');
if Shared.includeTesting
    dataDirTesting = fullfile('Datastores', 'testing');
end



% The classes are defined
withNoGesture = true;
classes = Shared.setNoGestureUse(withNoGesture);

% The datastores are created
trainingDatastore = SpectrogramDatastore(dataDirTraining, withNoGesture);
validationDatastore = SpectrogramDatastore(dataDirValidation, withNoGesture);
if Shared.includeTesting
    testingDatastore = SpectrogramDatastore(dataDirTesting, withNoGesture);
end
%dataSample = preview(trainingDatastore);

% Clean up variables
clear dataDirTraining dataDirValidation

%% THE INPUT DIMENSIONS ARE DEFINED
inputSize = trainingDatastore.DataDimensions;

%% DEFINE THE AMOUNT OF DATA
% The amount of data to be used in the creation is specified ]0:1]
trainingDatastore = setDataAmount(trainingDatastore, 1);
validationDatastore = setDataAmount(validationDatastore, 1);
if Shared.includeTesting
    testingDatastore = setDataAmount(testingDatastore, 1);
end

%% THE DATA IS DIVIDED IN TRAINING-VALIDATION-TESTING
% The total data for training-validation-tests is obtained
numTrainingSamples = ['Training samples: ', num2str(trainingDatastore.NumObservations)];
numValidationSamples = ['Validation samples: ', num2str(validationDatastore.NumObservations)];
if Shared.includeTesting
    numTestingSamples = ['Testing samples: ', num2str(testingDatastore.NumObservations)];
    fprintf('\n%s\n%s\n%s\n', numTrainingSamples, numValidationSamples, numTestingSamples);
else
    fprintf('\n%s\n%s\n', numTrainingSamples, numValidationSamples);
end
% Clean up variables
clear numTrainingSamples numValidationSamples numTestingSamples

%% THE NEURAL NETWORK ARCHITECTURE IS DEFINED
numClasses = trainingDatastore.NumClasses;
lgraph = setNeuralNetworkArchitecture(inputSize, numClasses);
analyzeNetwork(lgraph);
% Clean up variables
clear numClasses

%% THE OPTIONS ARE DIFINED
gpuDevice(1);
maxEpochs = 10;%7 10
miniBatchSize = 1024;%1024
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',8, ... %5 8
    'ExecutionEnvironment','gpu', ... %gpu
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'ValidationData', validationDatastore, ...
    'ValidationFrequency',floor(trainingDatastore.NumObservations/ miniBatchSize), ...
    'ValidationPatience',5, ...
    'Plots','training-progress');
% Clean up variables
clear maxEpochs miniBatchSize

%% NETWORK TRAINING
net = trainNetwork(trainingDatastore, lgraph, options);
% Clean up variables
clear options lgraph

%% ACCURACY FOR EACH DATASET
% The accuracy for training-validation-tests is obtained
accTraining = calculateAccuracy(net, trainingDatastore);
accValidation = calculateAccuracy(net, validationDatastore);
if Shared.includeTesting
    accTesting = calculateAccuracy(net, testingDatastore);
end
% The amount of training-validation-tests data is printed
strAccTraining = ['Training accuracy: ', num2str(accTraining)];
strAccValidation = ['Validation accuracy: ', num2str(accValidation)];
if Shared.includeTesting
    strAccTesting = ['Testing accuracy: ', num2str(accTesting)];
    fprintf('\n%s\n%s\n%s\n', strAccTraining, strAccValidation, strAccTesting);
else
    fprintf('\n%s\n%s\n', strAccTraining, strAccValidation);
end

% Clean up variables
clear accTraining accValidation accTesting strAccTraining strAccValidation strAccTesting

%% CONFUSION MATRIX FOR EACH DATASET
calculateConfusionMatrix(net, trainingDatastore, 'training', withNoGesture);
calculateConfusionMatrix(net, validationDatastore, 'validation', withNoGesture);
if Shared.includeTesting
    calculateConfusionMatrix(net, testingDatastore, 'testing', withNoGesture);
end

%% SAVE MODEL
save(['Models/model_', datestr(now,'dd-mm-yyyy_HH-MM-ss')], 'net');

%% FUNCTION TO STABLISH THE NEURAL NETWORK ARCHITECTURE
function lgraph = setNeuralNetworkArchitecture(inputSize, numClasses)
    % Create layer graph
    lgraph = layerGraph();
    
    % Add layer branches
    tempLayers = imageInputLayer(inputSize,"Name","data");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = convolution2dLayer([1 1],18,"Name","Inception_1a-1x1");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1a-3x3_reduce")
        reluLayer("Name","Inception_1a-3x3_relu_reduce")
        convolution2dLayer([3 3],18,"Name","Inception_1a-3x3","Padding",[1 1 1 1])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxPooling2dLayer([3 3],"Name","Inception_1a-pool","Padding",[1 1 1 1])
        convolution2dLayer([1 1],18,"Name","Inception_1a-pool_proj")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1a-5x5_reduce")
        reluLayer("Name","Inception_1a-5x5_relu_reduce_2")
        convolution2dLayer([5 5],18,"Name","Inception_1a-5x5","Padding",[2 2 2 2])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        depthConcatenationLayer(4,"Name","depthcat_1a")
        reluLayer("Name","Inception_1a_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = convolution2dLayer([1 1],18,"Name","Inception_1b-1x1");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1b-3x3_reduce")
        reluLayer("Name","Inception_1b-3x3_relu_reduce")
        convolution2dLayer([3 3],18,"Name","Inception_1b-3x3","Padding",[1 1 1 1])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1b-5x5_reduce")
        reluLayer("Name","Inception_1b-5x5_relu_reduce_2")
        convolution2dLayer([5 5],18,"Name","Inception_1b-5x5","Padding",[2 2 2 2])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxPooling2dLayer([3 3],"Name","Inception_1b-pool","Padding",[1 1 1 1])
        convolution2dLayer([1 1],18,"Name","Inception_1b-pool_proj")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        depthConcatenationLayer(4,"Name","depthcat_1b")
        reluLayer("Name","Inception_1b")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = convolution2dLayer([1 1],18,"Name","Inception_1c-1x1");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1c-3x3_reduce")
        reluLayer("Name","Inception_1c-3x3_relu_reduce")
        convolution2dLayer([3 3],18,"Name","Inception_1c-3x3","Padding",[1 1 1 1])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxPooling2dLayer([3 3],"Name","Inception_1c-pool","Padding",[1 1 1 1])
        convolution2dLayer([1 1],18,"Name","Inception_1c-pool_proj")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1c-5x5_reduce")
        reluLayer("Name","Inception_1c-5x5_relu_reduce_2")
        convolution2dLayer([5 5],18,"Name","Inception_1c-5x5","Padding",[2 2 2 2])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = depthConcatenationLayer(4,"Name","depthcat_1c");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        additionLayer(2,"Name","addition_1ac")
        reluLayer("Name","Inception_1c")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1d-5x5_reduce")
        reluLayer("Name","Inception_1d-5x5_relu_reduce_2")
        convolution2dLayer([5 5],18,"Name","Inception_1d-5x5","Padding",[2 2 2 2])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxPooling2dLayer([3 3],"Name","Inception_1d-pool","Padding",[1 1 1 1])
        convolution2dLayer([1 1],18,"Name","Inception_1d-pool_proj")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = convolution2dLayer([1 1],18,"Name","Inception_1d-1x1");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1d-3x3_reduce")
        reluLayer("Name","Inception_1d-3x3_relu_reduce")
        convolution2dLayer([3 3],18,"Name","Inception_1d-3x3","Padding",[1 1 1 1])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        depthConcatenationLayer(4,"Name","depthcat_1d")
        reluLayer("Name","Inception_1d")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1e-3x3_reduce")
        reluLayer("Name","Inception_1e-3x3_relu_reduce")
        convolution2dLayer([3 3],18,"Name","Inception_1e-3x3","Padding",[1 1 1 1])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = convolution2dLayer([1 1],18,"Name","Inception_1e-1x1");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxPooling2dLayer([3 3],"Name","Inception_1e-pool","Padding",[1 1 1 1])
        convolution2dLayer([1 1],18,"Name","Inception_1e-pool_proj")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1e-5x5_reduce")
        reluLayer("Name","Inception_1e-5x5_relu_reduce_2")
        convolution2dLayer([5 5],18,"Name","Inception_1e-5x5","Padding",[2 2 2 2])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = depthConcatenationLayer(4,"Name","depthcat_1e");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        additionLayer(2,"Name","addition_1ce")
        reluLayer("Name","Inception_1e")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1f-5x5_reduce")
        reluLayer("Name","Inception_1f-5x5_relu_reduce_2")
        convolution2dLayer([5 5],18,"Name","Inception_1f-5x5","Padding",[2 2 2 2])
        reluLayer("Name","Inception_1f-5x5_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1f-3x3_reduce")
        reluLayer("Name","Inception_1f-3x3_relu_reduce")
        convolution2dLayer([3 3],18,"Name","Inception_1f-3x3","Padding",[1 1 1 1])
        reluLayer("Name","Inception_1f-3x3_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxPooling2dLayer([3 3],"Name","Inception_1f-pool","Padding",[1 1 1 1])
        convolution2dLayer([1 1],18,"Name","Inception_1f-pool_proj")
        reluLayer("Name","Inception_1f-relu-pool_proj")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],18,"Name","Inception_1f-1x1")
        reluLayer("Name","Inception_1f-1x1_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        depthConcatenationLayer(4,"Name","depthcat_1f")
        crossChannelNormalizationLayer(5,"Name","crossnorm_1")
        fullyConnectedLayer(numClasses,"Name","fc_1")
        softmaxLayer("Name","softmax")
        classificationLayer("Name","classoutput")];
    lgraph = addLayers(lgraph,tempLayers);

    % clean up helper variable
    lgraph = connectLayers(lgraph,"data","Inception_1a-1x1");
    lgraph = connectLayers(lgraph,"data","Inception_1a-3x3_reduce");
    lgraph = connectLayers(lgraph,"data","Inception_1a-pool");
    lgraph = connectLayers(lgraph,"data","Inception_1a-5x5_reduce");
    lgraph = connectLayers(lgraph,"Inception_1a-1x1","depthcat_1a/in4");
    lgraph = connectLayers(lgraph,"Inception_1a-3x3","depthcat_1a/in1");
    lgraph = connectLayers(lgraph,"Inception_1a-pool_proj","depthcat_1a/in3");
    lgraph = connectLayers(lgraph,"Inception_1a-5x5","depthcat_1a/in2");
    lgraph = connectLayers(lgraph,"Inception_1a_relu","Inception_1b-1x1");
    lgraph = connectLayers(lgraph,"Inception_1a_relu","Inception_1b-3x3_reduce");
    lgraph = connectLayers(lgraph,"Inception_1a_relu","Inception_1b-5x5_reduce");
    lgraph = connectLayers(lgraph,"Inception_1a_relu","Inception_1b-pool");
    lgraph = connectLayers(lgraph,"Inception_1a_relu","addition_1ac/in1");
    lgraph = connectLayers(lgraph,"Inception_1b-1x1","depthcat_1b/in2");
    lgraph = connectLayers(lgraph,"Inception_1b-5x5","depthcat_1b/in1");
    lgraph = connectLayers(lgraph,"Inception_1b-pool_proj","depthcat_1b/in4");
    lgraph = connectLayers(lgraph,"Inception_1b-3x3","depthcat_1b/in3");
    lgraph = connectLayers(lgraph,"Inception_1b","Inception_1c-1x1");
    lgraph = connectLayers(lgraph,"Inception_1b","Inception_1c-3x3_reduce");
    lgraph = connectLayers(lgraph,"Inception_1b","Inception_1c-pool");
    lgraph = connectLayers(lgraph,"Inception_1b","Inception_1c-5x5_reduce");
    lgraph = connectLayers(lgraph,"Inception_1c-1x1","depthcat_1c/in1");
    lgraph = connectLayers(lgraph,"Inception_1c-3x3","depthcat_1c/in2");
    lgraph = connectLayers(lgraph,"Inception_1c-pool_proj","depthcat_1c/in4");
    lgraph = connectLayers(lgraph,"Inception_1c-5x5","depthcat_1c/in3");
    lgraph = connectLayers(lgraph,"depthcat_1c","addition_1ac/in2");
    lgraph = connectLayers(lgraph,"Inception_1c","Inception_1d-5x5_reduce");
    lgraph = connectLayers(lgraph,"Inception_1c","Inception_1d-pool");
    lgraph = connectLayers(lgraph,"Inception_1c","Inception_1d-1x1");
    lgraph = connectLayers(lgraph,"Inception_1c","Inception_1d-3x3_reduce");
    lgraph = connectLayers(lgraph,"Inception_1c","addition_1ce/in2");
    lgraph = connectLayers(lgraph,"Inception_1d-pool_proj","depthcat_1d/in4");
    lgraph = connectLayers(lgraph,"Inception_1d-1x1","depthcat_1d/in1");
    lgraph = connectLayers(lgraph,"Inception_1d-5x5","depthcat_1d/in3");
    lgraph = connectLayers(lgraph,"Inception_1d-3x3","depthcat_1d/in2");
    lgraph = connectLayers(lgraph,"Inception_1d","Inception_1e-3x3_reduce");
    lgraph = connectLayers(lgraph,"Inception_1d","Inception_1e-1x1");
    lgraph = connectLayers(lgraph,"Inception_1d","Inception_1e-pool");
    lgraph = connectLayers(lgraph,"Inception_1d","Inception_1e-5x5_reduce");
    lgraph = connectLayers(lgraph,"Inception_1e-1x1","depthcat_1e/in1");
    lgraph = connectLayers(lgraph,"Inception_1e-5x5","depthcat_1e/in3");
    lgraph = connectLayers(lgraph,"Inception_1e-pool_proj","depthcat_1e/in4");
    lgraph = connectLayers(lgraph,"Inception_1e-3x3","depthcat_1e/in2");
    lgraph = connectLayers(lgraph,"depthcat_1e","addition_1ce/in1");
    lgraph = connectLayers(lgraph,"Inception_1e","Inception_1f-5x5_reduce");
    lgraph = connectLayers(lgraph,"Inception_1e","Inception_1f-3x3_reduce");
    lgraph = connectLayers(lgraph,"Inception_1e","Inception_1f-pool");
    lgraph = connectLayers(lgraph,"Inception_1e","Inception_1f-1x1");
    lgraph = connectLayers(lgraph,"Inception_1f-5x5_relu","depthcat_1f/in3");
    lgraph = connectLayers(lgraph,"Inception_1f-relu-pool_proj","depthcat_1f/in4");
    lgraph = connectLayers(lgraph,"Inception_1f-3x3_relu","depthcat_1f/in2");
    lgraph = connectLayers(lgraph,"Inception_1f-1x1_relu","depthcat_1f/in1");
end

%% FUNCTION TO CALCULATE ACCURACY OF A DATASTORE
function accuracy = calculateAccuracy(net, datastore)
    YPred = classify(net,datastore);
    YValidation = datastore.Labels;
    % Calculate accuracy
    accuracy = sum(YPred == YValidation)/numel(YValidation);
end

%% FUNCTION TO CALCULATE AD PLOT A CONFUSION MATRIX
function calculateConfusionMatrix(net, datastore, datasetName, withNoGesture)
    % Get predictions of each frame
    predLabels = classify(net, datastore);
    realLabels = datastore.Labels;
    
    % Stablish clases
    classes = categorical(Shared.setNoGestureUse(withNoGesture));
    
    % Create the confusion matrix
    confusionMatrix = confusionmat(realLabels, predLabels, 'Order', classes);
    figure('Name', ['Confusion Matrix - ' datasetName])
        matrixChart = confusionchart(confusionMatrix, classes);
        % Chart options
        matrixChart.ColumnSummary = 'column-normalized';
        matrixChart.RowSummary = 'row-normalized';
        matrixChart.Title = ['Hand gestures - ' datasetName];
        sortClasses(matrixChart,classes);
end
