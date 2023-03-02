% modelEvaluation is used alongside the training process to evaluate training and validation samples.

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

laboratorio.ia@epn.edu.ec

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

Matlab 9.11.0.2022996 (R2021b) Update 4.
%}


%% DEFINE THE DIRECTORIES WHERE THE DATA WILL BE FOUND
dataDir = 'EMG-EPN-612 dataset';
trainingDir = 'trainingJSON';


%% LOAD THE MODEL
modelFileName = "Models\model_s30_e10.mat";
% #################################################################
% #################################################################


%% GET THE USERS DIRECTORIES
[users, trainingPath] = Shared.getUsers(dataDir, trainingDir);
if Shared.includeTesting
    % Divide in two datasets
    limit = length(users)- Shared.numTestUsers;
    usersTrainVal = users(1:limit, 1);
    usersTest = users(limit+1:length(users), 1);
else
    usersTrainVal = users;
end
clear dataDir trainingDir users numTestUsers limit

%% ===== JUST FOR debugging =====
%usersTrainVal = usersTrainVal(1:2);
%usersTest = usersTest(1:2);
%  ===== JUST FOR debugging =====

model = load(modelFileName).net;
clear modelFile modelFileName

%% PREALLOCATE SPACE FOR RESULTS TRAINING AND VALIDATION
% Training
[classifications, recognitions, overlapings, procesingTimes] =  ... 
    preallocateResults(length(usersTrainVal));
% Validation
[classificationsVal, recognitionsVal, overlapingsVal, procesingTimesVal] = ... 
    preallocateResults(length(usersTrainVal));

%% EVALUATE EACH USER FOR TRAINING AND VALIDATION
parfor i = 1:length(usersTrainVal)% parfor
    % Get user samples
    [trainingSamples, validationSamples] = Shared.getTrainingTestingSamples(trainingPath, usersTrainVal(i));
    
    % Transform samples
    transformedSamplesTraining = transformSamples(trainingSamples);
    userResults = evaluateSamples(transformedSamplesTraining, model);
    
    % Set user's training results
    [classifications(i, :), recognitions(i, :), overlapings(i, :), procesingTimes(i, :)] = ... 
        deal(userResults.classifications, userResults.recognitions, ... 
        userResults.overlapings, userResults.procesingTimes);
    
    % Validation data
    transformedSamplesValidation = transformSamples(validationSamples);
    userResults = evaluateSamples(transformedSamplesValidation, model);
    
    % Set user's training results
    [classificationsVal(i, :), recognitionsVal(i, :), overlapingsVal(i, :), procesingTimesVal(i, :)] = ... 
        deal(userResults.classifications, userResults.recognitions, ... 
        userResults.overlapings, userResults.procesingTimes);
end

% Print trainig results
fprintf('\n\n\tTraining data results\n\n');
resultsTrain = calculateValidationResults(classifications, recognitions, overlapings, ... 
    procesingTimes, length(usersTrainVal));

% Print validation results
fprintf('\n\n\tValidation data results\n\n');
resultsValidation = calculateValidationResults(classificationsVal, recognitionsVal, ... 
    overlapingsVal, procesingTimesVal, length(usersTrainVal));

clear i trainingSamples validationSamples transformedSamplesValidation classifications recognitions overlapings procesingTimes classificationsVal recognitionsVal overlapingsVal procesingTimesVal

%% PREALLOCATE SPACE FOR RESULTS TESTING
if Shared.includeTesting
    % Testing - users training samples
    [classificationsTest1, recognitionsTest1, overlapingsTest1, procesingTimesTest1] =  ...
        preallocateResults(length(usersTest));
    % Testing - users validation samples
    [classificationsTest2, recognitionsTest2, overlapingsTest2, procesingTimesTest2] =  ...
        preallocateResults(length(usersTest));
end

%% EVALUATE EACH USER FOR TESTING
if Shared.includeTesting
    parfor i = 1:length(usersTest) %parfor
        % Get user samples
        [trainingSamples, validationSamples] = Shared.getTrainingTestingSamples(trainingPath, usersTest(i));

        % Transform samples
        transformedSamplesTraining = transformSamples(trainingSamples);
        userResults = evaluateSamples(transformedSamplesTraining, model);

        % Set user's training results
        [classificationsTest1(i, :), recognitionsTest1(i, :), overlapingsTest1(i, :) ... 
            , procesingTimesTest1(i, :)] = deal(userResults.classifications, ... 
            userResults.recognitions, userResults.overlapings, userResults.procesingTimes);

        % Validation data
        transformedSamplesValidation = transformSamples(validationSamples);
        userResults = evaluateSamples(transformedSamplesValidation, model);

        % Set user's training results
        [classificationsTest2(i, :), recognitionsTest2(i, :), overlapingsTest2(i, :), ... 
            procesingTimesTest2(i, :)] = deal(userResults.classifications, ... 
            userResults.recognitions, userResults.overlapings, userResults.procesingTimes);
    end

    % Combine testing part (training and validation samples)
    [classificationsTest, recognitionsTest, overlapingsTest, procesingTimesTest] = ... 
        deal([classificationsTest1; classificationsTest2], [recognitionsTest1; recognitionsTest2], ... 
        [overlapingsTest1; overlapingsTest2], [procesingTimesTest1; procesingTimesTest2]);

    % Print the results
    fprintf('\n\n\tTesting data results\n\n');
    dataTest = calculateValidationResults(classificationsTest, recognitionsTest, ... 
        overlapingsTest, procesingTimesTest, length(usersTest));
end
clear i trainingSamples validationSamples transformedSamplesValidation classificationsTest1 recognitionsTest1 overlapingsTest1 procesingTimesTest1 classificationsTest2 recognitionsTest2 overlapingsTest2 procesingTimesTest2n classificationsTest recognitionsTest overlapingsTest procesingTimesTest

%% FUCTION TO PREALLOCATE SPACE FOR VALIDATION LIBRARY RESULT
function [clasifications, recognitions, overlapings, procesingTimes] = preallocateResults(numUsers)
    % Allocate space to save the results
    clasifications = zeros(numUsers, Shared.numSamplesUser);
    recognitions = zeros(numUsers, Shared.numSamplesUser);
    overlapings = zeros(numUsers, Shared.numSamplesUser);
    procesingTimes = zeros(numUsers, Shared.numSamplesUser);
end

%% CREATE SPECTROGRAM DATA
function transformedSamples = transformSamples(samples)
    % Get sample keys
    samplesKeys = fieldnames(samples);
    
    % Allocate space for the results
    transformedSamples = cell(length(samplesKeys), 3);
    
    for i = 1:length(samplesKeys)       
        % Get sample data
        sample = samples.(samplesKeys{i});
        emg = sample.emg;
        gestureName = sample.gestureName;
        
        % Get signal from sample
        signal = Shared.getSignal(emg);
        
        % Adding the transformed data
        transformedSamples{i,1} = signal;
        transformedSamples{i,2} = gestureName;
        if ~isequal(gestureName,'noGesture')
            groundTruth = sample.groundTruth;
            transformedSamples{i,3} = transpose(groundTruth);
        end
    end
end

%% FUCTION TO PREALLOCATE SPACE FOR VALIDATION LIBRARY RESULT
function [clasifications, recognitions, overlapings, procesingTimes] = preallocateUserResults(numObservations)
    % Allocate space to save the results
    clasifications = zeros(numObservations, 1);
    recognitions = -1*ones(numObservations, 1);
    overlapings = -1*ones(numObservations, 1);
    procesingTimes = zeros(numObservations, 1);
end

%% FUNCTION TO EVALUETE SAMPLES OF A USER
function userResults = evaluateSamples(samples, model)

    % Preallocate space for results
    [classifications, recognitions, overlapings, procesingTimes] = preallocateUserResults(length(samples));
    
    % For each sample of a user
    for i = 1:length(samples)
        
        % Get sample data
        emg = samples{i, 1};
        gesture = samples{i, 2};
        groundTruth = samples{i, 3};
        
        % Prepare repetition information
        if ~isequal(gesture,'noGesture')
            repInfo.groundTruth = logical(groundTruth);
        end
        repInfo.gestureName = categorical({gesture}, Shared.setNoGestureUse(true));
        
        % Evaluate a sample with slidding window
        [labels, timestamps, processingTimes] = evaluateSampleFrames(emg, groundTruth, model);

        % Set a class for the sample
        class = Shared.classifyPredictions(labels);
        
        % Postprocess the sample (labels)
        labels = Shared.postprocessSample(labels, char(class));
        % Transform to categorical
        %labels = categorical(labels, Shared.setNoGestureUse(true));
        
        % Prepare response
        response = struct('vectorOfLabels', labels, 'vectorOfTimePoints', timestamps, ... 
            'vectorOfProcessingTimes', processingTimes, 'class', class);
        
        % Send to validation toolbox
        result = evalRecognition(repInfo, response);
        
        % Save results
        classifications(i) = result.classResult;
        if ~isequal(gesture,'noGesture')
            recognitions(i) = result.recogResult;
            overlapings(i) = result.overlappingFactor;
        end
        procesingTimes(i) = mean(processingTimes); %time (frame)
                
        % Set User Results
        userResults = struct('classifications', classifications,  'recognitions', ... 
            recognitions, 'overlapings', overlapings, 'procesingTimes', procesingTimes);
        
    end
end

%% FUNCTION TO CALCULATE THE RESULTS OF A DATASTORE (MEAN USERS)
function [classificationPerUser, recognitionPerUser, overlapingPerUser, processingTimePerUser] = ... 
            calculateMeanUsers(classifications, recognitions, overlapings, procesingTimes, numUsers)

    % Allocate space for results
    [classificationPerUser, recognitionPerUser, overlapingPerUser, processingTimePerUser] = ... 
        deal(zeros(numUsers, 1), zeros(numUsers, 1), zeros(numUsers, 1), zeros(numUsers, 1));
    
    % Calculate results per user
    for i = 1:numUsers
        
        classificationPerUser(i, 1) = sum(classifications(i, :) == 1) / length(classifications(i, :));
        % NoGesture omitted it has value = -1 
        recognitionPerUser(i , 1) = sum(recognitions(i, :) == 1) / ... 
            sum(recognitions(i, :) == 1 | recognitions(i, :) == 0);
        overlapingsUser = overlapings(i, :);
        overlapingPerUser(i, 1) = mean(overlapingsUser(overlapingsUser ~= -1),'omitnan');
        processingTimePerUser(i, 1) = mean(procesingTimes(i, :));
        
    end
end

%% FUNCTION TO CALCULATE THE RESULTS OF A DATASTORE (GLOBAL)
function [globalResps, globalStds] = calculateResultsGlobalMean(all, perUser, numUsers)
    
    % Calculate accuracies 
    accClasification = sum(all.classifications==1) / length(all.classifications);
    % NoGesture omitted it has value = -1 
    accRecognition = sum(all.recognitions==1) / sum(all.recognitions == 1 | all.recognitions == 0); 
    avgOverlapingFactor = mean(all.overlapings(all.overlapings ~= -1), 'omitnan');
    avgProcesingTime = mean(all.procesingTimes);
    
    % Set results
    globalResps = struct('accClasification', accClasification, 'accRecognition', accRecognition, ... 
        'avgOverlapingFactor', avgOverlapingFactor, 'avgProcesingTime', avgProcesingTime);
    
    % Stract data per user
    [classificationPerUser, recognitionPerUser, overlapingPerUser, processingTimePerUser] = deal( ... 
        perUser.classifications, perUser.recognitions, perUser.overlapings, perUser.procesingTimes);
    [stdClassification, stdRecognition, stdOverlaping, stdProcessingTime] = deal(0,0,0,0);
    
    % Calculate standard deviations regarding users means
    for i = 1:numUsers
        
        stdClassification = stdClassification + (classificationPerUser(i,1) - accClasification) ^ 2;
        stdRecognition = stdRecognition + (recognitionPerUser(i, 1) - accRecognition) ^ 2;
        stdOverlaping = stdOverlaping + (overlapingPerUser(i, 1) - avgOverlapingFactor) ^ 2;
        stdProcessingTime = stdProcessingTime + (processingTimePerUser(i, 1) - avgProcesingTime) ^ 2;
        
    end
    
    % Check number of users
    if numUsers > 1
         [stdClassification, stdRecognition, stdOverlaping, stdProcessingTime] = deal( ... 
             stdClassification / (numUsers - 1), stdRecognition / (numUsers - 1), ... 
             stdOverlaping / (numUsers - 1), stdProcessingTime / (numUsers - 1));
    else 
        [stdClassification, stdRecognition, stdOverlaping, stdProcessingTime] = deal(0,0,0,0);
    end
    
    % Set standard deviations
    globalStds = struct('stdClassification', stdClassification, 'stdRecognition', stdRecognition, ... 
        'stdOverlaping', stdOverlaping, 'stdProcessingTime', stdProcessingTime);
end

%% FUNCTION TO CALCULATE THE RESULTS OF A DATASTORE
function results = calculateValidationResults(classifications, recognitions, overlapings, procesingTimes, numUsers)
    
    % Calculate results using the mean values of users results
    [classificationPerUser, recognitionPerUser, overlapingPerUser, processingTimePerUser] = ... 
    calculateMeanUsers(classifications, recognitions, overlapings, procesingTimes, numUsers);
    
    % Print results using mean values
    disp('Results (mean of user results)');
    fprintf('Classification | acc: %f | std: %f  \n', ... 
        mean(classificationPerUser), std(classificationPerUser));
    fprintf('Recognition | acc: %f | std: %f  \n', ... 
        mean(recognitionPerUser), std(recognitionPerUser));
    fprintf('Overlaping | avg: %f | std: %f  \n', ... 
        mean(overlapingPerUser), std(overlapingPerUser));
    fprintf('Processing time | avg: %f | std: %f  \n\n', ... 
        mean(processingTimePerUser), std(processingTimePerUser));
    
    % Flatten samples
    [classifications, recognitions, overlapings, procesingTimes] = ... 
    deal(classifications(:), recognitions(:), overlapings(:), procesingTimes(:));
    
    % Organize data in structs
    all = struct('classifications', classifications, 'recognitions', recognitions, ... 
        'overlapings', overlapings, 'procesingTimes', procesingTimes);
    perUser =  struct('classifications', classificationPerUser, 'recognitions', recognitionPerUser, ... 
        'overlapings', overlapingPerUser, 'procesingTimes', processingTimePerUser);
    
    % Calculate results using a global mean
    [globalResps, globalStds] = calculateResultsGlobalMean(all, perUser, numUsers);
    
    % Print results using global values
    disp('Results (Global results)');
    fprintf('Classification | acc: %f | std: %f  \n', ... 
        globalResps.accClasification, globalStds.stdClassification);
    fprintf('Recognition | acc: %f | std: %f  \n', ... 
        globalResps.accRecognition, globalStds.stdRecognition);
    fprintf('Overlaping | avg: %f | std: %f  \n', ... 
        globalResps.avgOverlapingFactor, globalStds.stdOverlaping);
    fprintf('Processing time | avg: %f | std: %f  \n\n', ... 
        globalResps.avgProcesingTime, globalStds.stdProcessingTime);
    
    % Set results
    results = struct('clasification',  globalResps.accClasification, 'recognition', ... 
        globalResps.accRecognition, 'overlapingFactor', globalResps.avgOverlapingFactor, ... 
        'procesingTime', globalResps.avgProcesingTime);
end

%% FUNCTION TO EVALUATE SAMPLE FRAMES
function [labels, timestamps, processingTimes] = evaluateSampleFrames(signal, groundTruth, model)
    
    % Calculate the number of windows
    numPoints = length(signal);
    if isequal(Shared.FILLING_TYPE_EVAL, 'before')
         
        numWindows = floor((numPoints - (Shared.FRAME_WINDOW / 2)) / Shared.WINDOW_STEP_RECOG) + 1;
        stepLimit = numPoints - floor(Shared.FRAME_WINDOW / 2) + 1;
         
    elseif isequal(Shared.FILLING_TYPE_EVAL, 'none')
        
        numWindows = floor((numPoints - (Shared.FRAME_WINDOW)) / Shared.WINDOW_STEP_RECOG) + 1;
        stepLimit = numPoints - Shared.FRAME_WINDOW + 1;
        
    end
    
    % Preallocate space for the spectrograms
    labels = cell(1, numWindows);
    timestamps = zeros(1, numWindows);
    processingTimes = zeros(1, numWindows);
    
    % Fill before frame classification
    if isequal(Shared.FILLING_TYPE_EVAL, 'before')
        
        % Get a nogesture portion of the sample to use as filling
        if groundTruth
            noGestureInSignal = signal(~groundTruth, :);
            filling = noGestureInSignal(1: floor(Shared.FRAME_WINDOW / 2), :);
        else
            filling = signal(1: floor(Shared.FRAME_WINDOW / 2), :);
        end
        % Combine the sample with the filling
        signal = [signal; filling];
    end
    
    % Start the frame classification
    idx = 1; inicio = 1;
    while inicio <= stepLimit
        % Start timer
        timer = tic;
        
        finish = inicio + Shared.FRAME_WINDOW - 1;
        timestamp = inicio + floor((Shared.FRAME_WINDOW - 1) / 2);
        
        % Get the frame signal
        frameSignal = signal(inicio:finish, :);
        frameSignal = Shared.preprocessSignal(frameSignal);
        
        % Classify the signal
        spectrograms = Shared.generateSpectrograms(frameSignal);
        [predicction, predictedScores] = classify(model,spectrograms);
        
        % Check if the prediction is over the frame classification threshold
        if max(predictedScores) < Shared.FRAME_CLASS_THRESHOLD
            predicction = 'noGesture';
        else
            predicction = char(predicction);
        end
        
        % Stop timer
        processingTime = toc(timer);
        
        % Save sample results
        labels{1, idx} =  predicction;
        timestamps(1, idx) = timestamp;
        processingTimes(1, idx) = processingTime;
        
        % Slide the window
        idx = idx + 1;
        inicio = inicio + Shared.WINDOW_STEP_RECOG;
    end
end

