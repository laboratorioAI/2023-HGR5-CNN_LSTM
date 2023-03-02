% testDataEvaluation runs the evaluation process of the CNN model required by the EMG-EPN-612 dataset.
% It creates the responses.json file required to submission in the public evaluator platform.
% ``responses`` files are generated in "CNN/Test-Data/" folder.
% Temporal files are generated in "CNN/Test-Data/user-Results/" folder.

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
dataDir = 'EMG-EPN612 Dataset';
testingDir = 'testingJSON';
modelFileName = "Models\model_s30_e10.mat";

% #################################################################
% #################################################################


%% GET THE USERS DIRECTORIES
[users, testingPath] = Shared.getUsers(dataDir, testingDir);
clear dataDir trainingDir

%% ===== JUST FOR debugging =====
%users = users(1:1);
%  ===== JUST FOR debugging =====


model = load(modelFileName).net;
clear modelFile modelFileName

%% EVALUATE EACH USER FOR TRAINING AND VALIDATION
tTotal = tic();
parfor i = 1:length(users)% parfor
    s = sprintf('Usuario: %d / %d\n', i, length(users));
    fprintf('%s', s)

    % Get user samples
    [~, testingSamples] = Shared.getTrainingTestingSamples(testingPath, users(i));

    % Transform samples
    transformedSamplesTesting = transformSamples(testingSamples);

    % Obtain user's results
    userResults = evaluateSamples(transformedSamplesTesting, model);

    % Save users results
    saveResults(userResults, users(i));
    fprintf(repmat('\b', 1, numel(s)));
end
fprintf('Finalizada evaluación de usuarios\n')
fprintf('Tiempo de parfor %.2f [segundos]\n', toc(tTotal))

fprintf('Creando json files\n')
%% CREATE JSON FILE
resultsPath = fullfile('CNN', 'Test-Data', 'User-Results');
filesInFolder = dir(resultsPath);
numFiles = length(filesInFolder);

for users = 4:numFiles

    if ~(strcmpi(filesInFolder(users).name, '.') || strcmpi(filesInFolder(users).name, '..') ||  ...
            strcmpi(filesInFolder(users).name, 'README.md'))

        % Create the name to save in the new format
        userName = filesInFolder(users).name;
        data = load(fullfile(resultsPath, filesInFolder(users).name));
        userResults = data.userResults;
        newStr = erase(filesInFolder(users).name,'results-test-user');
        newName =['user',newStr];
        newName = erase(newName,'.mat');

        % For each sample
        for i=1:150
            sample = userResults(i);
            c = size(userResults(i).vectorOfLabels, 2);
            % Set data
            response.testing.(newName).class{i,1} = userResults(i).class;
            response.testing.(newName).vectorOfLabels{i,1} = sample.vectorOfLabels;
            response.testing.(newName).vectorOfTimePoints{i,1} = sample.vectorOfTimePoints;
            response.testing.(newName).vectorOfProcessingTime{i,1} = sample.vectorOfProcessingTimes;
        end
    end
end

% Save the transformed data in a matfile
fileName = 'responses.mat';
filePath = fullfile('CNN', 'Test-Data', fileName);
save(filePath);

% Create the JSON file
jsonName = 'responses.json';
jsonFormat(fileName, jsonName);

% Clear variables
clearvars -except response userList

fprintf('Listo!\n')
%% CREATE SPECTROGRAM DATA
function transformedSamples = transformSamples(samples)
% Get sample keys
samplesKeys = fieldnames(samples);

% Allocate space for the results
transformedSamples = cell(length(samplesKeys), 1);

for i = 1:length(samplesKeys)
    % Get sample data
    sample = samples.(samplesKeys{i});
    emg = sample.emg;

    % Get signal from sample
    signal = Shared.getSignal(emg);

    % Adding the transformed data
    transformedSamples{i,1} = signal;
end
end

%% FUNCTION TO EVALUATE SAMPLE FRAMES (Se puede pasar)
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
timestamps = zeros(1,numWindows);
processingTimes = zeros(1,numWindows);

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



%% FUNCTION TO EVALUETE SAMPLES OF A USER
function userResults = evaluateSamples(samples, model)
% Preallocate space for results
userResults = struct;

% For each sample of a user
for i = 1:length(samples)
    s = sprintf('rep: %d / %d', i, length(samples));
    fprintf('%s', s)

    % Get sample data
    emg = samples{i, 1};

    % Evaluate a sample with slidding window
    [labels, timestamps, processingTimes] = evaluateSampleFrames(emg, [], model);

    % Set a class for the sample
    class = Shared.classifyPredictions(labels);

    % Postprocess the sample (labels)
    labels = Shared.postprocessSample(labels, char(class));
    % Transform to categorical
    %labels = categorical(labels, Shared.setNoGestureUse(true));

    % Prepare response
    userResults(i).class = categorical(class, Shared.setNoGestureUse(true));
    userResults(i).vectorOfTimePoints = timestamps;
    userResults(i).vectorOfProcessingTimes = processingTimes;
    userResults(i).vectorOfLabels = categorical(labels, Shared.setNoGestureUse(true));

    fprintf(repmat('\b',1, numel(s)));
end
end

%% FUNCTION TO SAVE SPECTROGRAMS IN DATASTORE
function saveResults(userResults, user)
% Save each frame
fileName = strcat('results-test-', strtrim(user));

% Save in tthe folder which  corresponds to the class
savePath = fullfile('CNN', 'Test-Data', 'User-Results', fileName);
save(savePath,'userResults');
end

%% FUNCTION TO CREATE JSON FILE
function jsonFormat(fileName,jsonName)
% Load the data file
filePath = fullfile('CNN', 'Test-Data', fileName);
jsonPath = fullfile('CNN', 'Test-Data', jsonName);
data = load(filePath);
usersResults = data.response.testing;

% For each user
userList = fieldnames(usersResults);
for i = 1:size(userList ,1)
    % For each sample
    for j = 0:Shared.numSamplesUser - 1

        sample = strcat('idx_', int2str(j));
        % Set results
        results.testing.(userList{i}).class.(sample) =  ...
            usersResults.(userList{i}).class{j+1,1} ;
        results.testing.(userList{i}).vectorOfLabels.(sample) = ...
            usersResults.(userList{i}).vectorOfLabels{j+1,1} ;
        results.testing.(userList{i}).vectorOfTimePoints.(sample) = ...
            usersResults.(userList{i}).vectorOfTimePoints{j+1,1} ;
        results.testing.(userList{i}).vectorOfProcessingTime.(sample) = ...
            usersResults.(userList{i}).vectorOfProcessingTime{j+1,1};

    end
end

% Save JSON in file
jsonData = jsonencode(results);
fid = fopen(jsonPath, 'wt');
fprintf(fid,jsonData);
fclose(fid);
end
