% spectrogramDatasetGeneration creates the spectrogram datastores for the CNN model.
% Files are generated in "Datastores/" folder.

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
% Clean up variables
clear dataDir trainingDir users numTestUsers limit

%% ===== JUST FOR DEBUGGING =====
%usersTrainVal = usersTrainVal(1:1);
%usersTest = usersTest(1:1);
%  ===== JUST FOR DEBUGGING =====

%% THE STRUCTURE OF THE DATASTORE IS DEFINED
categories = {'fist'; 'open'; 'pinch'; 'waveIn'; 'waveOut'};
trainingDatastore = createDatastore('Datastores/training', categories);
validationDatastore = createDatastore('Datastores/validation', categories);
if Shared.includeTesting
    testingDatastore = createDatastore('Datastores/testing', categories);
end
% Clean up variables
clear categories

%% GENERATION OF SPECTROGRAMS TO CREATE THE MODEL
if Shared.includeTesting
    usersSets = {usersTrainVal, 'usersTrainVal'; usersTest, 'usersTest'};
else
    usersSets = {usersTrainVal, 'usersTrainVal'};
end

% For each user set (trainVal and test)
for i = 1:size(usersSets, 1)
    
    % Select a set of users
    users = usersSets{i,1};
    usersSet = usersSets{i,2};
    
    % Set datastores
    if isequal(usersSet, 'usersTrainVal')
        [datastore1, datastore2] = deal(trainingDatastore, validationDatastore);
    elseif isequal(usersSet, 'usersTest')
        [datastore1, datastore2] = deal(testingDatastore, testingDatastore);
    end
        
    parfor j = 1:length(users)
        s = sprintf('Usuario: %d / %d\n', j, length(users));
        fprintf('%s', s)

        % Get user samples
        [trainingSamples, validationSamples] = Shared.getTrainingTestingSamples(trainingPath, users(j));
        
        % Transform samples
        transformedSamplesTraining = generateData(trainingSamples);
        transformedSamplesValidation = generateData(validationSamples);
       
        % Save samples
        saveSampleInDatastore(transformedSamplesTraining, users(j), 'train', datastore1);
        saveSampleInDatastore(transformedSamplesValidation, users(j), 'validation', datastore2);

        fprintf(repmat('\b', 1, numel(s)));
    end
end
% Clean up variables
clear i j categories validationSamples transformedSamplesValidation users usersSet datastore1 datastore2

%% INCLUDE NOGESTURE
% Define the directories where the frames will be found
if Shared.includeTesting
    datastores = {trainingDatastore; validationDatastore; testingDatastore};
    usersInDatastore = {length(usersTrainVal); length(usersTrainVal); length(usersTest)};
else
    datastores = {trainingDatastore; validationDatastore};
    usersInDatastore = {length(usersTrainVal); length(usersTrainVal)};
end
noGesturePerUser = cell(length(datastores), 1);
% Clean up variables
clear trainingSamples transformedSamplesTraining trainingDatastore validationDatastore testingDatastore

%% CALCULATE THE MIN NUMBER OF FRAMES FOR EACH DATASTORE
parfor i = 1:length(datastores)
    % Create a file datastore.
    fds = fileDatastore(datastores{i,1}, ...
        'ReadFcn',@Shared.readFile, ...
        'IncludeSubfolders',true);
    % Create labels
    labels = Shared.createLabels(fds.Files, false);
    % Get the mininum number of frames for all category
    catCounts = sort(countcats(labels));
    minNumber = catCounts(1);
    % Generate noGesture frames
    noGesturePerUser{i,1} = ceil(minNumber / usersInDatastore{i,1});
end
% Clean up variables
clear i labels fds catCounts minNumber

%% THE STRUCTURE OF THE DATASTORE IS DEFINED
categories = {'noGesture'};
trainingDatastore = createDatastore(datastores{1,1}, categories);
validationDatastore = createDatastore(datastores{2,1}, categories);
if Shared.includeTesting
    testingDatastore = createDatastore(datastores{3,1}, categories);
end
clear categories datastores

%% GENERATION OF NOGESTURE SPECTROGRAMS TO CREATE THE MODEL

% Get the number of noGesture per dataset
noGestureTraining = noGesturePerUser{1,1};
noGestureValidation = noGesturePerUser{2,1};
if Shared.includeTesting
    noGestureTesting = ceil(noGesturePerUser{3,1} / 2);
end

for i = 1:size(usersSets, 1)
    % Select a set of users
    users = usersSets{i,1};
    usersSet = usersSets{i,2};
    
    % Set noGesture size and datastores
    if isequal(usersSet, 'usersTrainVal')            
        [noGestureSize1 ,noGestureSize2, datastore1, datastore2] = deal(noGestureTraining, ... 
                noGestureValidation, trainingDatastore, validationDatastore);
    elseif isequal(usersSet, 'usersTest')
        [noGestureSize1 ,noGestureSize2, datastore1, datastore2] = deal(noGestureTesting, ... 
            noGestureTesting, testingDatastore, testingDatastore);
    end
    
    parfor j = 1:length(users)
        % Get user samples
        [trainingSamples, validationSamples] = Shared.getTrainingTestingSamples(trainingPath, users(j));
        
        % Transform samples
        transformedSamplesTraining = generateDataNoGesture(trainingSamples, noGestureSize1);
        transformedSamplesValidation = generateDataNoGesture(validationSamples, noGestureSize2);

        % Save samples
        saveSampleInDatastore(transformedSamplesTraining, users(j), 'validation',datastore1);
        saveSampleInDatastore(transformedSamplesValidation, users(j), 'train', datastore2);
    end
end

% Clean up variables
clear i j validationSamples transformedSamplesValidation trainingDatastore validationDatastore 
clear testingDatastore users usersSet noGestureTraining noGestureValidation noGestureTesting

%% FUNCTION TO CREATE DATASTORE
function datastore = createDatastore(datastore, labels)
    if ~exist(datastore, 'dir')
       mkdir(datastore)
    end
    % One folder is created for each class
    for i = 1:length(labels)
        path = fullfile(datastore, char(labels(i)));
        if ~exist(path, 'dir')
             mkdir(path);
        end
    end
end

%% FUNCTION TO GENERATE FRAMES
function [data] = generateFrames(signal, groundTruth, numGesturePoints)
    
    % Allocate space for the results
    numWindows = floor((length(signal)-Shared.FRAME_WINDOW) /Shared.WINDOW_STEP)+1;
    data = cell(numWindows, 2);
    
    % Creating frames
    for i = 1:numWindows
        
        % Get signal data to create a frame
        traslation = ((i-1)* Shared.WINDOW_STEP);
        inicio = 1 + traslation;
        finish = Shared.FRAME_WINDOW + traslation;
        timestamp = inicio + floor(Shared.FRAME_WINDOW / 2);
        frameGroundTruth = groundTruth(inicio:finish);
        totalOnes = sum(frameGroundTruth == 1);
        
        % Check the thresahold to include the frame or discard it
        if totalOnes >= Shared.FRAME_WINDOW * Shared.TOLERANCE_WINDOW || ...
                totalOnes >= numGesturePoints * Shared.TOLERNCE_GESTURE
            % Get Spectrogram of the window
            frameSignal = signal(inicio:finish, :);
            spectrograms = Shared.generateSpectrograms(frameSignal);
            % Save data
            data{i,1} = spectrograms; % datum
            data{i,2} = timestamp; % time
        end
        
        % Filter to get the gesture frames and discard the noGestures
        idx = cellfun(@(x) ~isempty(x), data(:,1));
        data = data(idx,:);
    end  
end

%% FUCTION TO GENERATE THE DATA
function transformedSamples = generateData(samples)

    % Number of noGesture samples to discard them
    noGesturePerUser = Shared.numGestureRepetitions;
    
    % Allocate space for the results
    samplesKeys = fieldnames(samples);
    transformedSamples = cell(length(samplesKeys)- noGesturePerUser, 3);
    
    % For each gesture sample
    for i = noGesturePerUser + 1:length(samplesKeys)
        
        % Get sample data
        sample = samples.(samplesKeys{i});
        emg = sample.emg;
        gestureName = sample.gestureName;
        groundTruth = sample.groundTruth;
        numGesturePoints = sample.groundTruthIndex(2) - sample.groundTruthIndex(1);
        
        % Get signal from sample
        signal = Shared.getSignal(emg);
        signal = Shared.preprocessSignal(signal);
        
        % Generate spectrograms
        data = generateFrames(signal, groundTruth, numGesturePoints);
        
        % Save the transformed data
        transformedSamples{i - noGesturePerUser, 1} = data;
        transformedSamples{i - noGesturePerUser, 2} = gestureName;
        transformedSamples{i - noGesturePerUser, 3} = transpose(groundTruth);
    end
end

%% FUNCTION TO GENERATE NO GESTURE FRAMES
function data = generateFramesNoGesture(signal, numWindows)
    % Allocate space for the results
    data = cell(numWindows, 2);
    
    % For each window
    for i = 1:numWindows
        % Get window information
        traslation = ((i-1) * Shared.WINDOW_STEP) + 100; %displacement included
        inicio = 1 + traslation;
        finish = Shared.FRAME_WINDOW + traslation;
        timestamp = inicio + floor(Shared.FRAME_WINDOW / 2);
        
        % Generate a spectrogram
        frameSignal = signal(inicio:finish, :);
        spectrograms = Shared.generateSpectrograms(frameSignal);
        
        % Save data
        data{i,1} = spectrograms; % datum
        data{i,2} = timestamp; % label
    end  
end

%% FUNCTION TO GENERATE DATA OF NOGESTURE
function transformedSamples = generateDataNoGesture(samples, totalFrames)
    % Number of noGesture samples to use them
    noGesturePerUser = Shared.numGestureRepetitions;
    
    % Allocate space for the results
    samplesKeys = fieldnames(samples);
    transformedSamples = cell(noGesturePerUser, 2);
    
    % Calculate the number of frames needed per sample
    framesPerSample = ceil(totalFrames / noGesturePerUser);
    
    for i = 1:noGesturePerUser
        
        % Get sample data
        sample = samples.(samplesKeys{i});
        emg = sample.emg;
        gestureName = sample.gestureName;
        
        % Get signal from sample
        signal = Shared.getSignal(emg);
        signal = Shared.preprocessSignal(signal);
        
        % Generate spectrograms
        data = generateFramesNoGesture(signal, framesPerSample);
        
        % Save the transformed data
        transformedSamples{i,1} = data;
        transformedSamples{i,2} = gestureName;
    end
end

%% FUNCTION TO SAVE SPECTROGRAMS IN DATASTORE
function saveSampleInDatastore(samples, user, type, dataStore)
    
    % For each sample
    for i = 1:length(samples)
        
        % Get data from trabnsformed samples
        frames = samples{i,1};
        class = samples{i,2};
        
        % Get data in frames
        spectrograms = frames(:,1);
        timestamps = frames(:,2);
        
        % Save each frame
        for j = 1:length(spectrograms)
            % Set data
            data = spectrograms{j, 1};
            
            % Create a file name (user-type-sample-start-finish)
            start = floor(timestamps{j,1} - Shared.FRAME_WINDOW/2);
            finish = floor(timestamps{j,1} + Shared.FRAME_WINDOW/2);
            fileName = strcat(strtrim(user),'-', type, '-',int2str(i), '-', ...
                '[',int2str(start), '-', int2str(finish), ']');
            
            % Save in tthe folder which  corresponds to the class 
            savePath = fullfile(dataStore, char(class), fileName);
            save(savePath,'data');
        end
    end
end
