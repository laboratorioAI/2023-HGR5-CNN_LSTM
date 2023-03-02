% spectrogramDatasetGeneration creates the spectrogram datastores for the CNN-LSTM model.
% Files are generated in "DatastoresLSTM/" folder.

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
% usersTrainVal = usersTrainVal(1:1);
% usersTest = usersTest(1:1);
%  ===== JUST FOR DEBUGGING =====

%% THE STRUCTURE OF THE DATASTORE IS DEFINED
categories = {'fist'; 'open'; 'pinch'; 'waveIn'; 'waveOut'};
trainingDatastore = createDatastore('DatastoresLSTM/training', categories);
validationDatastore = createDatastore('DatastoresLSTM/validation', categories);
if Shared.includeTesting
    testingDatastore = createDatastore('DatastoresLSTM/testing', categories);
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
    
    fprintf('User set %s\n', usersSet)

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
clear i j categories validationSamples transformedSamplesValidation users usersSet usersTrainVal usersTest datastore1 datastore2

%% INCLUDE NOGESTURE
% Define the directories where the sequences will be added
if Shared.includeTesting
    datastores = {trainingDatastore; validationDatastore; testingDatastore};
else
    datastores = {trainingDatastore; validationDatastore};
end
noGestureFramesPerSample = cell(length(datastores), 1);
% Clean up variables
clear trainingSamples transformedSamplesTraining trainingDatastore validationDatastore testingDatastore

%% CALCULATE THE NUMBER OF FRAMES IN A SEQUENCE FOR EACH DATASTORE
parfor i = 1:length(datastores)
    
    % Create a file datastore.
    fds = fileDatastore(datastores{i,1}, ...
        'ReadFcn',@Shared.readFile, ...
        'IncludeSubfolders',true);
    
    % Check the type of filling
    if isequal(Shared.NOGESTURE_FILL, 'all')
        
        % Calulate the mean of frames for all samples
        numFiles = length(fds.Files);
        numFramesSamples = zeros(numFiles, 1);
        for j = 1:numFiles
            frames = load(fds.Files{j, 1}).data.sequenceData;
            numFramesSamples(j, 1) = length(frames);
        end
        % Save the mean of frames for all samples
        noGestureFramesPerSample{i,1} = round(mean(numFramesSamples));
        
    elseif isequal(Shared.NOGESTURE_FILL, 'some')
        
        % Create labels to identify the class
        labels = Shared.createLabels(fds.Files, false);
        gestures = Shared.setNoGestureUse(false);
        avgNumFramesClass = zeros(length(gestures), 1);
        
        % For each class
        for j = 1:length(gestures)
            
            % Get the files of the class
            class = gestures(1, j);
            idxs = cellfun(@(label) isequal(label,class), cellstr(labels));
            filesClass = fds.Files(idxs, 1);
            
            % Get the number of frames of each sampple of the class
            numFilesClass = length(filesClass);
            numFramesSamples = zeros(numFilesClass, 1);
            for k = 1:numFilesClass
                frames = load(filesClass{k, 1}).data.sequenceData; 
                numFramesSamples(k, 1) = length(frames);                
            end
            
            % Calculate the mena number of frames for the class
            avgNumFramesClass(j, 1) = round(mean(numFramesSamples));
            
        end
        
        % Save the minimun and maximun number of frames of all classes
        noGestureFramesPerSample{i,1} = [min(avgNumFramesClass), max(avgNumFramesClass)];
    end
end
% Clean up variables
clear i j k class gestures filesClass frames idxs avgNumFramesClass labels numFilesClass numFramesSamples fds catCounts minNumber

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
noGestureTraining = noGestureFramesPerSample{1,1};
noGestureValidation = noGestureFramesPerSample{2,1};
if Shared.includeTesting
    noGestureTesting = noGestureFramesPerSample{3,1};
end

for i = 1:size(usersSets, 1)
    % Select a set of users
    users = usersSets{i,1};
    usersSet = usersSets{i,2};
    
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

clear i j datastore1 datastore2 noGestureSize1 noGestureSize2 noGestureTesting noGestureTraining noGestureValidation
clear testingDatastore trainingDatastore trainingPath users usersSet validationDatastore transformedSamplesValidation validationSamples

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
function [data, groundTruth] = generateFrames(signal, groundTruth, numGesturePoints, gestureName)

    % Fill before frame classification
    if isequal(Shared.FILLING_TYPE_EVAL, 'before')
        
        % Get a nogesture portion of the sample to use as filling
        noGestureInSignal = signal(~groundTruth, :);
        filling = noGestureInSignal(1: floor(Shared.FRAME_WINDOW / 2), :);

        % Combine the sample with the filling
        signal = [signal; filling];
        groundTruth = [groundTruth; zeros(floor(Shared.FRAME_WINDOW / 2), 1)];
    end

    % Allocate space for the results
    numWindows = floor((length(signal)-Shared.FRAME_WINDOW) / Shared.WINDOW_STEP_LSTM) + 1;
    data = cell(numWindows, 3);
    data(:,2) = {'noGesture'};
    isIncluded = false(numWindows, 1);
    
    % Creating frames
    for i = 1:numWindows
        
        % Get signal data to create a frame
        traslation = ((i-1)* Shared.WINDOW_STEP_LSTM);
        inicio = 1 + traslation;
        finish = Shared.FRAME_WINDOW + traslation;
        timestamp = inicio + floor(Shared.FRAME_WINDOW / 2);
        frameGroundTruth = groundTruth(inicio:finish);
        totalOnes = sum(frameGroundTruth == 1);
        
         % Get Spectrogram of the window
        frameSignal = signal(inicio:finish, :);
        spectrograms = Shared.generateSpectrograms(frameSignal);
        
        % Set data
        data{i,1} = spectrograms; % datum
        data{i,3} = timestamp; % time
        
        % Check the thresahold to consider gesture
        if totalOnes >= Shared.FRAME_WINDOW * Shared.TOLERANCE_WINDOW || ...
                totalOnes >= numGesturePoints * Shared.TOLERNCE_GESTURE_LSTM
            isIncluded(i,1) = true;
            data{i,2} = gestureName;
        end
    end
    
    % Include nogestures in the sequence
    if isequal(Shared.NOGESTURE_FILL, 'all')
        
        isIncluded(:,1) = true;
        
    elseif isequal(Shared.NOGESTURE_FILL, 'some')
        
        first = find(isIncluded, true, 'first');
        last = find(isIncluded, true, 'last');
        
        for i = 1:Shared.NOGESTURE_IN_SEQUENCE
            % Include some from left
           if first - i >= 1
                isIncluded(first-i, 1) = true;
           end
           % Include some from right
           if last + i <= numWindows
                isIncluded(last + i, 1) = true;
           end
        end
        
    end
    
    % Filer results    
    data = data(isIncluded,:);
            
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
        [data, newGroundTruth] = generateFrames(signal, groundTruth, numGesturePoints, gestureName);
        
        % Save the transformed data
        transformedSamples{i - noGesturePerUser, 1} = data;
        transformedSamples{i - noGesturePerUser, 2} = gestureName;
        transformedSamples{i - noGesturePerUser, 3} = transpose(newGroundTruth); % Can have filling
    end
end

%% FUNCTION TO SAVE SPECTROGRAMS IN DATASTORE
function saveSampleInDatastore(samples, user, type, dataStore)
    % For each sample
    for i = 1:length(samples)
        
        % Get data from trabnsformed samples
        sequenceData = samples{i,1};
        class = samples{i,2};
        
        % Get data in sequence
        timestamps = sequenceData(:,3);
        
        % Create a file name (user-type-sample-start-finish)
        fileName = strcat(strtrim(user),'-', type, '-',int2str(i), '-', ...
                '[',int2str(timestamps{1,1}), '-', int2str(timestamps{length(timestamps), 1}), ']');
            
        % Set data to save
        data.sequenceData = sequenceData;
        if ~isequal(class,'noGesture')
            data.groundTruth = samples{i,3};
        end
                
        % Save data
        savePath = fullfile(dataStore, char(class), fileName);
        save(savePath,'data');
        
    end
end

%% FUNCTION O GENERATE NO GESTURE FRAMES
function data = generateFramesNoGesture(signal, requestedWindows)

    % Calculate the number of windows to apply the signal
    numWindows = floor((length(signal) - Shared.FRAME_WINDOW) / Shared.WINDOW_STEP_LSTM) + 1;
    
    % Calculate if signal needs filling
    numWindowsFill = requestedWindows - numWindows;
    if numWindowsFill > 0
        % Get a nogesture portion of the sample to use as filling
        filling = signal(1:numWindowsFill*Shared.WINDOW_STEP_LSTM , :);
        % Fill before frame classification
        signal = [signal; filling];
    end
    
    % Allocate space for the results
    data = cell(requestedWindows, 3);
    data(:,2) = {'noGesture'};
      
    % For each window
    for i = 1:requestedWindows
        
        % Get window information
        traslation = ((i-1) * Shared.WINDOW_STEP_LSTM);
        inicio = 1 + traslation;
        finish = Shared.FRAME_WINDOW + traslation;
        timestamp = inicio + floor(Shared.FRAME_WINDOW / 2);
        
        % Generate a spectrogram
        frameSignal = signal(inicio:finish, :);
        spectrograms = Shared.generateSpectrograms(frameSignal);
        
        % Save data
        data{i,1} = spectrograms; % datum
        data{i,3} = timestamp; % label
    end  
end

%% FUNCTION TO GENERATE DATA OF NOGESTURE
function transformedSamples = generateDataNoGesture(samples, numFrames)
    % Number of noGesture samples to use them
    noGesturePerUser = Shared.numGestureRepetitions;
    
    % Allocate space for the results
    samplesKeys = fieldnames(samples);
    transformedSamples = cell(noGesturePerUser, 2);

    for i = 1:noGesturePerUser
        
        % Get sample data
        sample = samples.(samplesKeys{i});
        emg = sample.emg;
        gestureName = sample.gestureName;
        
        % Get signal from sample
        signal = Shared.getSignal(emg);
        signal = Shared.preprocessSignal(signal);
        
        if isequal(Shared.NOGESTURE_FILL, 'all')
            framesPerSample = numFrames;
        elseif isequal(Shared.NOGESTURE_FILL, 'some')
            rng(i);
            framesPerSample = randi(numFrames); % [min, max]
        end

        % Generate spectrograms
        data = generateFramesNoGesture(signal, framesPerSample);
        
        % Save the transformed data
        transformedSamples{i,1} = data;
        transformedSamples{i,2} = gestureName;
    end
end
