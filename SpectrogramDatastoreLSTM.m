%{ 
    LSTM
%}

classdef SpectrogramDatastoreLSTM < matlab.io.Datastore & ...
                                matlab.io.datastore.MiniBatchable & ...
                                matlab.io.datastore.Shuffleable & ...
                                matlab.io.datastore.Partitionable
     
    properties
        Datastore
        Labels
        SequenceLength
        FrameDimensions
        NumClasses
        MiniBatchSize
        CurrentFileIndex
    end
    
    properties(SetAccess = protected)
        NumObservations
    end
        
    methods
        
        function ds = SpectrogramDatastoreLSTM(folder)
            
            % Create a file datastore.
            fds = fileDatastore(folder, ...
                'ReadFcn',@Shared.readFile, ...
                'IncludeSubfolders',true);
            ds.Datastore = fds;
            
            % Read labels from folder names
            [labels, numObservations] = Shared.createLabels(fds.Files, true);
            ds.Labels = labels;
            ds.NumClasses = length(Shared.setNoGestureUse(true));
            
            % Initialize datastore properties
            ds.MiniBatchSize = 64;
            ds.NumObservations = numObservations;
            ds.CurrentFileIndex = 1;

            % Determine sequence and frame dimensions
            sample = load(ds.Datastore.Files{1}).data;
            ds.FrameDimensions = size(sample.sequenceData{1,1});
            
            % Shuffle
            ds = shuffle(ds);
        end
        
        function tf = hasdata(ds)
            % Return true if more data is available
            tf = ds.CurrentFileIndex + ds.MiniBatchSize - 1 ...
                <= ds.NumObservations;
        end
        
        function [data,info] = read(ds)            
            % Function to read data
            miniBatchSize = ds.MiniBatchSize;
            [sequencesData, responses, groundTruths]  = deal(cell(miniBatchSize, 1));
            % Data for minibatch size is read
            for i = 1:miniBatchSize
                %data = read(ds.Datastore).data;
                content = read(ds.Datastore);
                sequencesData{i,1} = content.sequenceData;
                class = ds.Labels(ds.CurrentFileIndex);
                responses{i,1} = class;
                if ~isequal(class, 'noGesture')
                    groundTruths{i, 1} = content.groundTruth;
                end
                ds.CurrentFileIndex = ds.CurrentFileIndex + 1;
            end
            
            % Data is preprocessed;
            [data, timestamps] = preprocessData(ds, sequencesData);
            % Set information
            info = struct('responses', responses, 'timestamps', timestamps, 'groundTruths', groundTruths);
        end
        
        
        
        function [data, timestamps] = preprocessData(ds, sequencesData)
            miniBatchSize = ds.MiniBatchSize;
            [sequences, labelsSequences, timestamps]  = deal(cell(miniBatchSize, 1));
            
            % Get max frame number in sequences
            sequencesLengths = cellfun(@(sequence) length(sequence), sequencesData);
            
            % Set sequence dimensions
            if isequal(Shared.PAD_KIND, 'shortest')
                samplesLength = min(sequencesLengths);
            elseif isequal(Shared.PAD_KIND, 'longest')
                samplesLength = max(sequencesLengths);                
            end
            
            % Set sequence dimensions
            frameDimensions = ds.FrameDimensions;
            fixDimensions = [frameDimensions, samplesLength];

            % Process the minibath
            parfor i = 1:miniBatchSize
                
                % Put original data 
                sequenceData = sequencesData{i, 1};
                numFrames = length(sequenceData);

                if isequal(Shared.PAD_KIND, 'shortest') || isequal(Shared.PAD_KIND, 'longest')
                    
                    % Allocate space for new data
                    newLabels  = cell(1, samplesLength);
                    newTimestamps = cell(1, samplesLength);
                    newSequence = zeros(fixDimensions);
                    
                else
                    
                    % Allocate space for new data
                    newLabels  = cell(1, numFrames);
                    newTimestamps = cell(1, numFrames);
                    sequenceDimensions = [frameDimensions, numFrames];
                    newSequence = zeros(sequenceDimensions);
                    
                end
              
                if isequal(Shared.PAD_KIND, 'shortest')
                    
                    for j = 1:samplesLength
                        newSequence(:,:, :,j) = sequenceData{j,1};
                        newLabels{1, j} = sequenceData{j,2};
                        newTimestamps{1, j} = sequenceData{j,3};
                    end
                    
                else
                    
                    for j = 1:numFrames
                        newSequence(:,:, :,j) = sequenceData{j,1};
                        newLabels{1, j} = sequenceData{j,2};
                        newTimestamps{1, j} = sequenceData{j,3};
                     end
                     
                    if isequal(Shared.PAD_KIND, 'longest')
                        
                        % Put filling at the end to match the max length
                        lastTimestamps = sequenceData{numFrames,3};
                        %lastFrame = sequenceData{numFrames,1};

                        for j = 1:(samplesLength - numFrames)
                            newLabels{1, numFrames + j} = 'noGesture';
                            newTimestamps{1, numFrames + j} = lastTimestamps + (j * Shared.WINDOW_STEP_LSTM);
                        end
                    end
                end
               
                % Set data transformed
                sequences{i,1} = newSequence;
                labelsSequences{i,1} = categorical(newLabels, Shared.setNoGestureUse(true));
                timestamps{i,1} = newTimestamps;  
            end
            
            % Put the data in table form
            data = table(sequences, labelsSequences);
        end
        
        function reset(ds)
            % Reset to the start of the data
            reset(ds.Datastore);
            ds.CurrentFileIndex = 1;
            ds.NumObservations = size(ds.Datastore.Files, 1);
        end
        
        function [ds1, ds2] = partition(ds, percentage)
            % Create copys to set the result
            ds1 = copy(ds); ds2 = copy(ds);
            
            % Get the limit of the new division
            numObservations = ds.NumObservations;
            numClassSamples = floor(numObservations / ds.NumClasses);
            limitOfSamples = floor(numClassSamples*percentage);
            
            % Match the specidfied number of samples and order them
            dsNew = matchSampleNumberInOrder(ds, numClassSamples);
            dsLabels = dsNew.Labels;
            dsFiles = dsNew.Datastore.Files;
            
            % Create cell to set the new labels and files
            ds1Labels = {}; ds1Files = {}; ds2Labels = {}; ds2Files = {};
            
            % Divide data per gesture
            parfor i = 1:ds.NumClasses
                
                % Calculate the samples per partition
                labels = dsLabels;
                files = dsFiles;
                start = ((i-1) * numClassSamples) + 1;
                limit = start + limitOfSamples;
                last = i * numClassSamples;
                
                % Set new number of files and labels
                ds1Labels = [ds1Labels; labels(start:limit-1, 1)];
                ds1Files = [ds1Files; files(start:limit-1, 1)];
                ds2Labels = [ds2Labels; labels(limit:last, 1)];
                ds2Files = [ds2Files; files(limit:last, 1)];
            end
            
            % Set the data to new datastores
            ds1 = prepareNewDatastore(ds1, ds1Labels, ds1Files);
            ds2 = prepareNewDatastore(ds2, ds2Labels, ds2Files);
        end
  
        function dsNew = shuffle(ds)
            % Create a copy of datastore
            dsNew = copy(ds);
            fds = dsNew.Datastore;
            % Shuffle tthe filedatastore
            [fds, idxs] = Shared.shuffle(fds);
            % Save new order
            dsNew.Datastore = fds;
            dsNew.Labels = dsNew.Labels(idxs);
        end
        
        function dsNew = balanceGestureSamples(ds)
            % Calcule the class with less samples
            labels = ds.Labels;
            catCounts = sort(countcats(labels));
            minNumber = catCounts(1);
            % get a new balanced datastore and shuflle it
            dsNew = matchSampleNumberInOrder(ds, minNumber);
            dsNew = shuffle(dsNew);
        end
        
        function dsNew = setDataAmount(ds, percentage)
            if percentage == 1 
                dsNew = ds;
            else
                % Get the limit of the new division
                numObservations = ds.NumObservations;
                newLimit = floor(numObservations * percentage);
                numClassSamples = floor(newLimit/ds.NumClasses);
                % Set the new number of files
                dsNew = matchSampleNumberInOrder(ds, numClassSamples);
                dsNew = shuffle(dsNew);
            end
        end
    end
    
    methods (Access = protected)
        function dscopy = copyElement(ds)
            dscopy = copyElement@matlab.mixin.Copyable(ds);
            dscopy.Datastore = copy(ds.Datastore);
        end
        
        function n = maxpartitions(myds) 
            n = maxpartitions(myds.FileSet); 
        end  
    end
    
    methods (Hidden = true)
        function frac = progress(ds)
            % Determine percentage of data read from datastore
            frac = (ds.CurrentFileIndex - 1) / ds.NumObservations;
        end
    end
        
end

%% FUNCTION TO MATCH THE NUMBER OF SAMPLES OF EACH GESTURE
function ds = matchSampleNumberInOrder(ds, repetitions)
    labels = ds.Labels;
    gesturefiles = ds.Datastore.Files;
    gestures = categorical(categories(labels));
    
    % Allocate space for results
    newLabels = {}; newFiles = {};
    
    % Get equal number of samples for each gesture
    parfor i = 1:length(gestures)
        files = gesturefiles;
        gestureLabels = cell(repetitions, 1);
        gestureFiles = cell(repetitions, 1);
        
        % Put 1s where is the gesture and 0s where is not
        isGesture = ismember(labels, gestures(i));
        % Get indexes of ones
        gestureIdxs = find(isGesture);
        
        % Save data until the limit (repetitions)
        for j = 1:repetitions
            gestureLabels{j, 1} = char(gestures(i));
            gestureFiles{j, 1} = files{gestureIdxs(j)};
        end
        
        % Concatenate the labels and files
        newLabels = [newLabels; gestureLabels];
        newFiles = [newFiles; gestureFiles];
    end
    % Make data categorical
    newLabels = categorical(newLabels,categories(gestures));
    
    % Save the transformed data
    ds.Labels = newLabels;
    ds.NumObservations = length(newLabels);
    ds.Datastore.Files = newFiles;
end

%% FUNCTION TO PREPARE A NEW DATASTORE
function ds = prepareNewDatastore(ds, dsLabels, dsFiles)
    % Set the data to new datastores
    ds.NumObservations = length(dsLabels);
    ds.Labels = dsLabels;
    ds.Datastore.Files = dsFiles;
    % Shufle datastore
    ds = shuffle(ds);
end

%% EXTRA THINGS
%{
    % Orden by length of frames
    %ds = order(ds);

    function ds = order(ds)
        % Order the datastores by number of frames
        numObservations = ds.NumObservations;
        sequenceLengths = zeros(numObservations, 1);
        files =  ds.Datastore.Files;
        for i=1:numObservations
            filename = files{i};
            sequenceData = load(filename).data.sequenceData;
            sequenceLengths(i) = size(sequenceData, 1);
        end
        [~,idx] = sort(sequenceLengths);
        ds.Datastore.Files = ds.Datastore.Files(idx);
        ds.Labels = ds.Labels(idx);
    end


    %% CONFIGURATIONS INSTAEL OF ZERO PADDING
    if Shared.CONSIDER_PREVIOUS
        numRows = ds.FrameDimensions(1);
        numCols = ds.FrameDimensions(2);
        strideSequence = numCols - round((1 - (Shared.WINDOW_STEP_LSTM ... 
                / Shared.FRAME_WINDOW)) * numCols); % 1
    end

    if Shared.CONSIDER_PREVIOUS
        frameRemain = lastFrame(:, 1+strideSequence:numCols, :);
        % DEBE DE SER UN REELENO QUE SIMULE NG
        filling = zeros(numRows, strideSequence, Shared.numChannels);
        newFrame = [frameRemain, filling];
        newSequence(:,:, :,numFrames + j) = newFrame;
        lastFrame = newFrame;
    end

    % if isequal(Shared.SEQUENCE_INIT, 'noGesture') % Rellenar
    % con valores de frame de no gesture
                
%}
