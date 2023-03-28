%{ 
    CNN
%}

classdef SpectrogramDatastore < matlab.io.Datastore & ...
                                matlab.io.datastore.MiniBatchable & ...
                                matlab.io.datastore.Shuffleable & ...
                                matlab.io.datastore.Partitionable
     
    properties
        Datastore
        Labels
        NumClasses
        DataDimensions
        MiniBatchSize
    end
    
    properties(SetAccess = protected)
        NumObservations
    end
    
    properties(Access = private)
        CurrentFileIndex
    end
    
    methods
        function ds = SpectrogramDatastore(folder, withNoGesture)
            % Create a file datastore.
            fds = fileDatastore(folder, ...
                'ReadFcn',@readDatastore, ...
                'IncludeSubfolders',true);
            ds.Datastore = fds;
            
            % Read labels from folder names
            [labels, numObservations] = Shared.createLabels(fds.Files, withNoGesture);
            ds.Labels = labels;
            ds.NumClasses = numel(unique(labels));
            
            % Determine sequence dimension
            filename = ds.Datastore.Files{1};
            sample = load(filename).data;
            ds.DataDimensions = size(sample);
            
            % Initialize datastore properties
            ds.MiniBatchSize = 32;
            ds.NumObservations = numObservations;
            ds.CurrentFileIndex = 1;
            
            % Shuffle
            ds = balanceGestureSamples(ds);
            ds = shuffle(ds);
        end
        
        function tf = hasdata(ds)
            % Return true if more data is available
            tf = ds.CurrentFileIndex + ds.MiniBatchSize - 1 ...
                <= ds.NumObservations;
        end
        
        function [data,info] = read(ds)            
            % Function to read data
            info = struct;
            miniBatchSize = ds.MiniBatchSize;
            predictors = cell(miniBatchSize, 1);
            responses = cell(miniBatchSize, 1);
            
            % Data for minibatch size is read
            for i = 1:miniBatchSize
                %data = read(ds.Datastore).data;
                content = read(ds.Datastore);
                predictors{i,1} = content;
                class = ds.Labels(ds.CurrentFileIndex);
                responses{i,1} = class;
                ds.CurrentFileIndex = ds.CurrentFileIndex + 1;
            end
            
            % Data is preprocessed;
            data = preprocessData(ds, predictors, responses);
        end
        
        function data = preprocessData(ds, predictors, responses)
            % Function to preprocess data
            miniBatchSize = ds.MiniBatchSize;
            parfor i = 1:miniBatchSize
                predictors{i} = predictors{i}.data;
            end
            % Put the data in table form
            data = table(predictors,responses);
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


%% FUNCTION TO READ DATA FROM FILE
function data = readDatastore(filename)
    % Load a Matlab file
    data = load(filename);
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
