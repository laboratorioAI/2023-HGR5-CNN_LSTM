%{
    LSTM
%}

%% DEFINE THE DIRECTORIES WHERE THE DATA WILL BE FOUND
dataDir = 'DatastoresLSTM';
datastores = {'training'; 'validation'; 'testing'};

%% CREATE A FILEDATASTORE
datastore = datastores{3};
folder = fullfile(dataDir, datastore);
% Create a file datastore.
fds = fileDatastore(folder, ...
    'ReadFcn',@Shared.readFile, ...
    'IncludeSubfolders',true);
fds = Shared.shuffle(fds);
clear folder

%% DEFINE THE SAMPLE
position = 60;

%% READ THE DATASTORE
if (position > 0) && (position  <= length(fds.Files))
    [frames, labels] = readFrames(fds, position);
else
    disp('Outside of the allowed ranges');
end

%% VISUALIZE FRAMES
channel = 1;
visualizeFramesInDatstore(frames, labels, datastore, channel);

%% FUNCTION TO READ FRAMES
function [frames, labels] = readFrames(fds, position)            
    % Get the data
    file = fds.Files{position,1};
    data = Shared.readFile(file);
    sequenceData = data.sequenceData;
    frames = sequenceData(:,1);
    labels = sequenceData(:,2);
end

%% FUNCTION TO SUBPLOT A SPECTROGRAM FRAME
function subPlotSpectrogram(plotPosition, frame, label, channel, numFrames)
    % Initializze dimentions
    rows = 4;
    columns = ceil(numFrames / rows);
    % Obtain frame info
    f = 1:size(frame, 1);
    t = 1:size(frame, 2);
    ps = frame(:,:,channel);
    % Space = 15 spaces to plot
    subplot(rows, columns, plotPosition) 
        surf(t,f,ps,'EdgeColor','none');   
        axis xy; axis tight; colormap(jet); view(0,90);
        title(strcat('Gesture-', label));
end

%% FUNCTION TO VISUALIZE FRAMES
function visualizeFramesInDatstore(frames, labels, type, channel)
    figure('Name', strcat('Gestures-', type, '-channel-', int2str(channel)));
    for i = 1:length(frames)
        subPlotSpectrogram(i, frames{i}, labels{i}, channel, length(frames));
    end
end
