%{
    CNN
%}
dataDir = 'Datastores'; % folder where the datastores are created and stored

% #################################################################
% #################################################################

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

%% DEFINE THE START AND MINIBATCH SIZE
position = 1;
miniBatchSize = 15;

%% READ THE DATASTORE
if ismember(miniBatchSize,(1:15)) && (position + miniBatchSize <= length(fds.Files))
    [frames, labels, position] = readFrames(fds, position, miniBatchSize);
else
    disp('Outside of the allowed ranges');
end

%% VISUALIZE FRAMES
channel = 1;
visualizeFramesInDatstore(frames, labels, datastore, channel);

%% FUNCTION TO READ FRAMES
function [frames, labels, start] = readFrames(fds, start, miniBatchSize)
finish = start + miniBatchSize;
% Get the data
files = fds.Files(start:finish,1);
frames = cell(miniBatchSize,1);
labels = cell(miniBatchSize,1);
% For file in minibatch
for i = 1:miniBatchSize
    filepath = fileparts(files{i,1}); % ../datastore/class
    % The last part of the path is the label
    [~,label] = fileparts(filepath); % [../datastore, class]
    frame = Shared.readFile(files{i,1});
    frames{i,1} = frame;
    labels{i,1} = label;
end
start = start + miniBatchSize;
end

%% FUNCTION TO SUBPLOT A SPECTROGRAM FRAME
function subPlotSpectrogram(plotPosition, frame, label, channel)
f = 1:size(frame, 1);
t = 1:size(frame, 2);
ps = frame(:,:,channel);
% Space = 15 spaces to plot
subplot(3, 5, plotPosition)
surf(t,f,ps,'EdgeColor','none');
axis xy; axis tight; colormap(jet); view(0,90);
title(strcat('Gesture-', label));
end

%% FUNCTION TO VISUALIZE FRAMES
function visualizeFramesInDatstore(frames, labels, type, channel)
figure('Name', strcat('Gestures-', type, '-channel-', int2str(channel)));
for i = 1:length(frames)
    subPlotSpectrogram(i, frames{i}, labels{i}, channel);
end
end
