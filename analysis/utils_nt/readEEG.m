function eegData = readEEG( filename, channel )

% Extraction of eeg recorded data from edf file
%   INPUT
%       filename: EDF Register to be read
%       channel: integer indicating channel to be extracted
%   OUTPUT
%       eegData: reading results and parameters

%% Set parameters for extraction

p.recFile = filename;
p.channel = channel; % EEG Channel
fprintf('Reading register at %s... \n', p.recFile);

%% Extract channel record from .rec file (EDF Format)

[eegData.header, record] = edfread(p.recFile);
eegData.eegRecord = record(p.channel, :)';

% Sampling Frequency [Hz]
p.fs = eegData.header.frequency(p.channel); 

%% Output params as well

eegData.params = p;

fprintf('Finished.\n');
