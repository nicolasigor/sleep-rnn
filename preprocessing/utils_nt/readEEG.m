function eegData = readEEG( regName )

% Extraction of eeg recorded data from edf file
%   INPUT
%       regName: Name of the register to be extracted
%   OUTPUT
%       eegData: reading results and parameters

%% Set parameters for extraction

p.regName = regName;
fprintf('Reading register %s... \n', p.regName);

p.regContainer = 'ssdata/register';
p.regRecFile = [p.regContainer '/' p.regName '.rec'];

p.channel = 1; % EEG Channel

%% Extract channel record

% Read .rec file and obtain specified channel
[eegData.header, record] = edfread(p.regRecFile);
p.fs = eegData.header.frequency(p.channel); % Sampling Frequency [Hz]
eegData.eegRecord = record(p.channel, :)';

%% Output params as well

eegData.params = p;

fprintf('Finished.\n');
