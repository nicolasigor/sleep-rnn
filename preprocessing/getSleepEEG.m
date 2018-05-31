function eegData = getSleepEEG( regName , channel)
% Extraction of eeg recorded data from edf file
%   INPUT
%       regName: Name of the register to be extracted
%   OUTPUT
%       eegData: reading results and parameters

%% Set parameters for extraction

if nargin > 0
    p.regName = regName;
    p.channel = channel;        % EEG Channel
else % Default
    p.regName = 'ADGU101504';
    p.channel = 1;              % EEG Channel
end
fprintf('Reading register %s...\n', p.regName);

p.regContainer = 'ssdata';
p.regRecFile = [p.regContainer '/' p.regName '.rec'];
p.regStatesFile = [p.regContainer '/' p.regName '/Sleep States/States_' p.regName '.txt'  ];
p.regSpindlesFile = [p.regContainer '/' p.regName '/Sleep Spindles/SS1_' p.regName '.txt'  ];

% According to literature, SS durations are most commonly encountered
% between 0.5 and 1.5 seconds, but here we consider a broader range for 
% completeness.

p.minSSduration = 0.3;      % Min feasible SS duration
p.maxSSduration = 3.0;      % Max feasible SS duration

p.epochDuration = 30;        % Time of window page [s]

%% Extract channel record

% Read .rec file and obtain specified channel
[eegData.header, record] = edfread(p.regRecFile);
p.fs = eegData.header.frequency(p.channel); % Sampling Frequency [Hz]
eegData.eegRecord = record(p.channel, :)';
p.regDurationHrs = length(eegData.eegRecord)/(p.fs*3600);

%% Load labels

% Load Sleep States
regStates = load(p.regStatesFile);
regStates = regStates(:,8);
eegData.regStates = regStates;      % Sleep Stages, [1,2]:N3 3:N2  4:N1  5:R  6:W
p.nEpochs = length(regStates);       % Number of pages in record

% Load Sleep Spindles marks
regSpindles = load(p.regSpindlesFile);

%% Extract and clean marks from selected channel
fprintf('Total marks in file %d\n',length(regSpindles));
marks = regSpindles( regSpindles(:,6) == p.channel, : );
fprintf('Total marks in CH%d %d\n',p.channel, length(marks));

marks = cleanExpertMarks( marks, eegData.regStates, p.epochDuration , p.fs, p.minSSduration, p.maxSSduration );
eegData.marks = marks;
p.nMarks = length(marks);
p.marksDurationHrs = sum( diff(marks') ) /(p.fs*3600);

%% Output params as well

eegData.params = p;

fprintf('Reading finished\n');