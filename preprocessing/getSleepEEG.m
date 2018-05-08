function [p, eegData] = getSleepEEG( regName )
% regName: Name of the register to be extracted
% p: params used for extraction
% eegData: register extracted

%% Set parameters for extraction

if nargin > 0
    p.regName = regName;
else
    p.regName = 'ADGU101504';
end
fprintf('Reading register %s...\n', p.regName);

p.regContainer = 'ssdata';
p.regRecFile = [p.regContainer '/' p.regName '.rec'];
p.regStatesFile = [p.regContainer '/' p.regName '/Sleep States/States_' p.regName '.txt'  ];
p.regSpindlesFile = [p.regContainer '/' p.regName '/Sleep Spindles/SS1_' p.regName '.txt'  ];

p.channel = 1;              % EEG Channel
p.minSSduration = 0.3;      % Min feasible SS duration
p.maxSSduration = 3.0;      % Max feasible SS duration
p.pageDuration = 30;        % Time of window page [s]

%% Extract CH1 record

% Read .rec file and obtain specified channel
[eegData.header, record] = edfread(p.regRecFile);
p.fs = eegData.header.frequency(1); % Sampling Frequency [Hz]
eegData.eegRecord = record(p.channel, :);
p.regDurationHrs = length(eegData.eegRecord)/(p.fs*3600);

%% Load labels

% Load Sleep States
regStates = load(p.regStatesFile);
regStates = regStates(:,8);
eegData.regStates = regStates;
p.nPages = length(regStates);

% Load Sleep Spindles marks
regSpindles = load(p.regSpindlesFile);

%% Extract and clean marks from selected channel

marks = regSpindles( regSpindles(:,6) == p.channel, : );
marks = cleanExpertMarks( marks, p.fs, p.minSSduration, p.maxSSduration );
eegData.marks = marks;
p.nMarks = length(marks);
p.marksDurationHrs = sum( diff(marks') / p.fs )/3600;

