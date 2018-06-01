function eegData = getSleepEEG( regName )
% Extraction of eeg recorded data from edf file
%   INPUT
%       regName: Name of the register to be extracted
%   OUTPUT
%       eegData: reading results and parameters

%% Set parameters for extraction

p.regName = regName;
fprintf('Reading register %s...\n', p.regName);

p.regContainer = 'ssdata';
p.regRecFile = [p.regContainer '/' p.regName '.rec'];
%p.regStatesFile = [p.regContainer '/' p.regName '/Sleep States/States_' p.regName '.txt'  ];
p.regStatesFile = [p.regContainer '/' p.regName '/Sleep States/StagesOnly_' p.regName '.txt'  ];
p.regSpindlesFile = [p.regContainer '/' p.regName '/Sleep Spindles/SS1_' p.regName '.txt'  ];

% According to literature, SS durations are most commonly encountered
% between 0.5 and 1.5 seconds, but here we consider a broader range for 
% completeness.
p.minSSduration = 0.3;      % Min feasible SS duration
p.maxSSduration = 3.0;      % Max feasible SS duration

p.epochDuration = 30;        % Time of window page [s]
p.channel = 1;              % EEG Channel

%% Extract channel record

% Read .rec file and obtain specified channel
[eegData.header, record] = edfread(p.regRecFile);
p.fs = eegData.header.frequency(p.channel); % Sampling Frequency [Hz]
eegData.eegRecord = record(p.channel, :)';

%% Load labels

% Load Sleep States
% Sleep Stages, [1,2]:N3 3:N2  4:N1  5:R  6:W
regStates = load(p.regStatesFile);
%regStates = regStates(:,8);
% Fix Stages by combining old SQ3 and SQ4 stages
regStates(regStates == 1) = 2; 
% Now  2:N3  3:N2  4:N1  5:R  6:W
eegData.regStates = regStates;

% Load Sleep Spindles marks
regSpindles = load(p.regSpindlesFile);

%% Extract and clean marks from selected channel

n_marks_file = length(regSpindles);
fprintf('%d marks in file\n',n_marks_file);
marks = regSpindles( regSpindles(:,6) == p.channel, : );
n_marks_ch1 = length(marks);
fprintf('%d marks in CH%d\n',n_marks_ch1,p.channel);
[marks, steps] = cleanExpertMarks( marks, eegData.regStates, p.epochDuration, p.fs, p.minSSduration, p.maxSSduration );
eegData.marks = marks;
eegData.marks_steps = steps;
eegData.marks_steps.n_marks_file = n_marks_file;
eegData.marks_steps.n_marks_ch1 = n_marks_ch1;

%% Output params as well
eegData.params = p;

fprintf('Reading finished\n');