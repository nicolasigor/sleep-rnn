function eegLabel = readLabel(marks_filename, states_filename, channel)
% Extraction of eeg labels: hipnogram and sleep spindle marks
%   INPUT
%       marks_filename: Name of the file containing expert marks of SS
%       states_filename: Name of the file containing expert marks of sleep
%       states.
%       channel: Channel to be considered for SS marks
%   OUTPUT
%       eegLabel: reading results and parameters

%% Set parameters for extraction
p.statesFile = states_filename;
p.spindlesFile = marks_filename;
p.channel = channel;

%% Load hipnogram

fprintf('Reading states at %s...\n', p.statesFile);
% Sleep Stages, 1:SQ4  2:SQ3  3:SQ2  4:SQ1  5:REM  6:WA
states = load(p.statesFile);
% Upgrade old notation by combining old SQ3 and SQ4 into N3
states(states == 1) = 2; 
% Now  2:N3  3:N2  4:N1  5:R  6:W
eegLabel.states = states;

%% Load Sleep Spindles marks

fprintf('Reading marks at %s...\n', p.spindlesFile);
marks = load(p.spindlesFile);
stats.n_marks_all = size(marks, 1);

% Only one channel
marks = marks( marks(:,end) == p.channel, : );
% Keep only timesteps and validity
eegLabel.marks = marks(:,[1,2]);
eegLabel.marks_validity = marks(:,5);
stats.n_marks_channel = size(eegLabel.marks,1);

%% Output params and stats as well
eegLabel.stats = stats;
eegLabel.params = p;

fprintf('Reading finished\n');
