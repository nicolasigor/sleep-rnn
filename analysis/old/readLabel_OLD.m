function eegLabel = readLabel( regName, fs, flag_save_fix)
% Extraction of eeg labels: hipnogram and sleep spindle marks
%   INPUT
%       regName: Name of the register to be extracted
%       flag_save_fix: 1 to save fixed marks in a file
%   OUTPUT
%       eegLabel: reading results and parameters

%% Set parameters for extraction

p.regName = regName;
fprintf('Reading labels for %s...\n', p.regName);

p.regContainer = 'ssdata/label';
p.regStatesFile = [p.regContainer '/' p.regName '/Sleep States/StagesOnly_' p.regName '.txt'  ];
p.regSpindlesFile = [p.regContainer '/' p.regName '/Sleep Spindles/SS1_' p.regName '.txt'  ];

p.epochDuration = 30;        % Time of window page [s]

% According to literature, SS durations are most commonly encountered
% between 0.5 and 1.5 seconds, but here we consider a broader range for 
% completeness.
p.minSSduration = 0.3;      % Min SS duration [s]
p.maxSSduration = 3.0;      % Max SS duration [s]

p.channel = 1;              % EEG Channel
p.fs = fs;                  % Sampling Frequency [Hz]

%% Load hipnogram
% Sleep Stages, [1,2]:N3 3:N2  4:N1  5:R  6:W
states = load(p.regStatesFile);
% Upgrade Stages by combining old SQ3 and SQ4 stages
states(states == 1) = 2; 
% Now  2:N3  3:N2  4:N1  5:R  6:W
eegLabel.states = states;

%% Load Sleep Spindles marks
marks = load(p.regSpindlesFile);
n_marks_file = length(marks);

% Only one channel
marks = marks( marks(:,6) == p.channel, : );
n_marks_ch1 = length(marks);

% Fix marks
[marks, valid, stats] = cleanExpertMarks( marks, states, p.epochDuration, p.fs, p.minSSduration, p.maxSSduration );
eegLabel.marks = marks;
eegLabel.marks_validity = valid;

% Save stats of fixing marks
eegLabel.marks_stats = stats;
eegLabel.marks_stats.n_marks_file = n_marks_file;
eegLabel.marks_stats.n_marks_ch1 = n_marks_ch1;

% Save new marks
if flag_save_fix
    p.regSpindlesFileFixed = [p.regContainer '/' p.regName '/Sleep Spindles/SS1_' p.regName '_ch1_fixed.txt'  ];
    fid=fopen(p.regSpindlesFileFixed,'w');
    fprintf(fid, '%d %d %d\n', [eegLabel.marks, eegLabel.marks_validity]');
    fclose(fid);
end

%% Output params as well
eegLabel.params = p;

fprintf('Reading finished\n');
