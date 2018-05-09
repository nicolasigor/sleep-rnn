%% Select register
regName = 'ADGU101504';
dir = ['ssdata_clean/' regName '/' ];
channel = 1;

%% Create new register file

eegData = getSleepEEG( regName , channel);  % Get eeg with valid marks and states, one channel
eegData = getFilteredEEG( eegData );        % Extract frequency bands from eeg
tabData = getTabData( eegData );            % Extract tabular format of register

%% Extract segments of N2 from new files and normalize

% Cut normalized N2 segments from eeg data
downsampling = 1;       % 1 for True (i.e. do downsampling rate 2)
state = 3;              % Sleep Stages, [1,2]:N3 3:N2  4:N1  5:R  6:W
N2Data = getSegments( tabData, eegData , state, downsampling);

% Save obtained segments as new database
for k = 1:length(N2Data)
    %saveData(N2Data(k).marks, [dir '/segments_N2/expert_marks/' regName '_SS_marks_seg' num2str(k) '.txt'] );
    %saveData(N2Data(k).tabData, [dir '/segments_N2/tabular_data/' regName '_feats_label_seg' num2str(k) '.txt'] );
end