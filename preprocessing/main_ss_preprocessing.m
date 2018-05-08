%% Select register
regName = 'ADGU101504';
dir = ['ssdata_clean/' regName '/' ];
channel = 1;

%% Create new register file

eegData = getSleepEEG( regName , channel); % Get eeg with valid marks and states, one channel
%saveData( eegData.marks, [dir regName '_SS_marks.txt'] );
eegData = getFilteredEEG( eegData );      % Extract frequency bands from eeg
tabData = getTabData( eegData );     % Extract tabular format of register
%saveData( tabData, [dir regName '_feats_label.txt'] );

%% Extract segments of SQ2 from new files and normalize

% Cut normalized sq2 segments from eeg data
sq2Data = getSegments( regName, 'SQ2');
% Save obtained segments as new database
for k = 1:length(sq2Data)
    %saveData( sq2Data(k).marks, [dir '/segments_sq2/expert_marks/' regName '_SS_marks_seg' num2str(k) '.txt'] );
    %saveData( sq2Data(k).tabData, [dir '/segments_sq2/tabular_data/' regName '_feats_label_seg' num2str(k) '.txt'] );
end