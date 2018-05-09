function eegData = getFilteredEEG( eegData )
% Get medical frequency bands, and band-passed eeg record

fprintf('Filtering EEG data and extracting frequency bands...\n');

fs = eegData.params.fs;

% Frequency ranges to be extracted
fRange.delta = [0.5, 4];
fRange.theta = [3.5, 7];
fRange.sigma = [10, 16];

bandNames = fieldnames(fRange);
b = numel(bandNames);
n = length(eegData.eegRecord);
bands = zeros(n, b);
for k = 1:b
    [B, A] = butter(3, fRange.(bandNames{k})/(fs/2) , 'bandpass');
    bands(:,k) = filtfilt( B, A, eegData.eegRecord);
end
eegData.eegBands = bands;

%% Band-passed EEG Record, to cut 0Hz and high-frequency noise

fRange.cut = [0.5, 40];
[B, A] = butter(3, fRange.cut/(fs/2) , 'bandpass');
eegRecord_filtered = filtfilt( B, A, eegData.eegRecord);
eegData.eegRecordFiltered = eegRecord_filtered;

%% Save parameters
eegData.bandsFreqRanges = fRange;

fprintf('Filtering finished\n');