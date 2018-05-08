function eegData = getFilteredEEG( eegData )
% Get medical frequency bands, and band-passed eeg record

fs = eegData.params.fs;
fRange.sigma = [10, 16];
%fRange.alfa = [ 8, 15 ]; % ejemplo sin sentido para rangos
%fRange.beta = [10,15]; 
% param.frecRange=[10,16];
% [B,A]=butter(3,[param.frecRange(1)/(param.Fs/2) param.frecRange(2)/(param.Fs/2)] ,'bandpass');
% data(n).butter=filtfilt(B,A,data(n).record);

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