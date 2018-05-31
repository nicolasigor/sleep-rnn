%% Sleep Spindles marks statistics
fs = eegData.params.fs;
width = (eegData.marks(:,2)-eegData.marks(:,1))/fs;
histogram(width,20);
title('Distribution of SS marks duration'), xlabel('Duration [s]'), ylabel('Counts')

%% States statistics
% Sleep Stages, [1,2]:N3 3:N2  4:N1  5:R  6:W
Twin = eegData.params.epochDuration;
nEpochsStates = zeros(5,1);
nEpochsStates(1) = sum( eegData.regStates == 4 );
nEpochsStates(2) = sum( eegData.regStates == 3 );
nEpochsStates(3) = sum( eegData.regStates == 2 ) + sum( eegData.regStates == 1 );
nEpochsStates(4) = sum( eegData.regStates == 5 );
nEpochsStates(5) = sum( eegData.regStates == 6 );
fprintf('Total:%d N1:%d N2:%d N3:%d R:%d W:%d \n',sum(nEpochsStates),nEpochsStates(1),nEpochsStates(2),nEpochsStates(3),nEpochsStates(4),nEpochsStates(5));
%%
states = categorical({'N1','N2','N3','R', 'W'});
figure, bar(states,nEpochsStates)
title(sprintf('Total: %1.2f hours',eegData.params.regDurationHrs));
ylabel('Epochs')
%%

fs = eegData.params.fs;
interval = (167002-5000:167002+5000);
time = tabData(interval,1)/fs;
plot(time,tabData(interval,2),time,tabData(interval,3),time,100*tabData(interval,5))

% Ver las marcas raras del final en el archivo original de marcas, 1,2,3?

% Do several sanity checks for the insertion of marks and stages along the eeg signal

%% Spectrogram
% N2 segment: from epoch 135 to epoch 202
% 200Hz*30s = 6000 samples per epoch

% EEG segment
epoch = 170;
fs = eegData.params.fs;
interval = (epoch*6000):((epoch+1)*6000);
time = (tabData(interval,1)/fs);
time = time-min(time);
eegInterval = tabData(interval,2);
marksInterval = tabData(interval,end);
figure,
subplot(3,1,1)
plot(time,eegInterval,time,100*marksInterval)
xlim([time(1), time(end)]), xlabel('Time [s]')

% Spectrogram
% Downsampling first
ratio = 2;
%[B, A] = butter(3, [0.5, fs/ratio]/(fs/2) , 'bandpass');
%eegRecord_down = filtfilt( B, A, eegInterval);
eegRecord_down = eegInterval;
eegRecord_down = downsample(eegRecord_down,ratio);
fs_down = fs/ratio;
subplot(3,1,2)
plot(downsample(time,ratio),eegRecord_down,time,100*marksInterval)
xlim([time(1), time(end)]), xlabel('Time [s]')
subplot(3,1,3)
s = spectrogram(eegRecord_down,50,40,[],fs_down,'yaxis');

db_p = 10*log10(abs(s));
t = (0:length(s(1,:))-1)/fs_down;
f = (fs_down)*(0:length(s(:,1))-1)/(2*length(s(:,1))-2);
surf(t,f,db_p,'EdgeColor','none');
colormap hot
colorbar off
hold on 
plot(time,20*marksInterval)