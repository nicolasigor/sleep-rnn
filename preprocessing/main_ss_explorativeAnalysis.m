%% Sleep Spindles marks statistics
fs = eegData.params.fs;
width = (eegData.marks(:,2)-eegData.marks(:,1))/fs;
histogram(width,20);
title('Distribution of SS marks duration'), xlabel('Duration [s]'), ylabel('Counts')

%% States statistics
% Sleep Stages, [1,2]:N3 3:N2  4:N1  5:R  6:W
Twin = eegData.params.epochDuration;
nEpochsStates = zeros(5,1);
nEpochsStates(1) = sum( eegData.regStates == 4 )*Twin/3600;
nEpochsStates(1) = sum( eegData.regStates == 3 )*Twin/3600;
nEpochsStates(1) = (sum( eegData.regStates == 2 ) + sum( eegData.regStates == 1 ))*Twin/3600;
nEpochsStates(1) = sum( eegData.regStates == 5 )*Twin/3600;
nEpochsStates(1) = sum( eegData.regStates == 6 )*Twin/3600;
states = categorical({'N1','N2','N3','R', 'W'});
figure, bar(states,nEpochsStates)
title(sprintf('Total: %1.2f hours',eegData.params.regDurationHrs));
ylabel('Hours')
%%

fs = eegData.params.fs;
interval = (167002-5000:167002+5000);
time = tabData(interval,1)/fs;
plot(time,tabData(interval,2),time,tabData(interval,3),time,100*tabData(interval,5))

% Ver las marcas raras del final en el archivo original de marcas, 1,2,3?

% Do several sanity checks for the insertion of marks and stages along the eeg signal