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


%% Read all registers

% Names
allNames = {'ADGU101504' 
    'ALUR012904' 
    'BECA011405' 
    'BRCA062405' 
    'BRLO041102' 
    'BTOL083105' 
    'BTOL090105' 
    'CAPO092605' 
    'CRCA020205' 
    'ESCI031905' 
    'TAGO061203'}; 
m = length(allNames);
eegData = cell(1,m);
for k = 1:m
    eegData{k} = getSleepEEG( allNames{k} );
end

%% Print statistics about marks in each register

fprintf('\nID Name         Total    CH1   PartVal  NonZero  Valid    P1 (minDist)    P2 (short,long)      P3 ( N2, N3, trans)\n');
total_n2_marks = 0;
for k = 1:m
    st = eegData{k}.marks_steps;
    fprintf('%2.1d %s %7.d %7.d %7.d %7.d %7.d %7.d (%5.1f) %7.d(%3.1d,%3.1d) %7.d (%5.1d,%5.1d,%5.1d)\n',...
        k,allNames{k},st.n_marks_file, st.n_marks_ch1,st.n_marks_without0, st.n_marks_positive, st.n_marks_valid,...
        st.n_marks_aftercomb,st.n_marks_aftercomb_minDist,st.n_marks_durationcontrol,...
        st.n_marks_durationcontrol_too_short,st.n_marks_durationcontrol_too_long,...
        st.n_marks_n2n3,st.n_marks_n2only,st.n_marks_n3only,st.n_marks_trans);
    total_n2_marks = total_n2_marks + st.n_marks_n2only;
end
fprintf('Total number of N2-only marks: %d\n',total_n2_marks);
% Print statistics about epochs in each register
total_n2 = 0;
fprintf('\nID Name        Epochs   N1 Ep   N2 Ep   N3 Ep    R Ep    W Ep \n');
for k = 1:m
    Twin = eegData{k}.params.epochDuration;
    ns = zeros(5,1);
    ns(1) = sum( eegData{k}.regStates == 4 );
    ns(2) = sum( eegData{k}.regStates == 3 );
    ns(3) = sum( eegData{k}.regStates == 2 );
    ns(4) = sum( eegData{k}.regStates == 5 );
    ns(5) = sum( eegData{k}.regStates == 6 );
    fprintf('%2.1d %s %7.d %7.d %7.d %7.d %7.d %7.d\n',...
        k,allNames{k},sum(ns),ns(1),ns(2),ns(3),ns(4),ns(5));
    total_n2 = total_n2 + ns(2);
end
fprintf('Total number of N2 epochs: %d, i.e. %1.2f hrs\n',total_n2,total_n2*30/3600);
% Statistics about N2 segments

global_segments = [];
global_segments_marks = [];

for k = 1:m
    segments = [];
    start_segm = -1;
    end_segm = 0;
    states = eegData{k}.regStates;
    ind_n2 = states == 3;
    n = length(states);
    for i = 1:n
        if ind_n2(i) && end_segm>start_segm
            start_segm = i;         
        elseif ~ind_n2(i) && end_segm<start_segm
            end_segm = i-1;
            segments = [segments; start_segm,end_segm];
            global_segments = [global_segments; start_segm,end_segm];
        end
    end
    
    n_seg = length(segments);
    segments_marks = zeros(1,n_seg);
    marks = eegData{k}.marks;
    fprintf('%d marks in this register\n',length(marks)); %
    Twin = eegData{k}.params.epochDuration;
    fs = eegData{k}.params.fs;
    marks_pages = floor( marks/(Twin*fs) ) + 1;
    fprintf('%d marks in this register\n',length(marks_pages)); %
    for i = 1:n_seg       
        inside = marks_pages(:,1)>=segments(i,1) & marks_pages(:,2) <= segments(i,2);
        segments_marks(i) = sum(inside);
        if segments_marks(i) == 0
           fprintf('Warning! N2-Segment with 0 marks in %s\n',allNames{k}); 
        end
        global_segments_marks = [global_segments_marks; sum(inside)];
    end
    fprintf('%d segments found in %s with %d marks\n',length(segments),allNames{k},sum(segments_marks));
%     figure,
%     subplot(3,1,1), area(states==3),
%     xlabel('Epoch'),title(sprintf('%s N2 segments',allNames{k}))
%     subplot(3,1,2), bar((segments(:,2)-segments(:,1)+1)')
%     xlabel('Segment ID'), ylabel('Epochs')
%     subplot(3,1,3), bar(segments_marks./(segments(:,2)-segments(:,1)+1)')
%     xlabel('Segment ID'), ylabel('Marks per Epoch')
end


%% Plot of segments

figure,
subplot(2,1,1), bar((global_segments(:,2)-global_segments(:,1))')
xlabel('Segment ID'), ylabel('Epochs'),title('Complete database segments')
subplot(2,1,2), bar(global_segments_marks./(global_segments(:,2)-global_segments(:,1)))
xlabel('Segment ID'), ylabel('Marks per Epoch')

%% Histogram of length of segments and marks per epoch

figure,histogram((global_segments(:,2)-global_segments(:,1))',10),title('Length of N2 segment')
figure,histogram(global_segments_marks./(global_segments(:,2)-global_segments(:,1)),10), title('Marks per epoch in N2 segment')
