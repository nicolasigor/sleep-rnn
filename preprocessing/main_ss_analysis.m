%% Definitions
addpath('utils_nt');
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
work_index = 1:m;

%% Reader of registers
eegData = cell(1,m);
for k = work_index
    eegData{k} = readEEG( allNames{k} );
end

%% Reader of labels
save_newmarks = 1;
for k = work_index
    eegData{k}.label = readLabel( allNames{k} ,eegData{k}.params.fs, save_newmarks);
end

%% Segmentation of labels
save_segments_label = 1;
for k = work_index
    eegData{k}.segments = getSegments( allNames{k}, save_segments_label, eegData{k}.label.params);
end

%% Computation of features for each time-step

database = cell(1,max(work_index));
for k = work_index
    features = [];
    labels = [];
    
    intervals = eegData{k}.segments.intervals;
    n_epoch_in_segments = eegData{k}.segments.n_epoch_in_segments;
    n_segments = length(n_epoch_in_segments);
    marks = eegData{k}.segments.marks;
    
    cut = [0.5, 40];
    [B, A] = butter(3, cut/(eegData{k}.label.params.fs/2) , 'bandpass');
    eegRecordFiltered = filtfilt( B, A, eegData{k}.eegRecord);
    n_max = length(eegRecordFiltered);
    for i = 1:n_segments
        timestep_in_interval = epoch2timestep(intervals(i,:),eegData{k}.label.params);
        timestep_in_interval(2) = min(timestep_in_interval(2), n_max);
        offset = timestep_in_interval(1)-1;
        samples_in_interval = timestep_in_interval(2) - offset;
        aux = zeros(samples_in_interval,2);
        aux(:,1) = i;
        aux(:,2) = eegRecordFiltered(timestep_in_interval(1):timestep_in_interval(2));
        features = cat(1, features, aux);
    end
    
    for i = 1:n_segments
        timestep_in_interval = epoch2timestep(intervals(i,:),eegData{k}.label.params);
        timestep_in_interval(2) = min(timestep_in_interval(2), n_max);
        offset = timestep_in_interval(1)-1;
        samples_in_interval = timestep_in_interval(2) - offset;
        aux = zeros(samples_in_interval,2);
        aux(:,1) = i;
        for j = 1:length(marks{i})
            aux_marks = marks{i}(j,:) - offset;
            aux(aux_marks(1):aux_marks(2), 2) = 1;
        end
        labels = cat(1, labels, aux);
    end

    % Save feats and labels for this register
    fs = eegData{k}.label.params.fs;
    database{k}.features = features;
    database{k}.labels = labels;
    database{k}.fs = fs;
end
fprintf('Finish database creation\n')


%% Print Stats

show_index = 1:m;

% Print stats of marks

fprintf('\nID Name         Total    CH1   NoVal0  NoDur0   NoRep  Valid(Incl.)    P1 (minDist)    P2 (short,long)      P3 ( N2, N3, trans)\n');
total_n2_marks = 0;
for k = show_index
    st = eegData{k}.label.marks_stats;
    total_n2_marks = total_n2_marks + st.n_marks_statecontrol_n2only;
    fprintf('%2.1d %s %7.d %7.d %7.d %7.d %7.d %7.d (%4.d) %7.d (%5.1f) %7.d(%3.1d,%3.1d) %7.d (%5.1d,%5.1d,%5.1d)\n',...
        k,allNames{k},st.n_marks_file, st.n_marks_ch1,st.n_marks_no_val0, st.n_marks_no_dur0, st.n_marks_no_rep,...
        st.n_marks_valid, st.n_marks_valid_included, st.n_marks_aftercomb,st.n_marks_aftercomb_minDist,st.n_marks_durationcontrol,...
        st.n_marks_durationcontrol_too_short,st.n_marks_durationcontrol_too_long,...
        st.n_marks_statecontrol_n2n3,st.n_marks_statecontrol_n2only,st.n_marks_statecontrol_n3only,...
        st.n_marks_statecontrol_ntrans);    
end
fprintf('Total number of N2-only marks: %d\n',total_n2_marks);

% Print stats of hipnogram

total_n2_epochs = 0;
fprintf('\nID Name        Epochs   N1 Ep   N2 Ep   N3 Ep    R Ep    W Ep \n');
for k = show_index
    Twin = eegData{k}.label.params.epochDuration;
    ns = zeros(5,1);
    ns(1) = sum( eegData{k}.label.states == 4 );
    ns(2) = sum( eegData{k}.label.states == 3 );
    ns(3) = sum( eegData{k}.label.states == 2 );
    ns(4) = sum( eegData{k}.label.states == 5 );
    ns(5) = sum( eegData{k}.label.states == 6 );
    fprintf('%2.1d %s %7.d %7.d %7.d %7.d %7.d %7.d\n',...
        k,allNames{k},sum(ns),ns(1),ns(2),ns(3),ns(4),ns(5));
    total_n2_epochs = total_n2_epochs + ns(2);
end
fprintf('Total number of N2 epochs: %d, i.e. %1.2f hrs\n',total_n2_epochs,total_n2_epochs*Twin/3600);

%% Stats Segmentation of N2

fprintf('\n Segmentation Stats\n');

global_segments = [];
global_segments_marks = [];
warning_zero_marks = {};

show_individual_figure = 0;
for k = show_index    
    global_segments = cat(1, global_segments, eegData{k}.segments.intervals);
    n_marks = cellfun(@length, eegData{k}.segments.marks);
    global_segments_marks = cat(1, global_segments_marks,n_marks);
    for i = 1:length(n_marks)
        if n_marks(i) == 0
            warning_zero_marks{end+1} = sprintf('Warning! segment %d with %3.d epochs and 0 marks in %s\n',...
                i, eegData{k}.segments.n_epoch_in_segments(i),allNames{k}); 
        end
    end
    fprintf('%2.d segments found in %s with %5.d marks and %4.d epochs\n',...
        length(n_marks),allNames{k},sum(n_marks),sum(eegData{k}.segments.n_epoch_in_segments));
    
    if show_individual_figure
        figure
        
        subplot(3,1,1), area(eegData{1}.label.states==3),
        xlabel('Epoch'),title(sprintf('%s N2 segments',allNames{k}))
        subplot(3,1,2), bar(eegData{k}.segments.n_epoch_in_segments')
        xlabel('Segment ID'), ylabel('Epochs')
        subplot(3,1,3), bar((n_marks./eegData{k}.segments.n_epoch_in_segments)')
        xlabel('Segment ID'), ylabel('Marks per Epoch')
    end
end

fprintf('\n')
for i = 1:length(warning_zero_marks)
   fprintf(warning_zero_marks{i});
end

% Plot global segments

n_epochs_in_segment = (global_segments(:,2)-global_segments(:,1)+1);

figure,
subplot(2,1,1), bar(n_epochs_in_segment')
xlabel('Segment ID'), ylabel('Epochs'),title('Complete database segments')
subplot(2,1,2), bar(global_segments_marks./n_epochs_in_segment)
xlabel('Segment ID'), ylabel('Marks per Epoch')

% Histogram of length of segments and marks per epoch

figure
histogram(n_epochs_in_segment',20)
title('Length of N2 segment'), xlabel('Epochs'), ylabel('Count')
figure
histogram(global_segments_marks./n_epochs_in_segment,20)
title('Marks per epoch in N2 segment'), xlabel('Marks per Epoch'), ylabel('Count')

useful = global_segments_marks > 0;
fprintf('\n%d segments with %d epochs in total that have marks\n',...
    length(global_segments(useful)), sum(n_epochs_in_segment(useful)))


%% Statistics of segments of individual EEG
figure
for ind = show_index  
    clf
    individual_features = database{ind}.features(:,2);
    outlier_thr = prctile(abs(individual_features),99);
    fprintf('99 percent of data in register %d has magnitude less or equal than %1.2f\n',ind,outlier_thr);
    individual_features_clip = individual_features;
    individual_features_clip(individual_features>outlier_thr) = outlier_thr;
    individual_features_clip(individual_features<-outlier_thr) = - outlier_thr;
    individual_features_clip_norm = (individual_features_clip - mean(individual_features_clip))/std(individual_features_clip);
    subplot(3,1,1)
    histogram(individual_features,20),xlabel('Voltage [\muV]')
    title(sprintf('EEG Register %d',ind));
    subplot(3,1,2)
    histogram(individual_features_clip,20),xlabel('Voltage [\muV]')
    title(sprintf('EEG Register %d, Clipped percentil 99',ind));
    subplot(3,1,3)
    histogram(individual_features_clip_norm,20),xlabel('Amplitude')
    title(sprintf('EEG Register %d, Clipped percentil 99, Normalized',ind));
    pause;
end

%% Find outliers segments
fprintf('REG  SEGM  MIN  MAX\n')
for ind = 1:11
    available_segments = length(eegData{ind}.segments.intervals);
    for segment = 1:available_segments
        segment_features = database{ind}.features(database{ind}.features(:,1)==segment,2);
        min_value = min(segment_features);
        max_value = max(segment_features);
        fprintf('%3.1d %3.1d %8.2f %8.2f\n',ind,segment,min_value,max_value);
    end
end

% 2   8 -1861.61  1105.69
% 2   9 -1028.88  3975.20
% 5   3 -2843.23   957.60
% 5   8 -1953.70  1059.49
% 7   2 -1176.24  1941.78
% 7   3 -5160.09  2867.94
% 7   4 -4664.16  3859.89
% 7   5 -1536.43  1226.75
% 9   6 -1338.60  1270.80
% 10   1 -1252.31  2164.02
% 10   6 -2133.88  2284.35
% 11   5 -1868.35  1267.18
   
%% Visualization of EEG
plot_eeg = 1;
ind = 10;
segment = 7;
available_segments = length(eegData{ind}.segments.intervals);
available_epochs = eegData{ind}.segments.n_epoch_in_segments(segment);
fprintf('Number of available epochs: %d\n',available_epochs);
fprintf('Number of available segments: %d\n',available_segments);

segment_features = database{ind}.features(database{ind}.features(:,1)==segment,2);
segment_labels = database{ind}.labels(database{ind}.labels(:,1)==segment,2);
segment_intervals = eegData{ind}.segments.intervals(segment,:);
segment_timesteps = epoch2timestep(segment_intervals,eegData{ind}.label.params);
segment_timesteps(2) = min(segment_timesteps(2), length(eegData{ind}.eegRecord));
segment_timeinterval = (segment_timesteps(1):segment_timesteps(2))';
segment_timeinterval = segment_timeinterval/eegData{ind}.label.params.fs;
segment_timeinterval = segment_timeinterval/60; % [min]
min_value = min(segment_features);
max_value = max(segment_features);
if plot_eeg
    figure
    for i = 1:available_epochs
        clf
        epoch = i;
        epoch_width = 6000;
        start_time = (epoch-1)*epoch_width + 1;
        end_time = epoch*epoch_width;
        end_time = min(end_time, length(segment_features));
        restriction = start_time:end_time;
        hold on
        plot(segment_timeinterval(restriction),segment_features(restriction));
        plot(segment_timeinterval(restriction),200*segment_labels(restriction))
        hold off
        title(sprintf('Register %d, Segment %d, Relative Epoch %d',ind,segment,epoch))
        xlabel('Time [min]')
        ylim([min_value,max_value])
        pause;
    end
end
