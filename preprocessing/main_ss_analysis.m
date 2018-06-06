%% Definitions

% Path of functions
addpath('utils_nt');

% Files to be read
all_names = {'ADGU101504' 
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
rec_folder = 'ssdata/register';
spindles_folder = 'ssdata/label/marks';
states_folder = 'ssdata/label/states';

% Indexes
m = length(all_names);
work_index = 1:m;
show_index = 1:m;

% Settings
set.channel = 1;
set.dur_epoch = 30;        % Time of window page [s]
set.dur_min_ss = 0.3;      % Min SS duration [s]
set.dur_max_ss = 3.0;      % Max SS duration [s]
% According to literature, SS durations are most commonly encountered
% between 0.5 and 2 seconds, but here we consider a broader range

% Handler for histogram function (due to issues of retro-compatibility when
% using the newer 'histogram' function
use_hist = 1;
if use_hist
   my_histogram = @hist; 
else
   my_histogram = @histogram;
end

%% Reader of registers
eegData = cell(1,m);
for k = work_index
    rec_filename = [rec_folder '/' all_names{k} '.rec'];
    eegData{k} = readEEG( rec_filename , set.channel);
end
% Save sampling frequency (they all have the same one);
set.fs = eegData{1}.params.fs;

%% OPTIONAL: Cleaning of Marks
clean_stats = cell(1,m);
for k = work_index
    marks_filename = [spindles_folder '/SS_' all_names{k} '.txt'];
    marks_filename_new = [spindles_folder '/FixedSS_' all_names{k} '.txt'];
    clean_stats{k} = cleanExpertMarks( marks_filename, marks_filename_new, set);
end

%% Reader of labels
for k = work_index
    %marks_filename = [spindles_folder '/SS_' all_names{k} '.txt'];
    marks_filename = [spindles_folder '/FixedSS_' all_names{k} '.txt'];
    states_filename = [states_folder '/StagesOnly_' all_names{k} '.txt']; 
    eegData{k}.label = readLabel( marks_filename, states_filename, set.channel);
end

%% Segmentation of epochs
% Only N2 epochs will be selected.
% A small context is considered before and after the epoch (3s)

% in a epoch, save id (reg, seg, ep), eeg, marks in that epoch.
eegSegmentation = cell(1,m);
for k = work_index

end


%% Normalization of epochs
% Z-score by-individual, mean and std computed considering percentile 99



%% Selection of epochs
% Epochs without outliers values and with marks will be selected
% Count number of removed epochs and available
% epochs for training.


%% Spectrogram of selected epochs
% Compute a single spectrogram for each epoch. Start with one epoch to
% experiment with.



%% Generate database for machine learning model
% Generate a simple matrix DB for the non-context case
% Generate something else for the context case



%% Segmentation of labels
save_segments_label = 0;
for k = work_index
    eegData{k}.segments = getSegments( all_names{k}, save_segments_label, params);
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
        for j = 1:size(marks{i},1)
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

%%
% Print stats of marks

fprintf('\nID Name         Total    CH1   NoVal0  NoDur0   NoRep  Valid(Incl.)    P1 (minDist)    P2 (short,long)      P3 ( N2, N3, trans)\n');
total_n2_marks = 0;
for k = show_index
    st = eegData{k}.label.marks_stats;
    total_n2_marks = total_n2_marks + st.n_marks_statecontrol_n2only;
    fprintf('%2.1d %s %7.d %7.d %7.d %7.d %7.d %7.d (%4.d) %7.d (%5.4f) %7.d(%3.1d,%3.1d) %7.d (%5.1d,%5.1d,%5.1d)\n',...
        k,all_names{k},st.n_marks_file, st.n_marks_ch1,st.n_marks_no_val0, st.n_marks_no_dur0, st.n_marks_no_rep,...
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
        k,all_names{k},sum(ns),ns(1),ns(2),ns(3),ns(4),ns(5));
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
                i, eegData{k}.segments.n_epoch_in_segments(i),all_names{k}); 
        end
    end
    fprintf('%2.d segments found in %s with %5.d marks and %4.d epochs\n',...
        length(n_marks),all_names{k},sum(n_marks),sum(eegData{k}.segments.n_epoch_in_segments));
    
    if show_individual_figure
        figure
        
        subplot(3,1,1), area(eegData{1}.label.states==3),
        xlabel('Epoch'),title(sprintf('%s N2 segments',all_names{k}))
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
my_histogram(n_epochs_in_segment',20)
title('Length of N2 segment'), xlabel('Epochs'), ylabel('Count')
figure
my_histogram(global_segments_marks./n_epochs_in_segment,20)

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
    my_histogram(individual_features,20),xlabel('Voltage [\muV]')
    title(sprintf('EEG Register %d',ind));
    subplot(3,1,2)
    my_histogram(individual_features_clip,20),xlabel('Voltage [\muV]')
    title(sprintf('EEG Register %d, Clipped percentil 99',ind));
    subplot(3,1,3)
    my_histogram(individual_features_clip_norm,20),xlabel('Amplitude')
    title(sprintf('EEG Register %d, Clipped percentil 99, Normalized',ind));
    pause;
end

%% Find outliers segments
fprintf('REG  SEGM  MIN  MAX\n')
outlier_thr = 400;
for ind = 1:11
    available_segments = size(eegData{ind}.segments.intervals,1);
    for segment = 1:available_segments
        segment_features = database{ind}.features(database{ind}.features(:,1)==segment,2);
        min_value = min(segment_features);
        max_value = max(segment_features);
        if max(abs(min_value),abs(max_value)) > outlier_thr
            fprintf('%3.1d %3.1d %8.2f %8.2f\n',ind,segment,min_value,max_value);
        end
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

% Set register and segment
ind = 1;
segment = 3;

% Flags
plot_eeg = 1;
crop_plot = 1;
normalization = 1;

fprintf('Reading from register %s\n',all_names{ind});
available_segments = length(eegData{ind}.segments.intervals);
fprintf('Number of available segments: %d\n',available_segments);
available_epochs = eegData{ind}.segments.n_epoch_in_segments(segment);
fprintf('Number of available epochs in segment %d: %d\n',segment,available_epochs);

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
markplot = 150;

if normalization
    % Normalization ignoring outliers
    reg_features = database{ind}.features(:,2);
    outlier_thr = prctile(abs(reg_features),99);
    reg_features_no_outlier = reg_features;
    reg_features_no_outlier(abs(reg_features)>outlier_thr) = nan;
    reg_mean = nanmean(reg_features_no_outlier);
    reg_std = nanstd(reg_features_no_outlier);
    segment_features = (segment_features - reg_mean)/reg_std;
    fprintf('Normalization with mean %1.2f and std %1.2f\n',reg_mean,reg_std);
    markplot = 3;
end

if crop_plot && ~normalization
    min_value = -200;
    max_value = 200;
elseif crop_plot && normalization
    min_value = -4;
    max_value = 4;  
end

if plot_eeg
    figure('position', [0, 200, 1366, 300]);
    for i = 1:available_epochs
        clf
        epoch = i;
        epoch_width = 6000;
        start_time = (epoch-1)*epoch_width + 1;
        end_time = epoch*epoch_width;
        end_time = min(end_time, length(segment_features));
        restriction = start_time:end_time;
        
        % Generate marks
        fill_marks_times = segment_labels(restriction);
        fill_marks_times = (seq2inter(fill_marks_times)-1)/(eegData{ind}.label.params.fs*60);
        offset = segment_timeinterval(restriction);
        offset = offset(1);
        fill_marks_times = offset + fill_marks_times;
        fill_areas_x = zeros(4 , size(fill_marks_times,1));
        fill_areas_x([1,4],:) = [fill_marks_times(:,1)'; fill_marks_times(:,1)'];
        fill_areas_x([2,3],:) = [fill_marks_times(:,2)'; fill_marks_times(:,2)'];
        fill_areas_y = zeros(4 , size(fill_marks_times,1));
        fill_areas_y([1,2], :) = min_value;
        fill_areas_y([3,4], :) = max_value;
        % Plot
        hold on
        patch(fill_areas_x,fill_areas_y,[0 0 0],'FaceAlpha',.15,'EdgeColor','none')
        plot(downsample(segment_timeinterval(restriction),2),downsample(segment_features(restriction),2),'Color',[0 0.4470 0.7410]);
        hold off
        title(sprintf('Register %d, Segment %d, Relative Epoch %d',ind,segment,epoch))
        xlabel('Time in Register [min]')
        ylim([min_value,max_value])
        pause;
    end
end

%% Stats per epoch on a register

ind = 9;

states = eegData{ind}.label.states;
marks = eegData{ind}.label.marks;
marks_epoch = timestep2epoch( marks, eegData{ind}.label.params );
n_epoch = length(states);
marks_per_epoch = cell(n_epoch,1);
for i = 1:n_epoch
    marks_per_epoch{i} = marks( marks_epoch(:,1)==i , : );
end
n_marks_per_epoch = cellfun(@(x) size(x,1), marks_per_epoch);
time_eeg = (1:length(eegData{ind}.eegRecord)) / (eegData{ind}.label.params.fs*3600);
time_hypno = ((1:length(states))*30-15)/3600;

% Plot entire register
figure
subplot(3,1,1)
plot(time_eeg, eegData{ind}.eegRecord)
xlim([time_eeg(1),time_eeg(end)]) %, ylim([-400,400])
title(sprintf('Sleep EEG Recording %d (%1.1f hrs)',ind,length(eegData{ind}.eegRecord)/(3600*eegData{ind}.label.params.fs)))
ylabel('F4-C4 [\muV]')
subplot(3,1,2)
hold on 
area(time_hypno, 3*(states==3),'EdgeColor','none','FaceColor',[0 0.5 0.5])
plot(time_hypno, states,'LineWidth',1.5,'Color',[0 0.4470 0.7410])
hold off
xlim([time_eeg(1),time_eeg(end)])
ylim([1,7]), yticks([2,3,4,5,6])
ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'on';
% Now  2:N3  3:N2  4:N1  5:R  6:W
yticklabels({'N3','N2','N1','R','W'})
title(sprintf('Hypnogram (30s Epochs), %d Epochs in Stage N2',sum(states==3)))
subplot(3,1,3)
bar(time_hypno, n_marks_per_epoch)
title('Count of Sleep Spindle Instances per Epoch')
xlabel('Time [Hrs]')