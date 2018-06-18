%% notes:
% maybe the algorithms should be tuned for this database and should be
% encouraged to find spindles only in N2 for now.
% We should obtain an operation curve for each detector rather than a
% single point. For the curve, only use the average across registers
% what's the deal with tsanas :( 
% Leave for later newer baselines

%% Settings

n_reg = 11;
n_models = 6;

% Index representing N2 and N3 stages in hypnograms
n2_val = 3;
n3_val = 2;

%% Ground truth
groundTruth.marks_events = cell(n_reg,1);
groundTruth.marks_samples = cell(n_reg,1);
for ind = 1:n_reg
    marks_events = eegData{ind}.label.marks;
    marks_samples = zeros(size(eegData{ind}.eegRecord,1),1);
    for i = 1:size(marks_events,1)
        marks_samples( marks_events(i,1):marks_events(i,2) ) = 1;
    end
    groundTruth.marks_events{ind} = marks_events;
    groundTruth.marks_samples{ind} = marks_samples;
end
fprintf('Ground Truth loaded.\n')
%% Clipped registers and states
clipped_eeg = cell(n_reg,1);
states = cell(n_reg,1);
for ind = 1:n_reg
    allnight = eegData{ind}.eegRecord;
    outlier_thr = prctile(abs(allnight),99);
    clipped_allnight = allnight;
    clipped_allnight( allnight>outlier_thr ) = outlier_thr;
    clipped_allnight( allnight<-outlier_thr ) = -outlier_thr;
    clipped_eeg{ind} = clipped_allnight;
    states{ind} = eegData{ind}.label.states; % Now  2:N3  3:N2  4:N1  5:R  6:W
end
fprintf('Registers and states loaded.\n')
%% Model evaluations
publishedModels = cell(n_models,1);

%% Warby A2
model_index = 1;
publishedModels{model_index}.name = 'Warby A2';
publishedModels{model_index}.detection_samples = cell(n_reg,1);
publishedModels{model_index}.detection_events = cell(n_reg,1);
for ind = 1:n_reg
    fprintf('|');
    n_total = length(clipped_eeg{ind});
    % Needs a specific stage file % Now  2:N3  3:N2  4:N1  5:R  6:W
    stage_file = states{ind};
    stage_file(stage_file==4) = 1; % 4:N1
    stage_file(stage_file==3) = 2; % 3:N2
    stage_file(stage_file==2) = 3; % 2:N3
    stage_file = [stage_file, stage_file];    
    [detection_samples, detection_events] = warby2014_a2_spindle_detection(clipped_eeg{ind},set.fs,stage_file);
    publishedModels{model_index}.detection_samples{ind} = detection_samples;
    publishedModels{model_index}.detection_events{ind} = detection_events;
end
fprintf('%s evaluated.\n',publishedModels{model_index}.name)

%% Warby A3
model_index = 2;
publishedModels{model_index}.name = 'Warby A3';
publishedModels{model_index}.detection_samples = cell(n_reg,1);
publishedModels{model_index}.detection_events = cell(n_reg,1);
for ind = 1:n_reg
    fprintf('|');
    n_total = length(clipped_eeg{ind});
    % Needs N2 segments
    allowed_epochs = (states{ind} == n2_val);
    allowed_epochs_inter = seq2inter(allowed_epochs);
    n_segments = size(allowed_epochs_inter,1);
    only_n2 = cell(n_segments,1);
    for i = 1:n_segments
       samples = epoch2timestep(allowed_epochs_inter(i,:),set);
       samples(2) = min(samples(2),n_total);
       only_n2{i} = clipped_eeg{ind}(samples(1):samples(2)); 
    end
    [detection_samples, detection_events] = warby2014_a3_spindle_detection(only_n2,clipped_eeg{ind},set.fs);
    publishedModels{model_index}.detection_samples{ind} = detection_samples;
    publishedModels{model_index}.detection_events{ind} = detection_events;
end
fprintf('%s evaluated.\n',publishedModels{model_index}.name)

%% Warby A4
model_index = 3;
publishedModels{model_index}.name = 'Warby A4';
publishedModels{model_index}.detection_samples = cell(n_reg,1);
publishedModels{model_index}.detection_events = cell(n_reg,1);
for ind = 1:n_reg
    fprintf('|');
    n_total = length(clipped_eeg{ind});
    % Needs N2+N3 segments
    allowed_epochs = any((states{ind} == [n2_val,n3_val])');
    allowed_epochs_inter = seq2inter(allowed_epochs);
    n_segments = size(allowed_epochs_inter,1);
    only_n2n3 = cell(n_segments,1);
    for i = 1:n_segments
       samples = epoch2timestep(allowed_epochs_inter(i,:),set);
       samples(2) = min(samples(2),n_total);
       only_n2n3{i} = clipped_eeg{ind}(samples(1):samples(2)); 
    end
    [detection_samples, detection_events] = warby2014_a4_spindle_detection(only_n2n3,clipped_eeg{ind},set.fs);
    publishedModels{model_index}.detection_samples{ind} = detection_samples;
    publishedModels{model_index}.detection_events{ind} = detection_events;
end
fprintf('%s evaluated.\n',publishedModels{model_index}.name)

%% Warby A5
model_index = 4;
publishedModels{model_index}.name = 'Warby A5';
publishedModels{model_index}.detection_samples = cell(n_reg,1);
publishedModels{model_index}.detection_events = cell(n_reg,1);
for ind = 1:n_reg
    fprintf('|');
    n_total = length(clipped_eeg{ind});
    % Needs N2 segments
    allowed_epochs = (states{ind} == n2_val);
    allowed_epochs_inter = seq2inter(allowed_epochs);
    n_segments = size(allowed_epochs_inter,1);
    only_n2 = cell(n_segments,1);
    for i = 1:n_segments
       samples = epoch2timestep(allowed_epochs_inter(i,:),set);
       samples(2) = min(samples(2),n_total);
       only_n2{i} = clipped_eeg{ind}(samples(1):samples(2)); 
    end
    [detection_samples, detection_events] = warby2014_a5_spindle_detection(only_n2,clipped_eeg{ind},set.fs);
    publishedModels{model_index}.detection_samples{ind} = detection_samples;
    publishedModels{model_index}.detection_events{ind} = detection_events;
end
fprintf('%s evaluated.\n',publishedModels{model_index}.name)

 %% Tsanas A7
% model_index = 5;
% publishedModels{model_index}.name = 'Tsanas A7';
% publishedModels{model_index}.detection_samples = cell(n_reg,1);
% publishedModels{model_index}.detection_events = cell(n_reg,1);
% for ind = 1:n_reg
%     fprintf('|');
%     [detection_samples, detection_events] = tsanas2015_a7_spindle_detection(clipped_eeg{ind},set.fs);
%     publishedModels{model_index}.detection_samples{ind} = detection_samples;
%     publishedModels{model_index}.detection_events{ind} = detection_events;
% end
% fprintf('%s evaluated.\n',publishedModels{model_index}.name)

%% Performance By-sample (N2 only)

show_models = [1,2,3,4];
for model_index = show_models
    publishedModels{model_index}.metrics = cell(n_reg,1);
    publishedModels{model_index}.details = cell(n_reg,1);
    for ind = 1:n_reg  
        % Need N2 samples only in detections
        only_n2_index = [];
        n_total = length(clipped_eeg{ind});
        n2_segments = seq2inter(states{ind} == n2_val);
        for i = 1:size(n2_segments,1)
           samples = epoch2timestep(n2_segments(i,:),set);
           samples(2) = min(samples(2),n_total);
           only_n2_index = cat(2,only_n2_index,samples(1):samples(2)); 
        end         
        ground_truth = groundTruth.marks_samples{ind};
        detection = publishedModels{model_index}.detection_samples{ind};
        [metrics, details] = by_sample_performance(ground_truth(only_n2_index), detection(only_n2_index));
        publishedModels{model_index}.metrics{ind} = metrics;
        publishedModels{model_index}.details{ind} = details;
    end
    precision_array = cellfun(@(c) c.precision, publishedModels{model_index}.metrics);
    recall_array = cellfun(@(c) c.recall, publishedModels{model_index}.metrics);
    f1_score_array = cellfun(@(c) c.f1_score, publishedModels{model_index}.metrics);
    publishedModels{model_index}.overall_metrics.precision = [mean(precision_array), std(precision_array)];
    publishedModels{model_index}.overall_metrics.recall = [mean(recall_array),std(recall_array)];
    publishedModels{model_index}.overall_metrics.f1_score = [mean(f1_score_array), std(f1_score_array )];
    fprintf('%s by-sample performance evaluated.\n',publishedModels{model_index}.name)
end

%% Show performance By-Sample

show_models = [1,2,3,4];
figure

% Each register separated
subplot(1,2,1)
axis square
xlabel('Recall (1-FNR)'), ylabel('Precision (1-FDR)')
xlim([0,1]),ylim([0,1])
title('By-Subject By-Sample Performance')
legend('Location','eastoutside');
hold on
for model_index = show_models
    model_name = publishedModels{model_index}.name;
    recall = cellfun(@(c) c.recall, publishedModels{model_index}.metrics);
    precision = cellfun(@(c) c.precision, publishedModels{model_index}.metrics);
    scatter(recall,precision,25,'Fill','DisplayName',model_name);
end
% Show F1_Score contour lines
x = 0:0.01:1;
y = 0:0.01:1;
[X,Y] = meshgrid(x,y);
Z = 2*X.*Y./(X+Y);
contour(X,Y,Z,'ShowText','on','TextList',[0.1,0.9],'Color',0.6*[1,1,1],'LineStyle','--','HandleVisibility','off')
hold off

% Average across registers 
subplot(1,2,2)
axis square
xlabel('Recall (1-FNR)'), ylabel('Precision (1-FDR)')
xlim([0,1]),ylim([0,1])
title('Average By-Sample Performance')
legend('Location','eastoutside');
hold on
fprintf('\nModel       Precision        Recall           F1-Score\n')
for model_index = show_models
    model_name = publishedModels{model_index}.name;
    mean_recall = publishedModels{model_index}.overall_metrics.recall(1);
    mean_precision = publishedModels{model_index}.overall_metrics.precision(1);
    mean_f1_score = publishedModels{model_index}.overall_metrics.f1_score(1);
    std_recall = publishedModels{model_index}.overall_metrics.recall(2);
    std_precision = publishedModels{model_index}.overall_metrics.precision(2);
    std_f1_score = publishedModels{model_index}.overall_metrics.f1_score(2);   
    fprintf('%s %8.2f (%5.2f) %8.2f (%5.2f) %8.2f (%5.2f)\n',...
        model_name,100*mean_precision,100*std_precision,...
        100*mean_recall, 100*std_recall,...
        100*mean_f1_score, 100*std_f1_score)   
    theta = linspace(0,2*pi,100);
    x = mean_recall + std_recall*cos(theta);
    y = mean_precision + std_precision*sin(theta);
    patch(x,y,'green','FaceColor','black','FaceAlpha',.2,'EdgeColor','none','HandleVisibility','off')
    scatter(mean_recall,mean_precision,25,'Fill','DisplayName',model_name);
    text(mean_recall+0.01,mean_precision,sprintf('%1.2f',mean_f1_score),'HorizontalAlignment','left')     
end
% Show F1_Score contour lines
x = 0:0.01:1;
y = 0:0.01:1;
[X,Y] = meshgrid(x,y);
Z = 2*X.*Y./(X+Y);
contour(X,Y,Z,'ShowText','on','TextList',[0.1,0.9],'Color',0.6*[1,1,1],'LineStyle','--','HandleVisibility','off')
hold off

%% Precision-Recall Curve (several operation points)

%% Performance By-Event (N2 only)

%% Show performance By-Event 