%% notes:
% maybe the algorithms should be tuned for this database and should be
% encouraged to find spindles only in N2 for now.
% We should obtain an operation curve for each detector rather than a
% single point. For the curve, only use the average across registers
% what's the deal with tsanas :( 
% Leave for later newer baselines

%% Settings

n_reg = 11;
n_models = 4;

% Index representing N2 and N3 stages in hypnograms
n2_val = 3;
% n3_val = 2;

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
clear marks_events marks_samples
fprintf('Ground Truth loaded.\n')
%% Preparing data

newData = cell(n_reg,1);

for ind = 1:n_reg
    states = eegData{ind}.label.states;
    whole_night = eegData{ind}.eegRecord;
    n_total = length(whole_night);
    % N2 segments
    epochs = (states == n2_val);
    epochs_inter = seq2inter(epochs);
    n_segments = size(epochs_inter,1);
    n2 = cell(n_segments,1);
    for i = 1:n_segments
       samples = epoch2timestep(epochs_inter(i,:),set);
       samples(2) = min(samples(2),n_total);
       n2{i} = whole_night(samples(1):samples(2)); 
    end
    % Find clip value from N2
    outlier_thr = prctile(abs(cat(1,n2{:})),99);
	fprintf('Reg %d outlier thr: %1.4f\n',ind,outlier_thr);
    % Find mean and std without outliers from N2
    n2_concat_nan = cat(1,n2{:});
    n2_concat_nan(abs(n2_concat_nan)>outlier_thr) = nan;
    n2_mean = nanmean(n2_concat_nan);
    n2_std = nanstd(n2_concat_nan);
    % Let's clip and normalize the entire register
    
    % Whole Night
    whole_night_norm = whole_night;
    whole_night_norm( whole_night>outlier_thr ) = outlier_thr;
	whole_night_norm( whole_night<-outlier_thr ) = -outlier_thr;   
    whole_night_norm = (whole_night_norm - n2_mean) / n2_std;
    
    % N2 stages
    n2_norm = cell(n_segments,1);
    for i = 1:n_segments
       n2_norm{i} = n2{i};
       n2_norm{i}( n2{i}>outlier_thr ) = outlier_thr;
       n2_norm{i}( n2{i}<-outlier_thr ) = -outlier_thr;   
       n2_norm{i} = (n2_norm{i} - n2_mean) / n2_std;
    end

    newData{ind}.whole_night_norm = whole_night_norm;
    newData{ind}.n2_norm = n2_norm;
    newData{ind}.states = states;
    
end
clear whole_night n_total epochs epochs_inter n_segments n2
clear samples outlier_thr n2_concat_nan n2_mean n2_std
clear whole_night_norm n2_norm states
fprintf('Registers and states loaded.\n')

%% See what happened
% 7 es raro
ind = 7;
figure
subplot(1,2,1)
histogram(newData{ind}.whole_night_norm,50)
title('Whole Night')
subplot(1,2,2)
histogram(cat(1,newData{ind}.n2_norm{:}),50)
title('N2 only')

%% Model evaluations
publishedModels = cell(n_models,1);

%% Models

% Warby A2
model_index = 1;
publishedModels{model_index}.name = 'Warby A2';
publishedModels{model_index}.detection_samples = cell(n_reg,1);
publishedModels{model_index}.detection_events = cell(n_reg,1);
for ind = 1:n_reg
    fprintf('|');
    [detection_samples, detection_events] = warby2014_a2_spindle_detection(newData{ind}.whole_night_norm,set.fs,newData{ind}.states);
    publishedModels{model_index}.detection_samples{ind} = detection_samples;
    publishedModels{model_index}.detection_events{ind} = detection_events;
end
clear detection_samples detection_events
fprintf('%s evaluated.\n',publishedModels{model_index}.name)

% Warby A3
model_index = 2;
publishedModels{model_index}.name = 'Warby A3';
publishedModels{model_index}.detection_samples = cell(n_reg,1);
publishedModels{model_index}.detection_events = cell(n_reg,1);
for ind = 1:n_reg
    fprintf('|');
    [detection_samples, detection_events] = warby2014_a3_spindle_detection(newData{ind}.n2_norm, newData{ind}.whole_night_norm, set.fs);
    publishedModels{model_index}.detection_samples{ind} = detection_samples;
    publishedModels{model_index}.detection_events{ind} = detection_events;
end
clear detection_samples detection_events
fprintf('%s evaluated.\n',publishedModels{model_index}.name)

% Warby A4
model_index = 3;
publishedModels{model_index}.name = 'Warby A4';
publishedModels{model_index}.detection_samples = cell(n_reg,1);
publishedModels{model_index}.detection_events = cell(n_reg,1);
for ind = 1:n_reg
    fprintf('|');
    [detection_samples, detection_events] = warby2014_a4_spindle_detection(newData{ind}.n2_norm, newData{ind}.whole_night_norm, set.fs);
    publishedModels{model_index}.detection_samples{ind} = detection_samples;
    publishedModels{model_index}.detection_events{ind} = detection_events;
end
clear detection_samples detection_events
fprintf('%s evaluated.\n',publishedModels{model_index}.name)
%%
% Warby A5
model_index = 4;
publishedModels{model_index}.name = 'Warby A5';
publishedModels{model_index}.detection_samples = cell(n_reg,1);
publishedModels{model_index}.detection_events = cell(n_reg,1);
for ind = 1:n_reg
    fprintf('|');
    [detection_samples, detection_events] = warby2014_a5_spindle_detection(newData{ind}.n2_norm, newData{ind}.whole_night_norm, set.fs);
    publishedModels{model_index}.detection_samples{ind} = detection_samples;
    publishedModels{model_index}.detection_events{ind} = detection_events;
end
clear detection_samples detection_events
fprintf('%s evaluated.\n',publishedModels{model_index}.name)


%% Performance By-sample (N2 only)

show_models = [1,2,3,4];
for model_index = show_models
    publishedModels{model_index}.metrics_samples = cell(n_reg,1);
    publishedModels{model_index}.details_samples = cell(n_reg,1);
    for ind = 1:n_reg  
    % Need N2 samples only in detections
        only_n2_index = [];
        n_total = length(newData{ind}.whole_night_norm);
        n2_segments = seq2inter(newData{ind}.states == n2_val);
        for i = 1:size(n2_segments,1)
           samples = epoch2timestep(n2_segments(i,:),set);
           samples(2) = min(samples(2),n_total);
           only_n2_index = cat(2,only_n2_index,samples(1):samples(2)); 
        end         
        ground_truth = groundTruth.marks_samples{ind};
        detection = publishedModels{model_index}.detection_samples{ind};
        [metrics, details] = by_sample_performance(ground_truth(only_n2_index), detection(only_n2_index));
        publishedModels{model_index}.metrics_samples{ind} = metrics;
        publishedModels{model_index}.details_samples{ind} = details;
    end
%     precision_array = cellfun(@(c) c.precision, publishedModels{model_index}.metrics);
%     recall_array = cellfun(@(c) c.recall, publishedModels{model_index}.metrics);
%     f1_score_array = cellfun(@(c) c.f1_score, publishedModels{model_index}.metrics);
%     publishedModels{model_index}.overall_metrics.precision = [mean(precision_array), std(precision_array)];
%     publishedModels{model_index}.overall_metrics.recall = [mean(recall_array),std(recall_array)];
%     publishedModels{model_index}.overall_metrics.f1_score = [mean(f1_score_array), std(f1_score_array )];
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
    recall = cellfun(@(c) c.recall, publishedModels{model_index}.metrics_samples);
    precision = cellfun(@(c) c.precision, publishedModels{model_index}.metrics_samples);
    scatter(recall,precision,25,'Fill','DisplayName',model_name);
end
for model_index = show_models
    recall_1 = publishedModels{model_index}.metrics_samples{1}.recall;
    precision_1 = publishedModels{model_index}.metrics_samples{1}.precision;
    scatter(recall_1,precision_1,100,'HandleVisibility','off','MarkerEdgeColor',[0 0 0],'LineWidth',2);
end
% Show F1_Score contour lines
x = 0:0.01:1;
y = 0:0.01:1;
[X,Y] = meshgrid(x,y);
Z = 2*X.*Y./(X+Y);
contour(X,Y,Z,'ShowText','on','TextList',[0.1,0.9],'Color',0.6*[1,1,1],'LineStyle','--','HandleVisibility','off')
hold off

% Summing across registers 
subplot(1,2,2)
axis square
xlabel('Recall (1-FNR)'), ylabel('Precision (1-FDR)')
xlim([0,1]),ylim([0,1])
title('By-Sample Performance')
legend('Location','eastoutside');
hold on
fprintf('\nModel       Precision        Recall           F1-Score\n')
for model_index = show_models
    model_name = publishedModels{model_index}.name;
    TP_array = cellfun(@(c) c.TP, publishedModels{model_index}.details_samples);
    FP_array = cellfun(@(c) c.FP, publishedModels{model_index}.details_samples);
    FN_array = cellfun(@(c) c.FN, publishedModels{model_index}.details_samples);
    TP_all = sum(TP_array);
    FP_all = sum(FP_array);
    FN_all = sum(FN_array);
    precision_all = TP_all/(TP_all+FP_all);
    recall_all = TP_all/(TP_all+FN_all);
    f1_score_all = 2*(precision_all*recall_all)/(precision_all + recall_all) ;
    fprintf('%s %8.2f %16.2f %16.2f \n',...
        model_name,100*precision_all,...
        100*recall_all,...
        100*f1_score_all)
    scatter(recall_all,precision_all,25,'Fill','DisplayName',model_name);
    text(recall_all+0.01,precision_all,sprintf('%1.2f',f1_score_all),'HorizontalAlignment','left')
%     mean_recall = publishedModels{model_index}.overall_metrics.recall(1);
%     mean_precision = publishedModels{model_index}.overall_metrics.precision(1);
%     mean_f1_score = publishedModels{model_index}.overall_metrics.f1_score(1);
%     std_recall = publishedModels{model_index}.overall_metrics.recall(2);
%     std_precision = publishedModels{model_index}.overall_metrics.precision(2);
%     std_f1_score = publishedModels{model_index}.overall_metrics.f1_score(2);   
%     fprintf('%s %8.2f (%5.2f) %8.2f (%5.2f) %8.2f (%5.2f)\n',...
%         model_name,100*mean_precision,100*std_precision,...
%         100*mean_recall, 100*std_recall,...
%         100*mean_f1_score, 100*std_f1_score)   
%     theta = linspace(0,2*pi,100);
%     x = mean_recall + std_recall*cos(theta);
%     y = mean_precision + std_precision*sin(theta);
%     patch(x,y,'green','FaceColor','black','FaceAlpha',.2,'EdgeColor','none','HandleVisibility','off')
%     scatter(mean_recall,mean_precision,25,'Fill','DisplayName',model_name);
%     text(mean_recall+0.01,mean_precision,sprintf('%1.2f',mean_f1_score),'HorizontalAlignment','left')     
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
% Set threshold
thr = 0.2;
show_models = [1,2,3,4];

for model_index = show_models
    
    publishedModels{model_index}.metrics_event = cell(n_reg,1);
    publishedModels{model_index}.details_event = cell(n_reg,1);
    
    for ind = 1:n_reg  
    
        ground_truth = groundTruth.marks_events{ind};
        detection = publishedModels{model_index}.detection_events{ind};
        
        % We only need N2
        gs_epochs = timestep2epoch( ground_truth, set );
        ground_truth_n2 = [];
        for i = 1:size(gs_epochs,1)
            % If marks is completely inside N2 stage
            state_start = newData{ind}.states(gs_epochs(i,1));
            state_end = newData{ind}.states(gs_epochs(i,2)); 
           if state_start==n2_val && state_end==n2_val
               ground_truth_n2 = cat(1, ground_truth_n2, ground_truth(i,:));
           end
        end
        
        det_epochs = timestep2epoch( detection, set );
        detection_n2 = [];
        for i = 1:size(det_epochs,1)
            % If marks is completely inside N2 stage
            state_start = newData{ind}.states(det_epochs(i,1));
            state_end = newData{ind}.states(det_epochs(i,2)); 
           if state_start==n2_val && state_end==n2_val
               detection_n2 = cat(1, detection_n2, detection(i,:));
           end
        end
    
        [metrics, details] = by_event_performance(ground_truth_n2, detection_n2, thr);
        publishedModels{model_index}.metrics_events{ind} = metrics;
        publishedModels{model_index}.details_events{ind} = details;
    end
    fprintf('%s by-event performance evaluated.\n',publishedModels{model_index}.name)
end

%% Show overlap histogram a single model, a single individual

model_index = 1;
ind = 1;

overlaps = publishedModels{model_index}.details_events{ind}.overlaps;
ov_nonzero = overlaps(overlaps~=0);
ov_zero = overlaps(overlaps==0);
ufp = publishedModels{model_index}.metrics_events{ind}.UFP;
ufn = publishedModels{model_index}.metrics_events{ind}.UFN;
n_pairs = length(ov_nonzero);
figure

subplot(1,2,1)
h = histogram(ov_nonzero,'BinEdges',0:0.1:1);
ylim([0,n_pairs])
title(sprintf('Pairs:%d UFN: %d, UFP: %d, mean(Ov!=0)=%1.2f',n_pairs,ufn,ufp,mean(ov_nonzero)));
xlabel('Overlap');

subplot(1,2,2)
counts = h.Values;
bar(fliplr(cumsum(fliplr(counts))))
xlabel('Threshold')
ylabel('TP Count')
title(sprintf('Pairs:%d  UFN: %d, UFP: %d, mean(Ov!=0)=%1.2f',n_pairs,ufn,ufp,mean(ov_nonzero)));
xticklabels({'0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'})

%% Show performance By-Event Precision Recall Curve
show_models = [1,2,3,4];
figure

% Each register separated
subplot(1,2,1)
axis square
xlabel('Recall (1-FNR)'), ylabel('Precision (1-FDR)')
xlim([0,1]),ylim([0,1])
title('By-Subject By-Event Performance')
legend('Location','eastoutside');
hold on
for model_index = show_models
    model_name = publishedModels{model_index}.name;
    recall = cellfun(@(c) c.recall, publishedModels{model_index}.metrics_events);
    precision = cellfun(@(c) c.precision, publishedModels{model_index}.metrics_events);
    scatter(recall,precision,25,'Fill','DisplayName',model_name);
end
for model_index = show_models
    recall_1 = publishedModels{model_index}.metrics_events{1}.recall;
    precision_1 = publishedModels{model_index}.metrics_events{1}.precision;
    scatter(recall_1,precision_1,100,'HandleVisibility','off','MarkerEdgeColor',[0 0 0],'LineWidth',2);
end
% Show F1_Score contour lines
x = 0:0.01:1;
y = 0:0.01:1;
[X,Y] = meshgrid(x,y);
Z = 2*X.*Y./(X+Y);
contour(X,Y,Z,'ShowText','on','TextList',[0.1,0.9],'Color',0.6*[1,1,1],'LineStyle','--','HandleVisibility','off')
hold off

% Summing across registers 
subplot(1,2,2)
axis square
xlabel('Recall (1-FNR)'), ylabel('Precision (1-FDR)')
xlim([0,1]),ylim([0,1])
title('By-Event Performance')
legend('Location','eastoutside');
hold on
fprintf('\nModel       Precision        Recall           F1-Score\n')
for model_index = show_models
    model_name = publishedModels{model_index}.name;
    TP_array = cellfun(@(c) c.TPthr, publishedModels{model_index}.details_events);
    FP_array = cellfun(@(c) c.FPthr, publishedModels{model_index}.details_events);
    FN_array = cellfun(@(c) c.FNthr, publishedModels{model_index}.details_events);
    TP_all = sum(TP_array);
    FP_all = sum(FP_array);
    FN_all = sum(FN_array);
    precision_all = TP_all/(TP_all+FP_all);
    recall_all = TP_all/(TP_all+FN_all);
    f1_score_all = 2*(precision_all*recall_all)/(precision_all + recall_all) ;
    fprintf('%s %8.2f %16.2f %16.2f \n',...
        model_name,100*precision_all,...
        100*recall_all,...
        100*f1_score_all)
    scatter(recall_all,precision_all,25,'Fill','DisplayName',model_name);
    text(recall_all+0.01,precision_all,sprintf('%1.2f',f1_score_all),'HorizontalAlignment','left')  
end
% Show F1_Score contour lines
x = 0:0.01:1;
y = 0:0.01:1;
[X,Y] = meshgrid(x,y);
Z = 2*X.*Y./(X+Y);
contour(X,Y,Z,'ShowText','on','TextList',[0.1,0.9],'Color',0.6*[1,1,1],'LineStyle','--','HandleVisibility','off')
hold off

%% Statistics of by-event analysis

% Precision Recall Curve cuando el threshold se va variando.
% Valor del indicador de disagreement propuesto [x]
% Scatter duracion ground_truth vs overlap [x]
% Histograma de duracion para UFP y para UFN por separado [x]

%% Precision Recall for different thr

thr = [0, 0.2, 0.5, 0.75];
% Summing across registers 
figure
axis square
xlabel('Recall (1-FNR)'), ylabel('Precision (1-FDR)')
xlim([0,1]),ylim([0,1])
s = sprintf('%1.2f ',thr);
title(['By-Event Performance Using thr = [' s ']'])
legend('Location','eastoutside');
hold on
fprintf('\nModel       Precision        Recall           F1-Score\n')
for model_index = show_models
    model_name = publishedModels{model_index}.name;
    UFN = sum(cellfun(@(c) c.UFN, publishedModels{model_index}.metrics_events));
    UFP = sum(cellfun(@(c) c.UFP, publishedModels{model_index}.metrics_events));
    overlaps = [];
    for ind=1:n_reg
        overlaps = cat(1,overlaps,publishedModels{model_index}.details_events{ind}.overlaps);
    end
    overlaps_nz = overlaps(overlaps~=0);
    precision = zeros(length(thr),1);
    recall = zeros(length(thr),1);
    for i = 1:length(thr)
        idx_tp = overlaps_nz>thr(i);
        TP = sum(idx_tp);
        FP = UFP + sum(~idx_tp);
        FN = UFN + sum(~idx_tp);
        precision(i) = TP/(TP+FP);
        recall(i) = TP/(TP+FN);
    end
    f1_score = 2*(precision.*recall)./(precision + recall);
    scatter(recall,precision,50,'Fill','DisplayName',model_name);
    for i = 1:length(thr)
        text(recall(i)+0.01,precision(i),sprintf('%1.2f',f1_score(i)),'HorizontalAlignment','left')
    end
    plot([recall(1),recall(end)],[precision(1),precision(end)],'k--','HandleVisibility','off')
end
% Show F1_Score contour lines
x = 0:0.01:1;
y = 0:0.01:1;
[X,Y] = meshgrid(x,y);
Z = 2*X.*Y./(X+Y);
contour(X,Y,Z,'ShowText','off','TextList',[0.1,0.9],'Color',0.6*[1,1,1],'LineStyle','--','HandleVisibility','off')
hold off


%% Disagreement indicator
mean_agreement = zeros(n_models,1);
disagreement2 = zeros(n_models,1);
agreement_score = zeros(n_models,1);
fprintf('Model      Mean(a) Mean(d2) A-score\n')
for model_index = 1:n_models
    overlaps = [];
    for ind=1:n_reg
        overlaps = cat(1,overlaps,publishedModels{model_index}.details_events{ind}.overlaps);
    end
    mean_agreement(model_index) = mean(overlaps);
    disagreement2(model_index) = mean( (1-overlaps).^2 );
    agreement_score(model_index) = 1 - disagreement2(model_index);
    fprintf('%s %8.1f %8.1f %8.1f\n',publishedModels{model_index}.name,...
        100*mean_agreement(model_index),...
        100*disagreement2(model_index),...
        100*agreement_score(model_index))
end

%% Histogram of Exper marks duration, all registers
figure
gs_dur = [];
for ind = 1:n_reg
    gs = groundTruth.marks_events{ind};
    gs_dur = cat(2,gs_dur,diff(gs') / set.fs);
end
histogram(gs_dur,20), xlabel('Expert Mark Duration [s]')
ylabel('Count')
title(sprintf('Expert Marks, all registers (min dur %1.2f [s])',min(gs_dur)));

%% Scatter duracion ground truth vs overlap
% Por construccion, "pairings" primero se llena de los ground_truth
%model_index = 1;
figure
for model_index = [1,2,3,4]
gs_dur = [];
gs_ov = [];
for ind = 1:n_reg
    gs = publishedModels{model_index}.details_events{ind}.ground_truth;
    n_gs = length(gs);
    ov = publishedModels{model_index}.details_events{ind}.overlaps;
    gs_dur = cat(2,gs_dur,diff(gs') / set.fs);
    gs_ov = cat(1, gs_ov, ov(1:n_gs));
end
% Only the ones with nonzero overlap
gs_dur = gs_dur(gs_ov~=0);
gs_ov = gs_ov(gs_ov~=0);
subplot(2,2,model_index)
scatter(gs_dur,gs_ov,20,'Fill','LineWidth',1.5,'MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2)
xlabel('Expert Mark Duration [s]')
ylabel('Overlap'),ylim([0,1])
title(sprintf('%s, all registers (%d pairs)',publishedModels{model_index}.name,sum(gs_ov~=0)))
end

%% Histograma de duracion para UFP y para UFN por separado

model_index = 4;

% Find UFP, UFN, and their duration
ufn_dur = []; %1767
ufp_dur = []; %2
for ind = 1:n_reg
    pairing = publishedModels{model_index}.details_events{ind}.pairing;
    n_pairing = length(pairing);
    for i = 1:n_pairing
        if isempty(pairing{i}.gs)
            % We have a UFP
            ufp_dur = cat(1,ufp_dur, ( pairing{i}.det(2)-pairing{i}.det(1) ) / set.fs);
        elseif isempty(pairing{i}.det)
            % We have a UFN
            ufn_dur = cat(1,ufn_dur, ( pairing{i}.gs(2)-pairing{i}.gs(1) ) / set.fs);
        end
    end
end
max_dur = max(max(ufn_dur),max(ufp_dur));
if max_dur<4
    max_dur = 4;
else
    max_dur = ceil(max_dur/0.2)*0.2;
end

figure
% Plot Expert duration marks for comparison
gs_dur = [];
for ind = 1:n_reg
    gs = groundTruth.marks_events{ind};
    gs_dur = cat(2,gs_dur,diff(gs') / set.fs);
end
subplot(3,1,1)
h1 = histogram(gs_dur,'BinEdges',0.2:0.2:max_dur);
ylabel('Count')
title(sprintf('Expert Marks (min dur %1.2f [s])',min(gs_dur)));

subplot(3,1,2)
h2 = histogram(ufn_dur,'BinEdges',0.2:0.2:max_dur);
ylabel('Count')
title(sprintf('%s, UFN (min dur %1.2f [s])',publishedModels{model_index}.name,min(ufn_dur)))

subplot(3,1,2)

total = h1.Values;
missed = h2.Values;
fraction = zeros(length(total),1);
for t = 1:length(total)
    if total(t)>0
        fraction(t) = 100*missed(t)/total(t);
    end
end
histogram('BinEdges',0.2:0.2:max_dur,'BinCounts',fraction);
ylabel('[%] from total'),ylim([0,100])
title(sprintf('%s, UFN (min dur %1.2f [s])',publishedModels{model_index}.name,min(ufn_dur)))

subplot(3,1,3)
histogram(ufp_dur,'BinEdges',0.2:0.2:max_dur);
ylabel('Count')
title(sprintf('%s, UFP (min dur %1.2f [s])',publishedModels{model_index}.name,min(ufp_dur)))

xlabel('Duration[s]')