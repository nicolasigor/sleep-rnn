%% A4 detection

detection_a4 = cell(11,1);
precision_a4 = zeros(11,1);
recall_a4 = zeros(11,1);
for ind = 1:11
    % Get clipped allnight data (to remove xtreme values)
    allnight = eegData{ind}.eegRecord;
    n_total = length(allnight);
    outlier_thr = prctile(abs(allnight),99);
    clipped_allnight = allnight;
    clipped_allnight( allnight>outlier_thr ) = outlier_thr;
    clipped_allnight( allnight<-outlier_thr ) = -outlier_thr;
    
    % Get NREM (S2+S3+S4) episodes
    states = eegData{ind}.label.states; % Now  2:N3  3:N2  4:N1  5:R  6:W
    allowed_states = [2,3];
    allowed_epochs = any((states == allowed_states)');
    allowed_epochs_inter = seq2inter(allowed_epochs);
    n_segments = size(allowed_epochs_inter,1);
    only_n2n3 = cell(n_segments,1);
    for i = 1:n_segments
       samples = epoch2timestep(allowed_epochs_inter(i,:),set);
       samples(2) = min(samples(2),n_total);
       only_n2n3{i} = clipped_allnight(samples(1):samples(2)); 
    end

    % Detection of A4
    %fprintf('Starting detection in Register %2.1d...\n',ind)
    %tic
    detection_a4{ind} = warby2014_a4_spindle_detection(only_n2n3,clipped_allnight,set.fs);
    %detection_a4_in_inter = seq2inter(detection_a4{ind});
    %fprintf('Finished detection.\n')
    %toc

    % Ground truth 
    detection_gs_in_inter = eegData{ind}.label.marks;
    detection_gs = zeros(size(clipped_allnight,1),1);
    for i = 1:size(detection_gs_in_inter,1)
        detection_gs( detection_gs_in_inter(i,1):detection_gs_in_inter(i,2) ) = 1;
    end

    % Precision and Recall by-sample

    TP = sum( detection_a4{ind}==1 & detection_gs==1 );
    FP = sum( detection_a4{ind}==1 & detection_gs==0 );
    FN = sum( detection_a4{ind}==0 & detection_gs==1 );

    precision_a4(ind) = 100*TP/(TP+FP);
    recall_a4(ind) = 100*TP/(TP+FN);
    fprintf('A4 By-Sample, Register %2.1d: Precision %1.2f - Recall %1.2f\n',ind,precision_a4(ind),recall_a4(ind))
end

%% A5 detection
detection_a5 = cell(11,1);
precision_a5 = zeros(11,1);
recall_a5 = zeros(11,1);

for ind = 1:11
    % Get clipped allnight data (to remove xtreme values)
    allnight = eegData{ind}.eegRecord;
    n_total = length(allnight);
    outlier_thr = prctile(abs(allnight),99);
    clipped_allnight = allnight;
    clipped_allnight( allnight>outlier_thr ) = outlier_thr;
    clipped_allnight( allnight<-outlier_thr ) = -outlier_thr;
    
    % Get NREM (S2) episodes
    states = eegData{ind}.label.states; % Now  2:N3  3:N2  4:N1  5:R  6:W
    allowed_epochs = (states == 3);
    allowed_epochs_inter = seq2inter(allowed_epochs);
    n_segments = size(allowed_epochs_inter,1);
    only_n2 = cell(n_segments,1);
    for i = 1:n_segments
       samples = epoch2timestep(allowed_epochs_inter(i,:),set);
       samples(2) = min(samples(2),n_total);
       only_n2{i} = clipped_allnight(samples(1):samples(2)); 
    end
    
    % Detection of A5
    %fprintf('Starting detection in Register %2.1d...\n',ind)
    %tic
    detection_a5{ind} = warby2014_a5_spindle_detection(only_n2,clipped_allnight,set.fs);
    detection_a5_in_inter = seq2inter(detection_a5{ind});
    %fprintf('Finished detection.\n')
    %toc

    % Ground truth 
    detection_gs_in_inter = eegData{ind}.label.marks;
    detection_gs = zeros(size(clipped_allnight,1),1);
    for i = 1:size(detection_gs_in_inter,1)
        detection_gs( detection_gs_in_inter(i,1):detection_gs_in_inter(i,2) ) = 1;
    end

    % Precision and Recall by-sample
    
    % Falta restringir el analisis a solo N2

    TP = sum( detection_a5{ind}==1 & detection_gs==1 );
    FP = sum( detection_a5{ind}==1 & detection_gs==0 );
    FN = sum( detection_a5{ind}==0 & detection_gs==1 );

    precision_a5(ind) = 100*TP/(TP+FP);
    recall_a5(ind) = 100*TP/(TP+FN);
    fprintf('A5 By-Sample, Register %2.1d: Precision %1.2f - Recall %1.2f - GS:%4.1d - D:%4.1d\n',...
        ind,precision_a5(ind),recall_a5(ind), size(detection_gs_in_inter,1) ,length(detection_a5_in_inter) )
end


%%
scatter(recall_a4,precision_a4,15,'Fill');
xlabel('Recall'),ylabel('Precision')
xlim([0,100]),ylim([0,100]),
axis square

%%


