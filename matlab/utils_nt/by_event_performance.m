function  [metrics, details] = by_event_performance(ground_truth, detection, thr)
% We asume right order of timestamps (start <= end)
% Matching procedure


% Sort marks
gs = sortrows(ground_truth);
det = sortrows(detection);

details.ground_truth = gs;
details.detection = det;

% Matrix of overlap, rows are gs, columns are det
n_gs = size( gs ,1);
n_det = size( det ,1);
details.n_gs = n_gs;
details.n_det = n_det;

overlap_mat = zeros(n_gs, n_det);
% Naive approach, N^2
for i = 1:n_gs
    for j = 1:n_det
        inter_samples = max(gs(i,1),det(j,1)) : min(gs(i,2),det(j,2));
        if ~isempty(inter_samples)
            intersection = length(inter_samples);
            union = length( min(gs(i,1),det(j,1)) : max(gs(i,2),det(j,2)) );
            overlap_mat(i,j) = intersection/union;
        end
    end
end
details.overlap_mat = overlap_mat;

UFP = 0;
UFN = 0;

% Greedy matching
pairing = cell(n_gs,1);
det_paired = zeros(n_det,1);
for i = 1:n_gs
    if any(overlap_mat(i,:))
        % Find maximum overlap
       [Y,I] = max(overlap_mat(i,:));
       % If more than one max, pick the first
       ov = Y(1);
       j = I(1);
       % Make the pair
       pairing{i}.gs = gs(i,:);
       pairing{i}.det = det(j,:);
       pairing{i}.overlap = ov;
       % Now remove this detection from further gs
       overlap_mat(:,j) = 0;
       % Save the index for later on 
       det_paired(j) = 1;
    else
       % Make a fake pair (unpaired false negative)
       UFN = UFN + 1;
       pairing{i}.gs = gs(i,:);
       pairing{i}.det = [];
       pairing{i}.overlap = 0;
    end
end

for j = 1:n_det
    % If this detection was not paired before
    if det_paired(j)==0
       % Make a fake pair  (unpaired false positive)
       UFP = UFP + 1;
       aux = cell(1,1);
       aux{1}.gs = [];
       aux{1}.det = det(j,:);
       aux{1}.overlap = 0;
       pairing = [pairing; aux];
    end
end
details.pairing = pairing;
details.overlaps = cellfun(@(x) x.overlap ,pairing);

% Now we have every pair

% Some obvious metrics
metrics.UFP = UFP;
metrics.UFN = UFN;
metrics.with_overlap = n_gs - UFN;

% Some metrics obtained using a threshold
details.thr = thr;
valid = (details.overlaps>details.thr);
TP = sum(valid);
FP = n_det - TP;
FN = n_gs - TP;

details.FPthr = FP;
details.TPthr = TP;
details.FNthr = FN;

metrics.precision = TP/(TP+FP);
metrics.recall = TP/(TP+FN);
metrics.f1_score = f_beta_score(metrics.precision,metrics.recall,1);

end

function f_beta = f_beta_score(precision,recall,b)
    b2 = b^2;
    f_beta = (1+b2)*(precision*recall)/(b2*precision + recall) ;
end






