function  [metrics, details] = by_sample_performance(ground_truth, detection)

% Code  1:FP, 2:TP, 3:FN,  0:TN,
sample_class = zeros(length(ground_truth),1);

FP_condition = detection==1 & ground_truth==0;
sample_class( FP_condition ) = 1;
FP = sum( FP_condition );

TP_condition = detection==1 & ground_truth==1;
sample_class( TP_condition ) = 2;
TP = sum( TP_condition );

FN_condition = detection==0 & ground_truth==1;
sample_class( FN_condition ) = 3;
FN = sum( FN_condition );

details.sample_class = sample_class;
details.FP = FP;
details.TP = TP;
details.FN = FN;

metrics.precision = TP/(TP+FP);
metrics.recall = TP/(TP+FN);
metrics.f1_score = f_beta_score(metrics.precision,metrics.recall,1);
% metrics.f05_score = f_beta_score(metrics.precision,metrics.recall,0.5);
% metrics.f2_score = f_beta_score(metrics.precision,metrics.recall,2);

end

function f_beta = f_beta_score(precision,recall,b)
    b2 = b^2;
    f_beta = (1+b2)*(precision*recall)/(b2*precision + recall) ;
end