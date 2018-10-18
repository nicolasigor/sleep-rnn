% Asuming a binarization of prediction




states = eegData{1}.label.states;

net_params.umbral = 0.5;
net_params.min_pred = (set.dur_min_ss - 0.1);
net_params.max_pred = (set.dur_max_ss + 1);
net_params.n_signal = size(eegData{1}.eegRecord,1);
net_params.n2_val = n2_val;
net_params.fs = set.fs;
net_params.dur_min_ss = set.dur_min_ss;

prediction_events = cleanNeuralPrediction(states, net_params);
