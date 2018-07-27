function prediction_events = cleanNeuralPrediction(states, p)
    spindle_fc = load('prediction.csv'); % prediction for ind=1  
    spindle_fc_bin = spindle_fc > p.umbral;

    % Split epochs
    spindle_fc_epochs = cell(303,1);
    for i = 1:303
        sample_start = (i-1)*800 + 1;
        sample_end = i*800;
        with_context = spindle_fc_bin(sample_start:sample_end);
        without_context = with_context(101:700);
        spindle_fc_epochs{i} = without_context;
    end

    % Corresponding states
    
    epochs_n2 = find((states == p.n2_val));

    % giant vector for prediction:
    pred_samples = zeros(p.n_signal,1);
    for i = 1:303
        epoch_pred = spindle_fc_epochs{i};
        epoch = epochs_n2(i);
        sample_start = (epoch-1)*6000 + 1;
        for j = 1:600
            begin_pred = sample_start + 10*(j-1);
            end_pred = sample_start + 10*j - 1;
            pred_samples(begin_pred:end_pred) = epoch_pred(j);
        end
        if end_pred~=epoch*6000
            fprintf('Something wrong\n')
        end
    end
    fprintf('Ready\n')

    % Now we apply post processing
    predicted_marks = seq2inter(pred_samples);
    fprintf('%d Raw marks\n',size(predicted_marks,1));

    % Combine too close marks
    marks = predicted_marks;
    new_marks = marks(1, :);
    for k = 2:size(marks,1)
        distance = (marks(k,1) - new_marks(end,2)) / p.fs;
        % If too close, combine
        if distance <= p.dur_min_ss
            new_marks(end,2) = marks(k,2);
        else
            % If not too close, add it 
            new_marks = cat(1, new_marks, marks(k,:));
        end
    end
    fprintf('%d marks after combination\n',size(new_marks,1));

    % Remove too short marks
    interval = diff(new_marks') / p.fs;
    idx_too_short = interval <p.min_pred ;
    idx_too_long = interval >p.max_pred ;
    fprintf('%d marks shorter than %1.2f\n',sum(idx_too_short),p.min_pred);
    fprintf('%d marks longer than %1.2f\n',sum(idx_too_long), p.max_pred);
    keep_idx = ~(idx_too_short | idx_too_long);
    new_marks = new_marks( keep_idx , : );
    
    fprintf('%d marks after postprocessing\n',size(new_marks,1));
    prediction_events = new_marks;
end