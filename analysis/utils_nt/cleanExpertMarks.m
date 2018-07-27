function stats = cleanExpertMarks(marks_filename, marks_filename_new, set)

stats.old_marks_filename = marks_filename;
stats.new_marks_filename = marks_filename_new;

%% Preparing for processing

fprintf('Reading old marks at %s...\n', marks_filename);
marks = load(marks_filename);
stats.n_marks_all = size(marks, 1);

% Only one channel
marks = marks( marks(:,end) == set.channel, : );
stats.n_marks_channel = size(marks,1);

% Remove valid idx 0
valid = marks(:,5);         % Valid idx
keep_idx = (valid ~= 0);
marks = marks(keep_idx, [1 2]); % Initial and final sample-steps
valid = valid(keep_idx);        % Approval index

% In valid idx, we have:
%   2 is a mark made by an expert. We keep it.
%   1 is an accepted Leo's suggestion made by an expert. We keep it only if doesn't intersect with a "2" mark. 
stats.n_marks_no_val_zero = size(marks,1);

%% Format ordering properly

% First column has to be less than second column
for k = 1:size(marks,1)
   if marks(k,1) > marks(k,2)
       aux = marks(k,1);
       marks(k,1) = marks(k,2);
       marks(k,2) = aux;
   end
end

% Increasing order of different marks
marks = sortrows(marks);

% Remove marks with zero duration, to avoid troubles downstream
interval = diff(marks');
idx_positive = (interval ~= 0);
marks = marks(idx_positive,:);
valid = valid(idx_positive);
stats.n_marks_no_dur_zero = sum(idx_positive);

% In case of repetitions, keep only one, priority is "2"
new_marks = marks(1,:);
new_valid = valid(1);
for k = 2:size(marks,1)
    if any(new_marks(end,:) ~= marks(k,:))
        % This is not a repetition
        new_marks = cat(1, new_marks, marks(k,:));
        new_valid = cat(1, new_valid, valid(k));
    elseif valid(k) == 2 
        % We have a repetition and the new valid idx is 2
        new_marks(end,:) = marks(k,:);
        new_valid(end) = valid(k);         
    end
    % If we have a repetition and the new valid idx is 1
    % We keep the old one
end
stats.n_marks_no_rep = size(new_marks,1);

%% Remove not approved marks

marks = new_marks;
new_marks = marks(1,:);
new_valid = valid(1);
intersection = [];
n_included = 0;

for k = 2:size(marks,1)    
    distance = marks(k,1) - new_marks(end,2);
    if distance <= 0
        % Save duration and val index of both of them, and overlap
        aux = [new_valid(end), diff(new_marks(end,:))/set.fs, valid(k), diff(marks(k,:))/set.fs, -distance/set.fs];
        intersection = cat(1,intersection, aux);
        
        % Count completely included marks
        if diff(marks(k,:)) + distance <= 0
            n_included = n_included + 1;
        end
        
        % If new one is valid=2, we replace the old one
        if valid(k) == 2
            % Replace old mark
            new_marks(end,:) = marks(k,:);
            new_valid(end) = valid(k);
        end
        % If new one is valid=1, we keep the old one
    else
        % If no intersection, add it
        new_marks = cat(1, new_marks, marks(k,:));
        new_valid = cat(1, new_valid, valid(k));
    end
end

stats.n_marks_valid = size(new_marks,1);
stats.n_marks_valid_intersection = intersection;
stats.n_marks_valid_included = n_included;

%% Combine too close marks

marks = new_marks;
new_marks = marks(1, :);
new_valid = valid(1);
min_dist_found = Inf;

for k = 2:size(marks,1)
    distance = (marks(k,1) - new_marks(end,2)) / set.fs;
    % If too close, add the combination
    if distance <= set.dur_min_ss
        new_marks(end,2) = marks(k,2);
        new_valid(end) = max(valid(k),new_valid(end));
    else
        % If not too close, add it 
        new_marks = cat(1, new_marks, marks(k,:));
        new_valid = cat(1, new_valid, valid(k));
    end
    if distance < min_dist_found
        min_dist_found = distance;
    end
end

stats.n_marks_distctrl = size(new_marks,1);
stats.n_marks_distctrl_min_dist_found = min_dist_found;

%% Remove too short and too long marks

interval = diff(new_marks') / set.fs;
idx_too_short = interval < set.dur_min_ss;
idx_too_long = interval > set.dur_max_ss;

stats.n_marks_durctrl_too_short = sum(idx_too_short);
stats.n_marks_durctrl_too_long = sum(idx_too_long);
stats.n_marks_durctrl_duration_short = interval(idx_too_short);
stats.n_marks_durctrl_duration_long = interval(idx_too_long);
stats.n_marks_durctrl_validity_short = new_valid(idx_too_short);
stats.n_marks_durctrl_validity_long = new_valid(idx_too_long);

keep_idx = ~(idx_too_short | idx_too_long);
new_marks = new_marks( keep_idx , : );
new_valid = new_valid( keep_idx );

stats.n_marks_durctrl = size(new_marks,1);

%% Make them integer

new_marks = round(new_marks);

%% Save new marks, in the same format as the original

n = size(new_marks,1);
array_to_save = [new_marks, -50*ones(n,1), -50*ones(n,1), new_valid, set.channel*ones(n,1)];
array_to_save = array_to_save';

fid=fopen(marks_filename_new,'w');
fprintf(fid, '%d %d %d %d %d %d\n', array_to_save);
fclose(fid);
fprintf('New marks saved at %s\n', marks_filename_new);
