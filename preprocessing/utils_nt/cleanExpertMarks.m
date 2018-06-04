function [newMarks, newValid, stats] = cleanExpertMarks(oldMarks, states, Twin, fs, minT, maxT)

%% Preparing for processing
idx = oldMarks(:,5) ~= 0;       % 0 is invalid
marks = oldMarks(:, [1 2]);     % initial and final time-steps
marks = marks( idx, : );        % Remove invalid
valid = oldMarks(idx,5);        % Approval index
% In Approval index, we have:
%   2 is a mark made by an expert. We keep it.
%   1 is an accepted Leo's suggestion made by an expert. We keep it only if doesn't intersect with a "2" mark. 
stats.n_marks_no_val0 = size(marks,1);

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
interval = diff(marks') / fs;
idx_positive = (interval ~= 0);
marks = marks(idx_positive,:);
valid = valid(idx_positive);
stats.n_marks_no_dur0 = sum(idx_positive);

% In case of repetitions, keep only one, priority is "2"
newMarks = marks(1,:);
newValid = valid(1);
for k = 2:size(marks,1)
    if any(newMarks(end,:) ~= marks(k,:))
        % This is not a repetitions
        newMarks = cat(1, newMarks, marks(k,:));
        newValid = cat(1, newValid, valid(k));
    elseif valid(k) == 2 
        % We have a repetition and the new val index is 2
        newMarks(end,:) = marks(k,:);
        newValid(end) = valid(k);         
    end
    % If we have a repetition and the new val index is 1
    % We keep the old one
end
stats.n_marks_no_rep = size(newMarks,1);

%% Remove not approved marks

marks = newMarks;
newMarks = marks(1,:);
newValid = valid(1);
intersection = [];
n_included = 0;

for k = 2:size(marks,1)    
    distance = marks(k,1) - newMarks(end,2);
    if distance <= 0
        % Show duration and val index of both of them
        aux = [newValid(end),diff(newMarks(end,:))/fs,valid(k),diff(marks(k,:))/fs, -distance/fs];
        intersection = cat(1,intersection, aux);
        
        % Count completely included marks
        if diff(marks(k,:)) + distance <= 0
            n_included = n_included + 1;
        end
        % If new one is valid=2, we replace the old one
        if valid(k) == 2
            % Replace old mark
            newMarks(end,:) = marks(k,:);
            newValid(end) = valid(k);
        end
        % If new one is valid=1, we keep the old one
    else
        % If no intersection, add it
        newMarks = cat(1, newMarks, marks(k,:));
        newValid = cat(1, newValid, valid(k));
    end
end

stats.n_marks_valid = size(newMarks,1);
stats.n_marks_valid_intersection = intersection;
stats.n_marks_valid_included = n_included;

%% Combine too close marks

marks = newMarks;
newMarks = marks(1, :);
newValid = valid(1);
minDist = Inf;

for k = 2:size(marks,1)
    distance = (marks(k,1) - newMarks(end,2)) / fs;
    % If too close, add the combination
    if distance <= minT
        newMarks(end,2) = marks(k,2);
        newValid(end) = max(valid(k),newValid(end));
    else
        newMarks = cat(1, newMarks, marks(k,:));
        newValid = cat(1, newValid, valid(k));
    end
    if distance < minDist
        minDist = distance;
    end
end

stats.n_marks_aftercomb = size(newMarks,1);
stats.n_marks_aftercomb_minDist = minDist;

%% Remove too short and too long marks

interval = diff(newMarks') / fs;
too_short = interval < minT;
too_long = interval > maxT;
stats.n_marks_durationcontrol_too_short = sum(too_short);
stats.n_marks_durationcontrol_too_long = sum(too_long);
stats.n_marks_durationcontrol_duration_short = interval(too_short);
stats.n_marks_durationcontrol_duration_long = interval(too_long);
stats.n_marks_durationcontrol_validity_short = newValid(too_short);
stats.n_marks_durationcontrol_validity_long = newValid(too_long);

newMarks = newMarks( ~(too_short | too_long) , : );
newValid = newValid(~(too_short | too_long));
stats.n_marks_durationcontrol = size(newMarks,1);

%% Make them integer (step times)
newMarks = round(newMarks);

%% Remove marks that are outside N2, N3
% Sleep Stages, 2:N3  3:N2  4:N1  5:R  6:W

% allowedStates = [2, 3];
% 
% marks = newMarks;
% newMarks = [];
% valid = newValid;
% newValid = [];

n_n2_only = 0;
n_n3_only = 0;
n_trans = 0;
transition = [];
% params.fs = fs;
% params.epochDuration = Twin;
% for k = 1:size(marks,1)
%     epoch = timestep2epoch(marks(k,:),params);
%     stateStart = states(epoch(1));
%     stateEnd = states(epoch(2));
%     if any( allowedStates == stateStart | allowedStates == stateEnd)
%         % The mark starts or ends inside an allowed state, so we keep it
%         newMarks = cat(1, newMarks, marks(k,:));
%         newValid = cat(1, newValid, valid(k));
%         % Extra useful stats
%         if stateStart==stateEnd
%             if stateStart==3
%                 % The mark is completely inside N2
%                 n_n2_only = n_n2_only + 1;
%             else
%                 % The mark is completely inside N3
%                 n_n3_only = n_n3_only + 1;
%             end
%         else 
%             % The mark has some transition
%             n_trans = n_trans + 1 ;
%             transition = cat(1, transition, [stateStart,stateEnd]);
%         end
%     end
% end
stats.n_marks_statecontrol_n2n3 = size(newMarks,1);
stats.n_marks_statecontrol_n2only = n_n2_only;
stats.n_marks_statecontrol_n3only = n_n3_only;
stats.n_marks_statecontrol_ntrans = n_trans;
stats.n_marks_statecontrol_transition = transition;
