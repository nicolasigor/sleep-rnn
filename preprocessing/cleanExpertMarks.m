function newMarks = cleanExpertMarks(regMarks, regStates, Twin, fs, minT, maxT)

%% Remove not approved marks
idx = regMarks(:,5) ~= 0;
marks = regMarks(:, [1 2]);
marks = marks( idx, : );
valid = regMarks(idx,5);
marks = sortrows(marks);

n = length(marks);
newMarks = marks(1,:);

for k = 2:n
    inter = marks(k,1) < newMarks(end,2);
    % If intersection, only valid=2 survives
    if inter
        if valid(k) == 2
            newMarks(end,:) = marks(k,:); 
        end
    else
        newMarks = cat(1, newMarks, marks(k,:));
    end
end

fprintf('Total validated marks %d\n',length(newMarks));

%% Combine too close marks

marks = newMarks;
n = length(marks);
newMarks = marks(1, :);
minDist = 1e30;

for k = 2:n
    dist = marks(k,1) - newMarks(end,2);
    % If too close, add the combination
    if dist <= minT
        newMarks(end,2) = marks(k,2); 
    else
        newMarks = cat(1, newMarks, marks(k,:));
    end
    if dist < minDist
        minDist = dist;
    end
end

fprintf('Total validated marks after combination %d\n',length(newMarks));
fprintf('Min distance found %1.2f\n',minDist);
%% Remove too short and too long marks

interval = diff(newMarks') / fs;
indx_min = interval>=minT;
indx_max = interval<= maxT;
fprintf('Less than minT %d. Greater than maxT %d\n',sum(~indx_min),sum(~indx_max));
fprintf('Duration of left out marks:\n');
disp(interval(~indx_min));
disp(interval(~indx_max));

newMarks = newMarks( interval>=minT & interval<= maxT , : );
fprintf('Total validated marks after combination and duration-restriction %d\n',length(newMarks));
%% Make them integer (step times)

newMarks = round(newMarks);

%% Remove marks that are outside N2, N3
% Sleep Stages, [1,2]:N3 3:N2  4:N1  5:R  6:W

allowedStates = [1, 2, 3];

marks = newMarks;
n = length(marks);
newMarks = [];

for k = 1:n
    pageStart = floor(marks(k,1)/(Twin*fs)) + 1;
    pageEnd = floor(marks(k,2)/(Twin*fs)) + 1;
    cond1 = allowedStates == regStates(pageStart);
    cond2 = allowedStates == regStates(pageEnd);
    if any( cond1 | cond2)
        newMarks = cat(1, newMarks, marks(k,:));
    end
end
fprintf('Total validated marks after N2/N3 restriction %d\n',length(newMarks));

%% Get some extra statistics

n2_only = 0;
n3_only = 0;
trans = 0;
transition = [];

% Sleep Stages, [1,2]:N3 3:N2  4:N1  5:R  6:W
n = length(newMarks);
for k = 1:n
    pageStart = floor(marks(k,1)/(Twin*fs)) + 1;
    pageEnd = floor(marks(k,2)/(Twin*fs)) + 1;
    
    if regStates(pageStart)==regStates(pageEnd) || regStates(pageStart)==2 && regStates(pageEnd)==1 || regStates(pageStart)==1 && regStates(pageEnd)==2
        if regStates(pageStart)==3
            n2_only = n2_only + 1;
        else
            n3_only = n3_only + 1;
        end
    else 
        trans = trans + 1 ;
        transition = cat(1, transition, [regStates(pageStart),regStates(pageEnd)]);
    end
end

fprintf('Total N2-only: %d -- Total N3-only: %d -- Total trans: %d\n',n2_only,n3_only,trans);
fprintf('Transitions: ([1,2]:N3 3:N2  4:N1  5:R  6:W)\n');
disp(transition);
