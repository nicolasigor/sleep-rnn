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

%% Remove too short and too long marks

interval = diff(newMarks') / fs;
newMarks = newMarks( interval>=minT & interval<= maxT , : );

%% Combine too close marks

marks = newMarks;
n = length(marks);
newMarks = marks(1, :);

for k = 2:n
    dist = marks(k,1) - newMarks(end,2);
    % If too close, add the combination
    if dist <= minT
        newMarks(end,2) = marks(k,2); 
    else
        newMarks = cat(1, newMarks, marks(k,:));
    end
end

%% Remove too long marks

interval = diff(newMarks') / fs;
newMarks = newMarks( interval<= maxT , : );

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
    if any( allowedStates == regStates(pageStart) | allowedStates == regStates(pageEnd))
        newMarks = cat(1, newMarks, marks(k,:));
    end
end

