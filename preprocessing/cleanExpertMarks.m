function newMarks = cleanExpertMarks(marks, fs, minT, maxT)

%% Remove invalid marks
marks = marks( marks(:,5) ~= 0, : );
valid = marks(:,5);
marks = marks(:, [1 2]);
marks = sortrows(marks, 'ascend');

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