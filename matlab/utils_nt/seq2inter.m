function interval = seq2inter( sequence )
% Transform a 0-1 sequence to an interval format
interval = [];
n = length(sequence);
prev_val = 0;
for i = 1:n
    if sequence(i) > prev_val
        %We just turned on
        interval = cat(1, interval, [i, i]);
    elseif sequence(i) < prev_val
        %We just turned off
        interval(end, 2) = i-1;
    end
    prev_val = sequence(i);
end

% Border case (never turned off)
if sequence(n)==1
    interval(end, 2) = n;
end

