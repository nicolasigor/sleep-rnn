%% Debugging too close marks

tmp_marks_ind1_seg1 = eegData{1}.segments.marks{1};
n = length(tmp_marks_ind1_seg1);
distance = zeros(n-1,1);
for k = 2:n
    distance(k-1) = (tmp_marks_ind1_seg1(k,1) - tmp_marks_ind1_seg1(k-1,2))/200;
end
disp(distance(distance<0.3))
%  Distance between adjacent marks that are closer than 0.3s
%     0.2200
%     0.0750
%     0.2750
%     0.1600
