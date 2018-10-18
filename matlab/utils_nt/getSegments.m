function segments = getSegments( regName, flag_save_segments_label, params)

%% Set parameters 
p.epochDuration = params.epochDuration;
p.fs = params.fs;
p.regName = regName;
fprintf('Getting segments for %s...\n', p.regName);

p.regContainer = 'ssdata/label';
p.regStatesFile = [p.regContainer '/' p.regName '/Sleep States/StagesOnly_' p.regName '.txt'  ];
p.regSpindlesFile = [p.regContainer '/' p.regName '/Sleep Spindles/SS1_' p.regName '_ch1_fixed.txt'  ];

%% Read labels
states = load(p.regStatesFile);
marks = load(p.regSpindlesFile);
marks = marks(:,[1,2]);

%% Obtain epochs of each segment

ind_n2 = states == 3;
intervals = seq2inter(ind_n2);
segments.intervals = intervals;
segments.n_epoch_in_segments = (intervals(:,2)-intervals(:,1)+1);

%% Obtain epochs of marks
marks_epoch = timestep2epoch( marks, params );

segments.marks = cell(size(intervals,1),1);
for i = 1:size(intervals,1)
    beginning_inside = marks_epoch(:,1) >= intervals(i,1);
    ending_inside = marks_epoch(:,2) <= intervals(i,2);
    inside = beginning_inside & ending_inside;
    segments.marks{i} = marks(inside,:);    
end

% Save segments
if flag_save_segments_label
    % Save intervals
    p.regStatesSegmentFile = [p.regContainer '/' p.regName '/Sleep States/SegmentsN2_' p.regName '.txt'  ];
    fid=fopen(p.regStatesSegmentFile,'w');
    % ID EPOCH_START EPOCH_FINISH
    fprintf(fid, '%d %d %d\n', [ (1:size(intervals,1))' , intervals]');
    fclose(fid);
    
    % Save marks
    p.regSpindlesSegmentFile = [p.regContainer '/' p.regName '/Sleep Spindles/SS1_SegmentsN2_' p.regName '.txt'  ];
    fid=fopen(p.regSpindlesSegmentFile,'w');
    for i = 1:size(intervals,1)
        fprintf(fid, '%d %d %d\n',[i*ones(size(segments.marks{i},1),1) , segments.marks{i}]');
    end
    fclose(fid);
end

%% Output params as well
segments.params = p;

fprintf('Segmentation finished\n');