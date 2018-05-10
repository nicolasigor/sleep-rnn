function tabData = getTabData( eegData )
% Format: nSample | record | band_1 | ... | band_n | Stage | Mark(0 or 1)

fprintf('Converting to tabular form...\n');
%% Insert records

n = length(eegData.eegRecordFiltered);
b = length(eegData.eegBands(1,:));
tabData = zeros(n, b + 4);
tabData(:,1) = 1:n;                                 % Insert nSample
tabData(:,2) = eegData.eegRecordFiltered;           % Insert Record
tabData(:,3:(b+2)) = eegData.eegBands;              % Insert Bands

%% Insert stages labels
width = eegData.params.fs*eegData.params.epochDuration;
for k = 1:(length(eegData.regStates)-1)
    interval = ((k-1)*width + 1) : (k*width);
    tabData( interval, b+3 ) = eegData.regStates(k);
end
interval = ((k-1)*width + 1) : n;
tabData( interval, b+3 ) = eegData.regStates(k);

%% Insert marks labels
for k = 1:length(eegData.marks)
    tabData( eegData.marks(k,1):eegData.marks(k,2), b+4 ) = 1;
end

fprintf('Convertion finished\n');