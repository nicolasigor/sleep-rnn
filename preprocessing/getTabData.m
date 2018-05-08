function tabData = getTabData( eegData )
% Format: nSample | record | band_1 | ... | band_n | Stage | Mark(0 or 1)


%% Insert records

n = length(eegData.eegRecordFiltered);
b = length(eegData.eegBands(1,:));
tabData = zeros(n, b + 4);
tabData(:,1) = 1:n;                                 % Insert nSample
tabData(:,2) = eegData.eegRecordFiltered;           % Insert Record
tabData(:,3:(b+2)) = eegData.eegBands;              % Insert Bands

%% Insert stages labels
Twin = eegData.params.pageDuration;
for k = 1:length(eegData.regStates)
    tabData( ((k-1)*Twin+1) : k*Twin, b+3 ) = eegData.regStates(k);
end

%% Insert marks labels
for k = 1:length(eegData.marks)
    tabData( eegData.marks(k,1):eegData.marks(k,2), b+4 ) = 1;
end

