function segmentData = getSegments( tabData, eegData , state, downsampling)
% Segmentation of EEG data based on a certain state (e.g. N2)
% Additionally, normalization and, if applicable, downsampling is performed
% Sleep Stages, [1,2]:N3 3:N2  4:N1  5:R  6:W




% Rescatar diferentes segmentos de N2 que existen en el registro
% normalizar estos segmentos a N(0,1).
% extraer la duracion de cada uno, numero de marcas,
% mean duracion SS, std duracion SS, min y max

% p.context = 5;              %?
% p.fRange = [10, 16];        % Frequency range [Hz]
% p.cycle = 3;                % Sleep Stage
% Sleep Stages, [1,2]:N3 3:N2  4:N1  5:R  6:W