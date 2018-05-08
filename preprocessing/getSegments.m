function segmentData = getSegments( regName, state)

% Rescatar diferentes segmentos de sq2 que existen en el registro
% normalizar estos segmentos a N(0,1).
% extraer la duracion de cada uno, numero de marcas,
% mean duracion SS, std duracion SS, min y max

% p.context = 5;              %?
% p.fRange = [10, 16];        % Frequency range [Hz]
% p.cycle = 3;                % Sleep Stage
% 1:SQ4  2:SQ3  3:SQ2  4:SQ1  5:REM  6:WA 