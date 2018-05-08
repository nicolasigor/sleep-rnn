function [p, bands] = getBands( eegData )

% get medical frequency bands with Rosario's filter
% accros entire record.

p.alfa = [ 8, 15 ] % ejemplo sin sentido para rangos