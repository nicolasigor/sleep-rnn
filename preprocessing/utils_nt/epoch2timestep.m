function timestep = epoch2timestep( epoch, params )
timestep = zeros(1,2);
% Beginning of first epoch
timestep(1) = (epoch(1)-1)*params.epochDuration*params.fs + 1;
% End of last epoch
timestep(2) = epoch(end)*params.epochDuration*params.fs;
