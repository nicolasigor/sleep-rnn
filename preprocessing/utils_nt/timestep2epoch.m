function epoch = timestep2epoch( timestep, params )
epoch = ceil( timestep/(params.epochDuration*params.fs) );