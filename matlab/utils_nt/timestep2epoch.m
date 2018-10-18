function epoch = timestep2epoch( timestep, params )
epoch = ceil( timestep/(params.dur_epoch*params.fs) );