from sleep_data_inta import SleepDataINTA
from sleep_data_mass import SleepDataMASS

# Create checkpoint INTA
dataset = SleepDataINTA()
dataset.save_checkpoint()
del dataset

# Create checkpoint MASS
dataset = SleepDataMASS()
dataset.save_checkpoint()
del dataset

# Test if checkpoints are useful
dataset = SleepDataINTA(load_from_checkpoint=True)
del dataset
dataset = SleepDataMASS(load_from_checkpoint=True)
del dataset