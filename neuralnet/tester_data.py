from sleep_data_inta import SleepDataINTA

# New object
dataset = SleepDataINTA(load_from_checkpoint=True)

# Show something
dummy_feat, dummy_label = dataset.next_batch(64, 256, 1, sub_set="TRAIN")
print(dummy_feat.shape, dummy_label.shape)

# Save checkpoint
# dataset.save_checkpoint()

# Delete object
del dataset

# Load from checkpoint
#dataset2 = SleepDataINTA(load_from_checkpoint=True)

# Show something
#dummy_feat, dummy_label = dataset2.next_batch(64, 256, 1, sub_set="TRAIN")
#print(dummy_feat.shape, dummy_label.shape)
