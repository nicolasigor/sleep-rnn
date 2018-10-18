import pandas as pd
import numpy as np
import pickle

# Load data lite
#pickle_name = "pickle_data/n2fft_dataframe_lite.pkl"
#unpickled_df = pd.read_pickle(pickle_name)
#print('Loaded', unpickled_df.shape)

# Split lite data
#train_set_df = unpickled_df[unpickled_df['ID_REG'] == 0]
#print('Train set size:', train_set_df.shape)
#test_set_df = unpickled_df[unpickled_df['ID_REG'] == 1]
#print('Test set size:', test_set_df.shape)

# Save lite data
#train_lite_name = "pickle_data/n2fft_dataframe_lite_train.pkl"
#test_lite_name = "pickle_data/n2fft_dataframe_lite_test.pkl"
#pd.to_pickle(train_set_df, train_lite_name)
#pd.to_pickle(test_set_df, test_lite_name)
#print("Lite Database saved")

# Load data full
pickle_name = "pickle_data/n2fft_05_dataframe_full.pkl"
unpickled_df = pd.read_pickle(pickle_name)
print('Loaded', unpickled_df.shape)

# Split full data
train_set_df = unpickled_df[unpickled_df['ID_REG'] > 0]
print('Train set size:', train_set_df.shape)
test_set_df = unpickled_df[unpickled_df['ID_REG'] == 0]
print('Test set size:', test_set_df.shape)

# Save lite data
train_full_name = "pickle_data/n2fft_05_dataframe_full_train.pkl"
test_full_name = "pickle_data/n2fft_05_dataframe_full_test.pkl"
pd.to_pickle(train_set_df, train_full_name)
pd.to_pickle(test_set_df, test_full_name)
print("Full Database saved")
