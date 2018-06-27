import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

class SpindleFC(object):
    def __init__(self):


    def train(self, train_path):
        self.train_path = train_path
        train_df = pd.read_pickle(train_path)
        features = np.stack(train_df['FFT_DATA'].values)
        labels = train_df['MARK'].values.astype(int)
        del train_df
        

    def predict(self, test_path):
        pass

    def load_checkpoint(self, checkpoint_path):
        pass

    def _data_init(self):
        pass

    def _model_init(self):
        pass

    def _loss_init(self):
        pass

    def _optimizer_init(self):
        pass


if __name__ == "__main__":
    train_path = "pickle_data/n2fft_dataframe_lite_train.pkl"
    model = SpindleFC()
    model.train(train_path)