from __future__ import division
import tensorflow as tf
import numpy as np
import pandas as pd


def main():
    # Load dataset
    train_path = "pickle_data/n2fft_dataframe_full_train.pkl"
    test_path = "pickle_data/n2fft_dataframe_full_test.pkl"

    # Training set
    train_df = pd.read_pickle(train_path)
    train_features = np.stack(train_df['FFT_DATA'].values)
    train_labels = train_df['MARK'].values.astype(int)
    del train_df

    # Create weight for training
    train_weight = np.ones(train_features.shape[0])
    train_weight[train_labels == 1] = 4

    # Test set
    test_df = pd.read_pickle(test_path)
    test_features = np.stack(test_df['FFT_DATA'].values)
    test_labels = test_df['MARK'].values.astype(int)
    del test_df

    # Normalize data
    mean_train = np.mean(train_features, axis=0)
    std_train = np.std(train_features, axis=0)
    train_features = (train_features - mean_train)/std_train
    train_features = (test_features - mean_train)/std_train

    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=train_features.shape[1])]
    weight_column = tf.feature_column.numeric_column("weight")

    # Build 2 layer DNN with 128, 64, 1 units respectively
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[128, 64],
                                            n_classes=2,
                                            model_dir="model_data/spindle_estimator",
                                            optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                                            weight_column=weight_column,
                                            dropout=0.3)

    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_features, "weight": train_weight},
        y=train_labels,
        num_epochs=50,
        shuffle=True,
        batch_size=32
    )

    # Train model
    classifier.train(input_fn=train_input_fn)

    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_features, "weight": np.ones(test_features.shape[0])},
        y=test_labels,
        num_epochs=1,
        shuffle=False)

    # Evaluate
    metrics = classifier.evaluate(input_fn=test_input_fn)
    print(metrics)


if __name__ == "__main__":
    main()
