import tensorflow as tf

from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import Model

def build_model(output_dims=3):
    model = tf.keras.Sequential([
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dense(output_dims, activation='linear')
    ])
    return model