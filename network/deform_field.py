import tensorflow as tf

from tensorflow.keras.layers import Dense, BatchNormalization

def build_deform_field(output_type='PE'):
    assert output_type == 'PE' or output_type == 'coord'

    model = [
        # TODO 3: ablation on deform_field structure
        # Dense(128, activation='relu'),
        # BatchNormalization(),
        # Dense(128, activation='relu'),
        # BatchNormalization(),
        # Dense(128, activation='relu'),
        # BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu')
    ]
    if output_type == 'PE':
        model.append(Dense(40, activation='linear'))
    if output_type == 'coord':
        model.append(Dense(2, activation='tanh'))
    model = tf.keras.Sequential(model)
    return model