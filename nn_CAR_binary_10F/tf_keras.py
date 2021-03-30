import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#https://shiva-verma.medium.com/understanding-input-and-output-shape-in-lstm-keras-c501ee95c65e
model = keras.Sequential()
# 2: timesteps
# 10: dimensionality
model.add(keras.layers.GRU(units=3, input_shape=(2, 10), return_sequences=True))
model.add(keras.layers.Softmax(axis=0))
model.summary()
