import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# https://keras.io/api/layers/recurrent_layers/gru/

inputs = tf.random.normal([1, 2, 4])
print(inputs)
print(inputs.shape)
gru_i = tf.keras.layers.GRU(4)
print(tf.__version__)
#output = gru_i(inputs)
#print(output.shape)
#gru = tf.keras.layers.GRU(4, return_sequences=True, return_state=True)
#whole_sequence_output, final_state = gru(inputs)
#print(whole_sequence_output.shape)
#print(final_state.shape)
