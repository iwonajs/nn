import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Dropout, Bidirectional, Flatten, Activation, Multiply
from tensorflow.keras.layers import TimeDistributed, RepeatVector, Permute, Lambda, Concatenate, Reshape, Dot, Add, GlobalAveragePooling1D, MaxPooling1D
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPool2D, MaxPool1D, GlobalMaxPool2D, Attention, GlobalMaxPooling1D, LSTM
from tensorflow.keras.regularizers import l2
import json
import os
import sys

import tensorflow.keras.backend as K
#from tensorflow.keras.layers import K
#from tensorflow.keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
#import spp

# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
# https://stackoverflow.com/questions/50419674/keras-embedding-layer-for-multidimensional-time-steps
# https://stackoverflow.com/questions/50419674/keras-embedding-layer-for-multidimensional-time-steps
# https://datascience.stackexchange.com/questions/22177/how-to-use-embedding-with-3d-tensor-in-keras
# https://github.com/keras-team/keras/issues/8195 RESHAPE
# Attention using state: https://androidkt.com/text-classification-using-attention-mechanism-in-keras/
# https://androidkt.com/text-classification-using-attention-mechanism-in-keras/
# Paper Replications by Tobias Lee: https://github.com/TobiasLee/Text-Classification
# Keras Forum: https://github.com/keras-team/keras/issues/4962
# https://stackoverflow.com/questions/44960558/concatenating-embedded-inputs-to-feed-recurrent-lstm-in-keras
# https://stackoverflow.com/questions/44960558/concatenating-embedded-inputs-to-feed-recurrent-lstm-in-keras
# https://github.com/keras-team/keras/issues/2507
# https://stackoverflow.com/questions/52115527/keras-lstm-with-embeddings-of-2-words-at-each-time-step
# https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
# code_input = Input(shape=(hp_code_embedding_dim, hp_code_embedding_dim), name="CODE_2_DIM") # > OK!
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Lambda
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer
# https://stackoverflow.com/questions/49925374/how-lstm-deal-with-variable-length-sequence
# Cosine Sim: https://stackoverflow.com/questions/51003027/computing-cosine-similarity-between-two-tensors-in-keras
# Cosine Sim: https://datascience.stackexchange.com/questions/26784/keras-computing-cosine-similarity-matrix-of-two-3d-tensors
# Control Flow: https://stackoverflow.com/questions/53167108/how-to-conditionally-scale-values-in-keras-lambda-layer
# https://keras.io/layers/merge/
# https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html


def printout(note):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(os.path.realpath(__file__))
    print(*note)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

def tp_rate(y_true, y_pred):
    # https://datascience.stackexchange.com/questions/33587/keras-custom-loss-function-as-true-negatives-by-true-negatives-plus-false-posit

    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    tp = K.sum(y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    fp = K.sum(neg_y_true * y_pred)
    fn = K.sum(y_true * neg_y_pred)
    return tp / (tp + tn + fp + fn)

def tn_rate(y_true, y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    tp = K.sum(y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    fp = K.sum(neg_y_true * y_pred)
    fn = K.sum(y_true * neg_y_pred)
    return tn / (tp + tn + fp + fn)

def fp_rate(y_true, y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    tp = K.sum(y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    fp = K.sum(neg_y_true * y_pred)
    fn = K.sum(y_true * neg_y_pred)
    return fp / (tp + tn + fp + fn)

def fn_rate(y_true, y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    tp = K.sum(y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    fp = K.sum(neg_y_true * y_pred)
    fn = K.sum(y_true * neg_y_pred)
    return fn / (tp + tn + fp + fn)

def gru_attention_last_step(gru, gru_units):
    print("last_step: GRU sequence", gru.shape)

    dense = Dense(gru_units * 2, activation='tanh')(gru)
    print("last_step: GRU Dense", dense.shape)

    softmax = Activation('softmax')(dense)
    print("last_step: GRU Activation", softmax.shape)

    attention = RepeatVector(1)(softmax)
    print("last_step: GRU Repeat", attention.shape)

    weighted = Multiply()([attention, gru])
    print("last_step: GRU Weighted", weighted.shape)

    weighted = Lambda(collapse_steps, output_shape=(gru_units * 2,))(weighted)
    print("last_step: GRU Lambda-Collapse-Dim-2", weighted.shape)
    print("##############################################")

    return weighted

def img_attention(img, img_units):
    print("last_step: GRU sequence", img.shape)

    dense = Dense(img_units, activation='tanh')(img)
    print("last_step: GRU Dense", dense.shape)

    softmax = Activation('softmax')(dense)
    print("last_step: GRU Activation", softmax.shape)

    attention = RepeatVector(1)(softmax)
    print("last_step: GRU Repeat", attention.shape)

    weighted = Multiply()([attention, img])
    print("last_step: GRU Weighted", weighted.shape)

    weighted = Lambda(collapse_steps, output_shape=(img_units,))(weighted)
    print("last_step: GRU Lambda-Collapse-Dim-2", weighted.shape)
    print("##############################################")

    return weighted

def combined_attention(combined):
    print("------------------------------------")
    print("Combined input", combined.shape)

    dense = Dense(hp_code_gru_latent_dim * 2 * 2, activation='tanh')(combined)
    print("Combined, Dense", dense.shape)

    dense = RepeatVector(1)(dense)
    print("Combined Repeat", dense.shape)

    attention = Activation('softmax')(dense)
    print("Combined, Activation", attention.shape)

    attention = Reshape([1, 1, hp_code_gru_latent_dim * 2 * 2])(attention)
    print("Combined, Reshape", attention.shape)

    weighted = Multiply()([combined, attention])
    print("Combined, Weighted", weighted.shape)

    weighted = Lambda(collapse_steps, output_shape=(hp_code_gru_latent_dim * 2 * 2,))(weighted)
    print("Combined, Lambda-Collapse-Dim-2", weighted.shape)

    print("--------------------------------------------------------")
    return weighted

def collapse_steps(x):
    """
    Collapses the STEPS dimension, variable length
    axis = -2
    Source: https://github.com/keras-team/keras/issues/4962
    Source: https://androidkt.com/text-classification-using-attention-mechanism-in-keras/

    :param x: (B, S, U*?2)
    :return: (B, U*?2)
    """
    return K.sum(x, axis=-2)








def img_features(img):
    print("IMG: *******************************************************")
    # CONV2 OUT: (B=1, W*, H*, F=64)
    print("img:", img.shape)
    img = Permute((3, 1, 2))(img)
    print("img permute:", img.shape)
    # imgx = Reshape((tf.shape(img)[1]*tf.shape(img)[2], tf.shape(img)[3]))(img)
    img = TimeDistributed(Flatten())(img)
    print("FLATTEN FILTER: ", img.shape)
    # Global Pooling: (B=1, 1, F=64)
    img = Permute((2, 1))(img)
    img = GlobalAveragePooling1D()(img)
    print("GLOBAL POOLING:", img.shape)
    img = Reshape((1, 64))(img)
    print("FINAL IMG 1F, 64E:", img.shape)
    return img

def attention_distribution(query, key):
    scores = tf.matmul(query, key, transpose_b=True)
    #distribution = tf.nn.softmax(scores)
    #return distribution
    return scores


class Feed_Type:
    def __init__(self):
        self.feed_CAR = "CAR"

class CAR_variant_00:

    def __init__(self, configs):

        self.configs = configs
        #self.config_file = configs
        #if not os.path.exists(configs):
        #    print("Error, Experiment config file: " + configs + " does not exist.")
        #    sys.exit(1)
        #with open(self.config_file) as f:
        #    config = json.loads(f.read())

        choose = Feed_Type()
        self.feed_flag = choose.feed_CAR

        # KRA SEQ OBJECTS
        gru_units = self.configs["gru_units"]
        gru_units = 32
        kra_objects_input = Input(shape=(None, ), name="KRA_SEQ_OBJECTS_INPUT")
        kra_objects = Embedding(self.configs["kra_objects_features"],
                            self.configs["kra_objects_embedding"],
                            input_length=None)(kra_objects_input)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=True), name="KRA_SEQ_OBJECTS_GRU_1"
        )(kra_objects)
        kra_objects = Dropout(rate=0.2)(kra_objects)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=True), name="KRA_SEQ_OBJECTS_GRU_2"
        )(kra_objects)
        kra_objects = Dropout(rate=0.2)(kra_objects)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=False), name="KRA_SEQ_OBJECTS_GRU_3"
        )(kra_objects)


        # KRA SEQ LEVELS
        kra_levels_input = Input(shape=(None, ), name="KRA_SEQ_LEVELS_INPUT")
        kra_levels = Embedding(self.configs["kra_levels_features"],
                            self.configs["kra_levels_embedding"],
                            input_length=None)(kra_levels_input)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=True), name="KRA_SEQ_LEVELS_GRU_1"
        )(kra_levels)
        kra_levels = Dropout(rate=0.2)(kra_levels)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=True), name="KRA_SEQ_LEVELS_GRU_2"
        )(kra_levels)
        kra_levels = Dropout(rate=0.2)(kra_levels)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=False), name="KRA_SEQ_LEVELS_GRU_3"
        )(kra_levels)

        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(self.configs['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(self.configs["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=True, name='BUG_GRU_1')
            )(bug_input)
        bug = Dropout(rate=0.20)(bug)
        bug = Bidirectional(
            GRU(self.hp_bug_gru_latent_dim, return_sequences=True, name='BUG_GRU_2')
        )(bug)
        bug = Dropout(rate=0.20)(bug)
        bug = Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, name='BUG_GRU_3')
            )(bug)

        # Combined _____________________________________________________________________________________________________________
        bug = Reshape((1, 64), name="xxx1")(bug)
        kra_objects = Reshape((1, 64), name="xxx2")(kra_objects)
        kra_levels = Reshape((1, 64), name="xxx3")(kra_levels)

        print("NORM BEFORE: *******************************************************")
        kra_objects = Activation('softmax')(kra_objects)
        kra_levels = Activation('softmax')(kra_levels)

        print("KRA OBJECTS *******************************************************")
        attKraObjects = attention_distribution(bug, kra_objects)
        sim_bug_kra_objects = Multiply()([attKraObjects, kra_objects])

        print("KRA LEVELS *******************************************************")
        attKraLevels = attention_distribution(bug, kra_levels)
        sim_bug_kra_levels = Multiply()([attKraLevels, kra_levels])

        combined = Concatenate(axis=-2)([bug, sim_bug_kra_objects, sim_bug_kra_levels])  #, sim_img_combined])  #, sim_combined3])  #
        combined = Flatten()(combined)

        # >>>>>>>>>>>> combined = Dropout(rate=.5)(combined)
        print("All Combined:", combined.shape)
        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)  ##############, W_regularizer=l2(0.01))(combined)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.configs["loss"], metrics=['accuracy'])  #############3, tp_rate, tn_rate])
        printout(["Done Building Model", self.__class__.__name__])

    #def gen_mode(self):
    #    return self.feed_type

    def save_model_summary(self, path):
        '''
        https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
        https://dzone.com/articles/python-101-redirecting-stdout
        :param path:
        :return:
        '''


        original = sys.stdout
        sys.stdout = open(path+"model_summary.txt", 'w')
        self.model.summary()
        sys.stdout = original

# SOURCE CODE:
# Author: ?
# URL: https://androidkt.com/text-classification-using-attention-mechanism-in-keras/
class Attention_Custom(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Attention_Custom, self).__init__()
        self.units = units
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        #score = tf.nn.tanh(self.W1(features) + self.W2(hidden))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

    # https://www.tensorflow.org/guide/keras/save_and_serialize/
    def get_config(self):
        return {"units": self.units}

class CAR_variant_01:

    def __init__(self, configs):

        self.configs = configs
        choose = Feed_Type()
        self.feed_flag = choose.feed_CAR

        # KRA SEQ OBJECTS
        gru_units = self.configs["gru_units"]
        gru_units = 32
        kra_objects_input = Input(shape=(None, ), name="KRA_SEQ_OBJECTS_INPUT")
        #kra_objects_input = tf.keras.preprocessing.sequence.pad_sequences(kra_objects_input, padding='post')
        kra_objects = Embedding(self.configs["kra_objects_features"],
                            self.configs["kra_objects_embedding"],
                            input_length=None)(kra_objects_input)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.3), name="KRA_SEQ_OBJECTS_GRU_1"
        )(kra_objects)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.2), name="KRA_SEQ_OBJECTS_GRU_2"
        )(kra_objects)
        value_seq_CAR_O = Bidirectional(
            GRU(units=gru_units, return_sequences=True), name="KRA_SEQ_OBJECTS_GRU_3"
        )(kra_objects)


        # KRA SEQ LEVELS
        kra_levels_input = Input(shape=(None, ), name="KRA_SEQ_LEVELS_INPUT")
        kra_levels = Embedding(self.configs["kra_levels_features"],
                            self.configs["kra_levels_embedding"],
                            input_length=None)(kra_levels_input)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.3), name="KRA_SEQ_LEVELS_GRU_1"
        )(kra_levels)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.2), name="KRA_SEQ_LEVELS_GRU_2"
        )(kra_levels)
        value_seq_CAR_L = Bidirectional(
            GRU(units=gru_units, return_sequences=True), name="KRA_SEQ_LEVELS_GRU_3"
        )(kra_levels)

        # Bug Branch ___________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(self.configs['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(self.configs["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=True, name='BUG_GRU_1')
            )(bug_input)
        bug = Dropout(rate=0.20)(bug)
        bug = Bidirectional(
            GRU(self.hp_bug_gru_latent_dim, return_sequences=True, name='BUG_GRU_2')
        )(bug)
        bug = Dropout(rate=0.20)(bug)
        query_seq_Bug = Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=True, name='BUG_GRU_3')
            )(bug)

        value_seq_CAR = tf.keras.layers.Concatenate(axis=1)([value_seq_CAR_L, value_seq_CAR_O])
        # Attention ____________________________________________________________________________________________________
        query_value_attention_seq = tf.keras.layers.Attention()([query_seq_Bug, value_seq_CAR])
        query_encoding = tf.keras.layers.GlobalAveragePooling1D()(query_seq_Bug)
        query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(query_value_attention_seq)
        modulated = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])

        # DNN ____________________________________________________________________________________________________
        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(modulated)

        # Model: _______________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.configs["loss"],
                           metrics=['accuracy'])  ####, tp_rate, tn_rate, fp_rate, fn_rate])
        printout(["Done Building Model", self.__class__.__name__])

    def save_model_summary(self, path):
        '''
        https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
        https://dzone.com/articles/python-101-redirecting-stdout
        :param path:
        :return:
        '''
        original = sys.stdout
        sys.stdout = open(path+"model_summary.txt", 'w')
        self.model.summary()
        sys.stdout = original

class CAR_variant_04:

    def __init__(self, configs):

        self.configs = configs
        choose = Feed_Type()
        self.feed_flag = choose.feed_CAR

        # KRA SEQ OBJECTS
        gru_units = self.configs["gru_units"]
        gru_units = 32
        kra_objects_input = Input(shape=(None, ), name="KRA_SEQ_OBJECTS_INPUT")
        #kra_objects_input = tf.keras.preprocessing.sequence.pad_sequences(kra_objects_input, padding='post')
        kra_objects = Embedding(self.configs["kra_objects_features"],
                            self.configs["kra_objects_embedding"],
                            input_length=None)(kra_objects_input)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.3), name="KRA_SEQ_OBJECTS_GRU_1"
        )(kra_objects)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.2), name="KRA_SEQ_OBJECTS_GRU_2"
        )(kra_objects)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=False), name="KRA_SEQ_OBJECTS_GRU_3"
        )(kra_objects)


        # KRA SEQ LEVELS
        kra_levels_input = Input(shape=(None, ), name="KRA_SEQ_LEVELS_INPUT")
        kra_levels = Embedding(self.configs["kra_levels_features"],
                            self.configs["kra_levels_embedding"],
                            input_length=None)(kra_levels_input)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.3), name="KRA_SEQ_LEVELS_GRU_1"
        )(kra_levels)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.2), name="KRA_SEQ_LEVELS_GRU_2"
        )(kra_levels)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=False), name="KRA_SEQ_LEVELS_GRU_3"
        )(kra_levels)

        # Bug Branch ___________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(self.configs['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(self.configs["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=True, name='BUG_GRU_1')
            )(bug_input)
        bug = Dropout(rate=0.20)(bug)
        bug = Bidirectional(
            GRU(self.hp_bug_gru_latent_dim, return_sequences=True, name='BUG_GRU_2')
        )(bug)
        bug = Dropout(rate=0.20)(bug)
        bug = Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, name='BUG_GRU_3')
            )(bug)

        # Attention ____________________________________________________________________________________________________
        #state_h = Concatenate()([kra_objects, kra_levels])
        #a = Attention_Custom(32)
        modulated = tf.keras.layers.Attention()([bug, kra_levels])
        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(modulated)

        # Model: _______________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.configs["loss"], metrics=['accuracy'])  ####, tp_rate, tn_rate, fp_rate, fn_rate])
        printout(["Done Building Model", self.__class__.__name__])

    def save_model_summary(self, path):
        '''
        https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
        https://dzone.com/articles/python-101-redirecting-stdout
        :param path:
        :return:
        '''
        original = sys.stdout
        sys.stdout = open(path+"model_summary.txt", 'w')
        self.model.summary()
        sys.stdout = original

class CAR_variant_05:

    def __init__(self, configs):

        self.configs = configs
        choose = Feed_Type()
        self.feed_flag = choose.feed_CAR

        # KRA SEQ OBJECTS
        gru_units = self.configs["gru_units"]
        gru_units = 32
        kra_objects_input = Input(shape=(None, ), name="KRA_SEQ_OBJECTS_INPUT")
        #kra_objects_input = tf.keras.preprocessing.sequence.pad_sequences(kra_objects_input, padding='post')
        kra_objects = Embedding(self.configs["kra_objects_features"],
                            self.configs["kra_objects_embedding"],
                            input_length=None)(kra_objects_input)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.3), name="KRA_SEQ_OBJECTS_GRU_1"
        )(kra_objects)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.2), name="KRA_SEQ_OBJECTS_GRU_2"
        )(kra_objects)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=False), name="KRA_SEQ_OBJECTS_GRU_3"
        )(kra_objects)


        # KRA SEQ LEVELS
        kra_levels_input = Input(shape=(None, ), name="KRA_SEQ_LEVELS_INPUT")
        kra_levels = Embedding(self.configs["kra_levels_features"],
                            self.configs["kra_levels_embedding"],
                            input_length=None)(kra_levels_input)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.3), name="KRA_SEQ_LEVELS_GRU_1"
        )(kra_levels)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.2), name="KRA_SEQ_LEVELS_GRU_2"
        )(kra_levels)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=False), name="KRA_SEQ_LEVELS_GRU_3"
        )(kra_levels)

        # Bug Branch ___________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(self.configs['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(self.configs["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=True, name='BUG_GRU_1')
            )(bug_input)
        bug = Dropout(rate=0.20)(bug)
        bug = Bidirectional(
            GRU(self.hp_bug_gru_latent_dim, return_sequences=True, name='BUG_GRU_2')
        )(bug)
        bug = Dropout(rate=0.20)(bug)
        bug = Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, name='BUG_GRU_3')
            )(bug)

        # Attention ____________________________________________________________________________________________________
        #state_h = Concatenate()([kra_objects, kra_levels])
        #modulated = tf.keras.layers.Attention()([bug, state_h])
        modulated_O = tf.keras.layers.Attention()([bug, kra_objects])
        modulated_L = tf.keras.layers.Attention()([bug, kra_levels])
        modulated = Concatenate()([modulated_O, modulated_L])
        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(modulated)

        # Model: _______________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.configs["loss"], metrics=['accuracy'])  ####, tp_rate, tn_rate, fp_rate, fn_rate])
        printout(["Done Building Model", self.__class__.__name__])

    def save_model_summary(self, path):
        '''
        https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
        https://dzone.com/articles/python-101-redirecting-stdout
        :param path:
        :return:
        '''
        original = sys.stdout
        sys.stdout = open(path+"model_summary.txt", 'w')
        self.model.summary()
        sys.stdout = original

class CAR_variant_06:

    def __init__(self, configs):

        self.configs = configs
        choose = Feed_Type()
        self.feed_flag = choose.feed_CAR

        # KRA SEQ OBJECTS
        gru_units = self.configs["gru_units"]
        gru_units = 32
        kra_objects_input = Input(shape=(None, ), name="KRA_SEQ_OBJECTS_INPUT")
        #kra_objects_input = tf.keras.preprocessing.sequence.pad_sequences(kra_objects_input, padding='post')
        kra_objects = Embedding(self.configs["kra_objects_features"],
                            self.configs["kra_objects_embedding"],
                            input_length=None)(kra_objects_input)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.3), name="KRA_SEQ_OBJECTS_GRU_1"
        )(kra_objects)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.2), name="KRA_SEQ_OBJECTS_GRU_2"
        )(kra_objects)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=False), name="KRA_SEQ_OBJECTS_GRU_3"
        )(kra_objects)


        # KRA SEQ LEVELS
        kra_levels_input = Input(shape=(None, ), name="KRA_SEQ_LEVELS_INPUT")
        kra_levels = Embedding(self.configs["kra_levels_features"],
                            self.configs["kra_levels_embedding"],
                            input_length=None)(kra_levels_input)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.3), name="KRA_SEQ_LEVELS_GRU_1"
        )(kra_levels)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.2), name="KRA_SEQ_LEVELS_GRU_2"
        )(kra_levels)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=False), name="KRA_SEQ_LEVELS_GRU_3"
        )(kra_levels)

        # Bug Branch ___________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(self.configs['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(self.configs["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=True, name='BUG_GRU_1')
            )(bug_input)
        bug = Dropout(rate=0.20)(bug)
        bug = Bidirectional(
            GRU(self.hp_bug_gru_latent_dim, return_sequences=True, name='BUG_GRU_2')
        )(bug)
        bug = Dropout(rate=0.20)(bug)
        bug = Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, name='BUG_GRU_3')
            )(bug)

        # Attention ____________________________________________________________________________________________________
        #state_h = Concatenate()([kra_objects, kra_levels])
        #modulated = tf.keras.layers.Attention()([bug, state_h])
        modulated_O = tf.keras.layers.Attention()([bug, kra_objects])
        modulated_L = tf.keras.layers.Attention()([bug, kra_levels])
        modulated = Concatenate()([modulated_O, modulated_L])
        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(modulated)

        # Model: _______________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.configs["loss"], metrics=['accuracy'])  ####, tp_rate, tn_rate, fp_rate, fn_rate])
        printout(["Done Building Model", self.__class__.__name__])

    def save_model_summary(self, path):
        '''
        https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
        https://dzone.com/articles/python-101-redirecting-stdout
        :param path:
        :return:
        '''
        original = sys.stdout
        sys.stdout = open(path+"model_summary.txt", 'w')
        self.model.summary()
        sys.stdout = original

class CAR_variant_09:

    def __init__(self, configs):

        self.configs = configs
        choose = Feed_Type()
        self.feed_flag = choose.feed_CAR

        # KRA SEQ OBJECTS
        gru_units = self.configs["gru_units"]
        gru_units = 32
        kra_objects_input = Input(shape=(None, ), name="KRA_SEQ_OBJECTS_INPUT")
        #kra_objects_input = tf.keras.preprocessing.sequence.pad_sequences(kra_objects_input, padding='post')
        kra_objects = Embedding(self.configs["kra_objects_features"],
                            self.configs["kra_objects_embedding"],
                            input_length=None)(kra_objects_input)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.3), name="KRA_SEQ_OBJECTS_GRU_1"
        )(kra_objects)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.2), name="KRA_SEQ_OBJECTS_GRU_2"
        )(kra_objects)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=False), name="KRA_SEQ_OBJECTS_GRU_3"
        )(kra_objects)


        # KRA SEQ LEVELS
        kra_levels_input = Input(shape=(None, ), name="KRA_SEQ_LEVELS_INPUT")
        kra_levels = Embedding(self.configs["kra_levels_features"],
                            self.configs["kra_levels_embedding"],
                            input_length=None)(kra_levels_input)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.3), name="KRA_SEQ_LEVELS_GRU_1"
        )(kra_levels)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.2), name="KRA_SEQ_LEVELS_GRU_2"
        )(kra_levels)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=False), name="KRA_SEQ_LEVELS_GRU_3"
        )(kra_levels)

        # Bug Branch ___________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(self.configs['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(self.configs["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=True, name='BUG_GRU_1')
            )(bug_input)
        bug = Dropout(rate=0.20)(bug)
        bug = Bidirectional(
            GRU(self.hp_bug_gru_latent_dim, return_sequences=True, name='BUG_GRU_2')
        )(bug)
        bug = Dropout(rate=0.20)(bug)
        bug = Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, name='BUG_GRU_3')
            )(bug)

        # Attention ____________________________________________________________________________________________________
        #state_h = Concatenate()([kra_objects, kra_levels])
        #modulated = tf.keras.layers.Attention()([bug, state_h])
        modulated_O = tf.keras.layers.Attention()([bug, kra_objects])
        modulated_L = tf.keras.layers.Attention()([bug, kra_levels])
        modulated_OO = tf.keras.layers.Attention()([kra_objects, bug])
        modulated_LL = tf.keras.layers.Attention()([kra_levels, bug])
        modulated = Concatenate()([modulated_O, modulated_L, modulated_OO, modulated_LL])
        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(modulated)

        # Model: _______________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.configs["loss"], metrics=['accuracy'])  ####, tp_rate, tn_rate, fp_rate, fn_rate])
        printout(["Done Building Model", self.__class__.__name__])

    def save_model_summary(self, path):
        '''
        https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
        https://dzone.com/articles/python-101-redirecting-stdout
        :param path:
        :return:
        '''
        original = sys.stdout
        sys.stdout = open(path+"model_summary.txt", 'w')
        self.model.summary()
        sys.stdout = original

class CAR_variant_02:

    def __init__(self, configs):

        self.configs = configs
        choose = Feed_Type()
        self.feed_flag = choose.feed_CAR

        # KRA SEQ OBJECTS
        gru_units = self.configs["gru_units"]
        gru_units = 32
        kra_objects_input = Input(shape=(None, ), name="KRA_SEQ_OBJECTS_INPUT")
        #kra_objects_input = tf.keras.preprocessing.sequence.pad_sequences(kra_objects_input, padding='post')
        kra_objects = Embedding(self.configs["kra_objects_features"],
                            self.configs["kra_objects_embedding"],
                            input_length=None)(kra_objects_input)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.3), name="KRA_SEQ_OBJECTS_GRU_1"
        )(kra_objects)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.2), name="KRA_SEQ_OBJECTS_GRU_2"
        )(kra_objects)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=False), name="KRA_SEQ_OBJECTS_GRU_3"
        )(kra_objects)


        # KRA SEQ LEVELS
        kra_levels_input = Input(shape=(None, ), name="KRA_SEQ_LEVELS_INPUT")
        kra_levels = Embedding(self.configs["kra_levels_features"],
                            self.configs["kra_levels_embedding"],
                            input_length=None)(kra_levels_input)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.3), name="KRA_SEQ_LEVELS_GRU_1"
        )(kra_levels)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.2), name="KRA_SEQ_LEVELS_GRU_2"
        )(kra_levels)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=False), name="KRA_SEQ_LEVELS_GRU_3"
        )(kra_levels)

        # Bug Branch ___________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(self.configs['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(self.configs["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=True, name='BUG_GRU_1')
            )(bug_input)
        bug = Dropout(rate=0.20)(bug)
        bug = Bidirectional(
            GRU(self.hp_bug_gru_latent_dim, return_sequences=True, name='BUG_GRU_2')
        )(bug)
        bug = Dropout(rate=0.20)(bug)
        bug = Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, name='BUG_GRU_3')
            )(bug)

        # Attention query, value, key __________________________________________________________________________________
        # Query = Conditional Probability, Modulator
        # Value = state
        state_h = Concatenate()([kra_objects, kra_levels])
        bug_2x = Concatenate()([bug, bug])
        modulated = tf.keras.layers.Attention()([bug_2x, state_h])
        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(modulated)

        # Model: _______________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.configs["loss"], metrics=['accuracy'])  ####, tp_rate, tn_rate, fp_rate, fn_rate])
        printout(["Done Building Model", self.__class__.__name__])

    def save_model_summary(self, path):
        '''
        https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
        https://dzone.com/articles/python-101-redirecting-stdout
        :param path:
        :return:
        '''
        original = sys.stdout
        sys.stdout = open(path+"model_summary.txt", 'w')
        self.model.summary()
        sys.stdout = original

class CAR_variant_03:

    def __init__(self, configs):

        self.configs = configs
        choose = Feed_Type()
        self.feed_flag = choose.feed_CAR

        # KRA SEQ OBJECTS
        gru_units = self.configs["gru_units"]
        gru_units = 32
        kra_objects_input = Input(shape=(None, ), name="KRA_SEQ_OBJECTS_INPUT")
        #kra_objects_input = tf.keras.preprocessing.sequence.pad_sequences(kra_objects_input, padding='post')
        kra_objects = Embedding(self.configs["kra_objects_features"],
                            self.configs["kra_objects_embedding"],
                            input_length=None)(kra_objects_input)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.3), name="KRA_SEQ_OBJECTS_GRU_1"
        )(kra_objects)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.2), name="KRA_SEQ_OBJECTS_GRU_2"
        )(kra_objects)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=False), name="KRA_SEQ_OBJECTS_GRU_3"
        )(kra_objects)


        # KRA SEQ LEVELS
        kra_levels_input = Input(shape=(None, ), name="KRA_SEQ_LEVELS_INPUT")
        kra_levels = Embedding(self.configs["kra_levels_features"],
                            self.configs["kra_levels_embedding"],
                            input_length=None)(kra_levels_input)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.3), name="KRA_SEQ_LEVELS_GRU_1"
        )(kra_levels)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=True, dropout=0.2), name="KRA_SEQ_LEVELS_GRU_2"
        )(kra_levels)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=False), name="KRA_SEQ_LEVELS_GRU_3"
        )(kra_levels)

        # Bug Branch ___________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(self.configs['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(self.configs["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = Bidirectional(
                GRU(self.hp_bug_gru_latent_dim*2, return_sequences=True, name='BUG_GRU_1')
            )(bug_input)
        bug = Dropout(rate=0.20)(bug)
        bug = Bidirectional(
            GRU(self.hp_bug_gru_latent_dim*2, return_sequences=True, name='BUG_GRU_2')
        )(bug)
        bug = Dropout(rate=0.20)(bug)
        bug = Bidirectional(
                GRU(self.hp_bug_gru_latent_dim*2, return_sequences=False, name='BUG_GRU_3')
            )(bug)

        # Attention ____________________________________________________________________________________________________
        state_h = Concatenate()([kra_objects, kra_levels])
        modulated = tf.keras.layers.Attention()([bug, state_h])
        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(modulated)

        # Model: _______________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.configs["loss"], metrics=['accuracy'])  ####, tp_rate, tn_rate, fp_rate, fn_rate])
        printout(["Done Building Model", self.__class__.__name__])

    def save_model_summary(self, path):
        '''
        https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
        https://dzone.com/articles/python-101-redirecting-stdout
        :param path:
        :return:
        '''
        original = sys.stdout
        sys.stdout = open(path+"model_summary.txt", 'w')
        self.model.summary()
        sys.stdout = original

class CAR_variant_20:

    def __init__(self, configs):

        self.configs = configs
        choose = Feed_Type()
        self.feed_flag = choose.feed_CAR

        # KRA SEQ OBJECTS
        gru_units = self.configs["gru_units"]
        gru_units = 32
        kra_objects_input = Input(shape=(None, ), name="KRA_SEQ_OBJECTS_INPUT")
        #kra_objects_input = tf.keras.preprocessing.sequence.pad_sequences(kra_objects_input, padding='post')
        kra_objects = Embedding(self.configs["kra_objects_features"],
                            self.configs["kra_objects_embedding"],
                            input_length=None)(kra_objects_input)
        cnn_layer_CAR_O = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_O = cnn_layer_CAR_O(kra_objects)

        #kra_objects = Bidirectional(
        #    GRU(units=gru_units, return_sequences=True, dropout=0.3), name="KRA_SEQ_OBJECTS_GRU_1"
        #)(kra_objects)
        #kra_objects = Bidirectional(
        #    GRU(units=gru_units, return_sequences=True, dropout=0.2), name="KRA_SEQ_OBJECTS_GRU_2"
        #)(kra_objects)
        #kra_objects = Bidirectional(
        #    GRU(units=gru_units, return_sequences=True), name="KRA_SEQ_OBJECTS_GRU_3"
        #)(kra_objects)


        # KRA SEQ LEVELS
        kra_levels_input = Input(shape=(None, ), name="KRA_SEQ_LEVELS_INPUT")
        kra_levels = Embedding(self.configs["kra_levels_features"],
                            self.configs["kra_levels_embedding"],
                            input_length=None)(kra_levels_input)
        #kra_levels = Bidirectional(
        #    GRU(units=gru_units, return_sequences=True, dropout=0.3), name="KRA_SEQ_LEVELS_GRU_1"
        #)(kra_levels)
        #kra_levels = Bidirectional(
        #    GRU(units=gru_units, return_sequences=True, dropout=0.2), name="KRA_SEQ_LEVELS_GRU_2"
        #)(kra_levels)
        #kra_levels = Bidirectional(
        #    GRU(units=gru_units, return_sequences=True), name="KRA_SEQ_LEVELS_GRU_3"
        #)(kra_levels)
        cnn_layer_CAR_L = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_L = cnn_layer_CAR_L(kra_levels)

        value_seq_CAR = tf.keras.layers.Concatenate(axis=1)([value_seq_CAR_L, value_seq_CAR_O])


        # Bug Branch ___________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(self.configs['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(self.configs["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        #bug = Bidirectional(
        #        GRU(self.hp_bug_gru_latent_dim*2, return_sequences=True, name='BUG_GRU_1')
        #    )(bug_input)
        #bug = Dropout(rate=0.20)(bug)
        #bug = Bidirectional(
        #    GRU(self.hp_bug_gru_latent_dim*2, return_sequences=True, name='BUG_GRU_2')
        #)(bug)
        #bug = Dropout(rate=0.20)(bug)
        #bug = Bidirectional(
        #        GRU(self.hp_bug_gru_latent_dim*2, return_sequences=True, name='BUG_GRU_3')
        #    )(bug)
        cnn_layer_Bug = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        query_seq_Bug = cnn_layer_Bug(bug_input)

        # Attention ____________________________________________________________________________________________________
        #query_bug_seq = bug
        #values_ = Concatenate()([kra_objects, kra_levels])
        #modulated = tf.keras.layers.Attention()([bug, state_h])
        #binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(modulated)
        query_value_attention_seq = tf.keras.layers.AdditiveAttention()([query_seq_Bug, value_seq_CAR])

        query_encoding = tf.keras.layers.GlobalAveragePooling1D()(query_seq_Bug)
        query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(query_value_attention_seq)

        modulated = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(modulated)

        # Model: _______________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.configs["loss"], metrics=['accuracy'])  ####, tp_rate, tn_rate, fp_rate, fn_rate])
        printout(["Done Building Model", self.__class__.__name__])

    def save_model_summary(self, path):
        '''
        https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
        https://dzone.com/articles/python-101-redirecting-stdout
        :param path:
        :return:
        '''
        original = sys.stdout
        sys.stdout = open(path+"model_summary.txt", 'w')
        self.model.summary()
        sys.stdout = original

class CAR_variant_21:

    def __init__(self, configs):

        self.configs = configs
        choose = Feed_Type()
        self.feed_flag = choose.feed_CAR

        # KRA SEQ OBJECTS
        gru_units = self.configs["gru_units"]
        gru_units = 32
        kra_objects_input = Input(shape=(None, ), name="KRA_SEQ_OBJECTS_INPUT")
        #kra_objects_input = tf.keras.preprocessing.sequence.pad_sequences(kra_objects_input, padding='post')
        kra_objects = Embedding(self.configs["kra_objects_features"],
                            self.configs["kra_objects_embedding"],
                            input_length=None)(kra_objects_input)
        cnn_layer_CAR_O = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_O = cnn_layer_CAR_O(kra_objects)

        # KRA SEQ LEVELS
        kra_levels_input = Input(shape=(None, ), name="KRA_SEQ_LEVELS_INPUT")
        kra_levels = Embedding(self.configs["kra_levels_features"],
                            self.configs["kra_levels_embedding"],
                            input_length=None)(kra_levels_input)
        cnn_layer_CAR_L = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_L = cnn_layer_CAR_L(kra_levels)

        value_seq_CAR = tf.keras.layers.Concatenate(axis=1)([value_seq_CAR_L, value_seq_CAR_O])


        # Bug Branch ___________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(self.configs['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(self.configs["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        cnn_layer_Bug = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        query_seq_Bug = cnn_layer_Bug(bug_input)

        # Attention ____________________________________________________________________________________________________
        query_value_attention_seq = tf.keras.layers.Attention()([query_seq_Bug, value_seq_CAR])
        query_encoding = tf.keras.layers.GlobalAveragePooling1D()(query_seq_Bug)
        query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(query_value_attention_seq)
        modulated = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])

        # DNN ____________________________________________________________________________________________________
        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(modulated)

        # Model: _______________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.configs["loss"], metrics=['accuracy'])  ####, tp_rate, tn_rate, fp_rate, fn_rate])
        printout(["Done Building Model", self.__class__.__name__])

    def save_model_summary(self, path):
        '''
        https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
        https://dzone.com/articles/python-101-redirecting-stdout
        :param path:
        :return:
        '''
        original = sys.stdout
        sys.stdout = open(path+"model_summary.txt", 'w')
        self.model.summary()
        sys.stdout = original

class CAR_variant_22:

    def __init__(self, configs):

        self.configs = configs
        choose = Feed_Type()
        self.feed_flag = choose.feed_CAR

        # KRA SEQ OBJECTS
        gru_units = self.configs["gru_units"]
        gru_units = 32
        kra_objects_input = Input(shape=(None, ), name="KRA_SEQ_OBJECTS_INPUT")
        #kra_objects_input = tf.keras.preprocessing.sequence.pad_sequences(kra_objects_input, padding='post')
        kra_objects = Embedding(self.configs["kra_objects_features"],
                            self.configs["kra_objects_embedding"],
                            input_length=None)(kra_objects_input)
        cnn_layer_CAR_O = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_O = cnn_layer_CAR_O(kra_objects)

        # KRA SEQ LEVELS
        kra_levels_input = Input(shape=(None, ), name="KRA_SEQ_LEVELS_INPUT")
        kra_levels = Embedding(self.configs["kra_levels_features"],
                            self.configs["kra_levels_embedding"],
                            input_length=None)(kra_levels_input)
        cnn_layer_CAR_L = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_L = cnn_layer_CAR_L(kra_levels)

        value_seq_CAR = tf.keras.layers.Concatenate(axis=1)([value_seq_CAR_L, value_seq_CAR_O])


        # Bug Branch ___________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(self.configs['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(self.configs["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        cnn_layer_Bug = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        query_seq_Bug = cnn_layer_Bug(bug_input)
        query_encoding_Bug = tf.keras.layers.GlobalAveragePooling1D()(query_seq_Bug)
        # Attention ____________________________________________________________________________________________________
        attention_custom = Attention_Custom(32)
        query_value_attention, attention_weights = attention_custom(value_seq_CAR, query_encoding_Bug)
        #query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(query_value_attention_seq)
        modulated = tf.keras.layers.Concatenate()([query_encoding_Bug, query_value_attention])

        # DNN ____________________________________________________________________________________________________
        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(modulated)

        # Model: _______________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.configs["loss"], metrics=['accuracy'])  ####, tp_rate, tn_rate, fp_rate, fn_rate])
        printout(["Done Building Model", self.__class__.__name__])

    def save_model_summary(self, path):
        '''
        https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
        https://dzone.com/articles/python-101-redirecting-stdout
        :param path:
        :return:
        '''
        original = sys.stdout
        sys.stdout = open(path+"model_summary.txt", 'w')
        self.model.summary()
        sys.stdout = original

class CAR_variant_23:

    def __init__(self, configs):

        self.configs = configs
        choose = Feed_Type()
        self.feed_flag = choose.feed_CAR

        # KRA SEQ OBJECTS
        gru_units = self.configs["gru_units"]
        gru_units = 32
        kra_objects_input = Input(shape=(None, ), name="KRA_SEQ_OBJECTS_INPUT")
        #kra_objects_input = tf.keras.preprocessing.sequence.pad_sequences(kra_objects_input, padding='post')
        kra_objects = Embedding(self.configs["kra_objects_features"],
                            self.configs["kra_objects_embedding"],
                            input_length=None)(kra_objects_input)
        cnn_layer_CAR_O = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        car_o = cnn_layer_CAR_O(kra_objects)

        # KRA SEQ LEVELS
        kra_levels_input = Input(shape=(None, ), name="KRA_SEQ_LEVELS_INPUT")
        kra_levels = Embedding(self.configs["kra_levels_features"],
                            self.configs["kra_levels_embedding"],
                            input_length=None)(kra_levels_input)
        cnn_layer_CAR_L = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        car_l = cnn_layer_CAR_L(kra_levels)

        car_l = tf.keras.layers.GlobalAveragePooling1D()(car_l)
        car_o = tf.keras.layers.GlobalAveragePooling1D()(car_o)
        #car = tf.keras.layers.Concatenate(axis=1)([car_l, car_o])


        # Bug Branch ___________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(self.configs['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(self.configs["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        cnn_layer_Bug = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        bug = cnn_layer_Bug(bug_input)
        bug = tf.keras.layers.GlobalAveragePooling1D()(bug)

        # Attention ____________________________________________________________________________________________________
        #bug = Reshape((1, 64), name="xxx1")(bug)
        #car_o = Reshape((1, 64), name="xxx2")(kra_objects)
        #car_l = Reshape((1, 64), name="xxx3")(kra_levels)
        car_o = Activation('softmax')(car_o)
        car_l = Activation('softmax')(car_l)

        attKraObjects = attention_distribution(bug, car_o)
        sim_bug_kra_objects = Multiply()([attKraObjects, car_o])
        attKraLevels = attention_distribution(bug, car_l)
        sim_bug_kra_levels = Multiply()([attKraLevels, car_l])

        combined = Concatenate(axis=-2)(
            [bug, sim_bug_kra_objects, sim_bug_kra_levels])
        combined = Flatten()(combined)

        # DNN ____________________________________________________________________________________________________
        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)

        # Model: _______________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.configs["loss"], metrics=['accuracy'])  ####, tp_rate, tn_rate, fp_rate, fn_rate])
        printout(["Done Building Model", self.__class__.__name__])

    def save_model_summary(self, path):
        '''
        https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
        https://dzone.com/articles/python-101-redirecting-stdout
        :param path:
        :return:
        '''
        original = sys.stdout
        sys.stdout = open(path+"model_summary.txt", 'w')
        self.model.summary()
        sys.stdout = original

class CAR_variant_24:

    def __init__(self, configs):

        self.configs = configs
        choose = Feed_Type()
        self.feed_flag = choose.feed_CAR

        # KRA SEQ OBJECTS
        gru_units = self.configs["gru_units"]
        gru_units = 32
        kra_objects_input = Input(shape=(None, ), name="KRA_SEQ_OBJECTS_INPUT")
        #kra_objects_input = tf.keras.preprocessing.sequence.pad_sequences(kra_objects_input, padding='post')
        kra_objects = Embedding(self.configs["kra_objects_features"],
                            self.configs["kra_objects_embedding"],
                            input_length=None)(kra_objects_input)
        cnn_layer_CAR_O = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_O = cnn_layer_CAR_O(kra_objects)

        # KRA SEQ LEVELS
        kra_levels_input = Input(shape=(None, ), name="KRA_SEQ_LEVELS_INPUT")
        kra_levels = Embedding(self.configs["kra_levels_features"],
                            self.configs["kra_levels_embedding"],
                            input_length=None)(kra_levels_input)
        cnn_layer_CAR_L = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_L = cnn_layer_CAR_L(kra_levels)

        car_seq = tf.keras.layers.Concatenate(axis=1)([value_seq_CAR_L, value_seq_CAR_O])


        # Bug Branch ___________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(self.configs['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(self.configs["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        cnn_layer_Bug = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        bug_seq = cnn_layer_Bug(bug_input)

        # Attention ____________________________________________________________________________________________________
        bug = tf.keras.layers.GlobalAveragePooling1D()(bug_seq)
        car = tf.keras.layers.GlobalAveragePooling1D()(car_seq)
        modulated = tf.keras.layers.Concatenate()([bug, car])

        # DNN ____________________________________________________________________________________________________
        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(modulated)

        # Model: _______________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.configs["loss"], metrics=['accuracy'])  ####, tp_rate, tn_rate, fp_rate, fn_rate])
        printout(["Done Building Model", self.__class__.__name__])

    def save_model_summary(self, path):
        '''
        https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
        https://dzone.com/articles/python-101-redirecting-stdout
        :param path:
        :return:
        '''
        original = sys.stdout
        sys.stdout = open(path+"model_summary.txt", 'w')
        self.model.summary()
        sys.stdout = original

class CAR_variant_30:

    def __init__(self, configs):

        self.configs = configs
        choose = Feed_Type()
        self.feed_flag = choose.feed_CAR

        # KRA SEQ OBJECTS
        gru_units = self.configs["gru_units"]
        gru_units = 32
        kra_objects_input = Input(shape=(None, ), name="KRA_SEQ_OBJECTS_INPUT")
        #kra_objects_input = tf.keras.preprocessing.sequence.pad_sequences(kra_objects_input, padding='post')
        kra_objects = Embedding(self.configs["kra_objects_features"],
                            self.configs["kra_objects_embedding"],
                            input_length=None)(kra_objects_input)
        cnn_layer_CAR_O = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_O = cnn_layer_CAR_O(kra_objects)

        # KRA SEQ LEVELS
        kra_levels_input = Input(shape=(None, ), name="KRA_SEQ_LEVELS_INPUT")
        kra_levels = Embedding(self.configs["kra_levels_features"],
                            self.configs["kra_levels_embedding"],
                            input_length=None)(kra_levels_input)
        cnn_layer_CAR_L = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_L = cnn_layer_CAR_L(kra_levels)

        value_seq_CAR = tf.keras.layers.Concatenate(axis=1)([value_seq_CAR_L, value_seq_CAR_O])


        # Bug Branch ___________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(self.configs['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(self.configs["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        cnn_layer_Bug = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        query_seq_Bug = cnn_layer_Bug(bug_input)

        # Attention ____________________________________________________________________________________________________
        seq_Bug_to_CAR = tf.keras.layers.Attention()([query_seq_Bug, value_seq_CAR])
        seq_CAR_to_Bug = tf.keras.layers.Attention()([value_seq_CAR, query_seq_Bug])
        bug_encoding = tf.keras.layers.GlobalAveragePooling1D()(query_seq_Bug)
        CAR_encoding = tf.keras.layers.GlobalAveragePooling1D()(value_seq_CAR)
        encoded_Bug_to_CAR = tf.keras.layers.GlobalAveragePooling1D()(seq_Bug_to_CAR)
        encoded_CAR_to_Bug = tf.keras.layers.GlobalAveragePooling1D()(seq_CAR_to_Bug)
        modulated = tf.keras.layers.Concatenate()([bug_encoding, CAR_encoding, encoded_CAR_to_Bug, encoded_Bug_to_CAR])

        # DNN ____________________________________________________________________________________________________
        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(modulated)

        # Model: _______________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.configs["loss"], metrics=['accuracy'])  ####, tp_rate, tn_rate, fp_rate, fn_rate])
        printout(["Done Building Model", self.__class__.__name__])

    def save_model_summary(self, path):
        '''
        https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
        https://dzone.com/articles/python-101-redirecting-stdout
        :param path:
        :return:
        '''
        original = sys.stdout
        sys.stdout = open(path+"model_summary.txt", 'w')
        self.model.summary()
        sys.stdout = original

class CAR_variant_31:

    def __init__(self, configs):

        self.configs = configs
        choose = Feed_Type()
        self.feed_flag = choose.feed_CAR

        # KRA SEQ OBJECTS
        gru_units = self.configs["gru_units"]
        gru_units = 32
        kra_objects_input = Input(shape=(None, ), name="KRA_SEQ_OBJECTS_INPUT")
        #kra_objects_input = tf.keras.preprocessing.sequence.pad_sequences(kra_objects_input, padding='post')
        kra_objects = Embedding(self.configs["kra_objects_features"],
                            self.configs["kra_objects_embedding"],
                            input_length=None)(kra_objects_input)
        cnn_layer_CAR_O = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_O = cnn_layer_CAR_O(kra_objects)

        # KRA SEQ LEVELS
        kra_levels_input = Input(shape=(None, ), name="KRA_SEQ_LEVELS_INPUT")
        kra_levels = Embedding(self.configs["kra_levels_features"],
                            self.configs["kra_levels_embedding"],
                            input_length=None)(kra_levels_input)
        cnn_layer_CAR_L = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_L = cnn_layer_CAR_L(kra_levels)

        value_seq_CAR = tf.keras.layers.Concatenate(axis=1)([value_seq_CAR_L, value_seq_CAR_O])


        # Bug Branch ___________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(self.configs['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(self.configs["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        cnn_layer_Bug = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        query_seq_Bug = cnn_layer_Bug(bug_input)

        # Attention ____________________________________________________________________________________________________
        seq_CAR_with_Bug = tf.keras.layers.Attention()([query_seq_Bug, value_seq_CAR])
        seq_Bug_with_CAR = tf.keras.layers.Attention()([value_seq_CAR, query_seq_Bug])
        encoded_Bug_to_CAR = tf.keras.layers.GlobalAveragePooling1D()(seq_CAR_with_Bug)
        encoded_CAR_to_Bug = tf.keras.layers.GlobalAveragePooling1D()(seq_Bug_with_CAR)
        modulated = tf.keras.layers.Concatenate()([encoded_CAR_to_Bug, encoded_Bug_to_CAR])

        # DNN ____________________________________________________________________________________________________
        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(modulated)

        # Model: _______________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.configs["loss"], metrics=['accuracy'])  ####, tp_rate, tn_rate, fp_rate, fn_rate])
        printout(["Done Building Model", self.__class__.__name__])

    def save_model_summary(self, path):
        '''
        https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
        https://dzone.com/articles/python-101-redirecting-stdout
        :param path:
        :return:
        '''
        original = sys.stdout
        sys.stdout = open(path+"model_summary.txt", 'w')
        self.model.summary()
        sys.stdout = original

class CAR_variant_32:

    def __init__(self, configs):

        self.configs = configs
        choose = Feed_Type()
        self.feed_flag = choose.feed_CAR

        # KRA SEQ OBJECTS
        gru_units = self.configs["gru_units"]
        gru_units = 32
        kra_objects_input = Input(shape=(None, ), name="KRA_SEQ_OBJECTS_INPUT")
        #kra_objects_input = tf.keras.preprocessing.sequence.pad_sequences(kra_objects_input, padding='post')
        kra_objects = Embedding(self.configs["kra_objects_features"],
                            self.configs["kra_objects_embedding"],
                            input_length=None)(kra_objects_input)
        cnn_layer_CAR_O = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_O = cnn_layer_CAR_O(kra_objects)

        # KRA SEQ LEVELS
        kra_levels_input = Input(shape=(None, ), name="KRA_SEQ_LEVELS_INPUT")
        kra_levels = Embedding(self.configs["kra_levels_features"],
                            self.configs["kra_levels_embedding"],
                            input_length=None)(kra_levels_input)
        cnn_layer_CAR_L = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_L = cnn_layer_CAR_L(kra_levels)

        value_seq_CAR = tf.keras.layers.Concatenate(axis=1)([value_seq_CAR_L, value_seq_CAR_O])


        # Bug Branch ___________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(self.configs['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(self.configs["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        cnn_layer_Bug = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        query_seq_Bug = cnn_layer_Bug(bug_input)

        # Attention ____________________________________________________________________________________________________
        seq_CAR_O_with_Bug = tf.keras.layers.Attention()([query_seq_Bug, value_seq_CAR_O])
        seq_CAR_L_with_Bug = tf.keras.layers.Attention()([query_seq_Bug, value_seq_CAR_L])
        # seq_Bug_with_CAR = tf.keras.layers.Attention()([value_seq_CAR, query_seq_Bug])
        encoded_Bug = tf.keras.layers.GlobalAveragePooling1D()(query_seq_Bug)
        encoded_CAR_O_with_Bug = tf.keras.layers.GlobalAveragePooling1D()(seq_CAR_O_with_Bug)
        encoded_CAR_L_with_Bug = tf.keras.layers.GlobalAveragePooling1D()(seq_CAR_L_with_Bug)
        modulated = tf.keras.layers.Concatenate()([encoded_Bug, encoded_CAR_O_with_Bug, encoded_CAR_L_with_Bug])

        # DNN ____________________________________________________________________________________________________
        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(modulated)

        # Model: _______________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.configs["loss"], metrics=['accuracy'])  ####, tp_rate, tn_rate, fp_rate, fn_rate])
        printout(["Done Building Model", self.__class__.__name__])

    def save_model_summary(self, path):
        '''
        https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
        https://dzone.com/articles/python-101-redirecting-stdout
        :param path:
        :return:
        '''
        original = sys.stdout
        sys.stdout = open(path+"model_summary.txt", 'w')
        self.model.summary()
        sys.stdout = original

class CAR_variant_33:

    def __init__(self, configs):

        self.configs = configs
        choose = Feed_Type()
        self.feed_flag = choose.feed_CAR

        # KRA SEQ OBJECTS
        gru_units = self.configs["gru_units"]
        gru_units = 32
        kra_objects_input = Input(shape=(None, ), name="KRA_SEQ_OBJECTS_INPUT")
        #kra_objects_input = tf.keras.preprocessing.sequence.pad_sequences(kra_objects_input, padding='post')
        kra_objects = Embedding(self.configs["kra_objects_features"],
                            self.configs["kra_objects_embedding"],
                            input_length=None)(kra_objects_input)
        cnn_layer_CAR_O = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_O = cnn_layer_CAR_O(kra_objects)

        # KRA SEQ LEVELS
        kra_levels_input = Input(shape=(None, ), name="KRA_SEQ_LEVELS_INPUT")
        kra_levels = Embedding(self.configs["kra_levels_features"],
                            self.configs["kra_levels_embedding"],
                            input_length=None)(kra_levels_input)
        cnn_layer_CAR_L = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_L = cnn_layer_CAR_L(kra_levels)

        value_seq_CAR = tf.keras.layers.Concatenate(axis=1)([value_seq_CAR_L, value_seq_CAR_O])


        # Bug Branch ___________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(self.configs['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(self.configs["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        cnn_layer_Bug = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        query_seq_Bug = cnn_layer_Bug(bug_input)

        # Attention ____________________________________________________________________________________________________
        seq_CAR_O_with_Bug = tf.keras.layers.Attention()([query_seq_Bug, value_seq_CAR_O])
        seq_CAR_L_with_Bug = tf.keras.layers.Attention()([query_seq_Bug, value_seq_CAR_L])
        encoded_Bug = tf.keras.layers.GlobalAveragePooling1D()(query_seq_Bug)
        encoded_CAR_L = tf.keras.layers.GlobalAveragePooling1D()(value_seq_CAR_L)
        encoded_CAR_O = tf.keras.layers.GlobalAveragePooling1D()(value_seq_CAR_O)
        encoded_CAR_O_with_Bug = tf.keras.layers.GlobalAveragePooling1D()(seq_CAR_O_with_Bug)
        encoded_CAR_L_with_Bug = tf.keras.layers.GlobalAveragePooling1D()(seq_CAR_L_with_Bug)
        modulated = tf.keras.layers.Concatenate()([encoded_Bug, encoded_CAR_L, encoded_CAR_O, encoded_CAR_O_with_Bug, encoded_CAR_L_with_Bug])

        # DNN ____________________________________________________________________________________________________
        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(modulated)

        # Model: _______________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.configs["loss"], metrics=['accuracy'])  ####, tp_rate, tn_rate, fp_rate, fn_rate])
        printout(["Done Building Model", self.__class__.__name__])

    def save_model_summary(self, path):
        '''
        https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
        https://dzone.com/articles/python-101-redirecting-stdout
        :param path:
        :return:
        '''
        original = sys.stdout
        sys.stdout = open(path+"model_summary.txt", 'w')
        self.model.summary()
        sys.stdout = original

class CAR_variant_34:

    def __init__(self, configs):

        self.configs = configs
        choose = Feed_Type()
        self.feed_flag = choose.feed_CAR

        # KRA SEQ OBJECTS
        gru_units = self.configs["gru_units"]
        gru_units = 32
        kra_objects_input = Input(shape=(None, ), name="KRA_SEQ_OBJECTS_INPUT")
        #kra_objects_input = tf.keras.preprocessing.sequence.pad_sequences(kra_objects_input, padding='post')
        kra_objects = Embedding(self.configs["kra_objects_features"],
                            self.configs["kra_objects_embedding"],
                            input_length=None)(kra_objects_input)
        cnn_layer_CAR_O = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_O = cnn_layer_CAR_O(kra_objects)

        # KRA SEQ LEVELS
        kra_levels_input = Input(shape=(None, ), name="KRA_SEQ_LEVELS_INPUT")
        kra_levels = Embedding(self.configs["kra_levels_features"],
                            self.configs["kra_levels_embedding"],
                            input_length=None)(kra_levels_input)
        cnn_layer_CAR_L = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_L = cnn_layer_CAR_L(kra_levels)

        value_seq_CAR = tf.keras.layers.Concatenate(axis=1)([value_seq_CAR_L, value_seq_CAR_O])


        # Bug Branch ___________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(self.configs['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(self.configs["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        cnn_layer_Bug = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        query_seq_Bug = cnn_layer_Bug(bug_input)

        # Attention ____________________________________________________________________________________________________
        query_value_attention_seq = tf.keras.layers.Attention()([value_seq_CAR, query_seq_Bug])
        query_encoding = tf.keras.layers.GlobalAveragePooling1D()(value_seq_CAR)
        query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(query_value_attention_seq)
        modulated = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])

        # DNN ____________________________________________________________________________________________________
        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(modulated)

        # Model: _______________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.configs["loss"], metrics=['accuracy'])  ####, tp_rate, tn_rate, fp_rate, fn_rate])
        printout(["Done Building Model", self.__class__.__name__])

    def save_model_summary(self, path):
        '''
        https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
        https://dzone.com/articles/python-101-redirecting-stdout
        :param path:
        :return:
        '''
        original = sys.stdout
        sys.stdout = open(path+"model_summary.txt", 'w')
        self.model.summary()
        sys.stdout = original

class CAR_variant_35:

    def __init__(self, configs):

        self.configs = configs
        choose = Feed_Type()
        self.feed_flag = choose.feed_CAR

        # KRA SEQ OBJECTS
        gru_units = self.configs["gru_units"]
        gru_units = 32
        kra_objects_input = Input(shape=(None, ), name="KRA_SEQ_OBJECTS_INPUT")
        #kra_objects_input = tf.keras.preprocessing.sequence.pad_sequences(kra_objects_input, padding='post')
        kra_objects = Embedding(self.configs["kra_objects_features"],
                            self.configs["kra_objects_embedding"],
                            input_length=None)(kra_objects_input)
        cnn_layer_CAR_O = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_O = cnn_layer_CAR_O(kra_objects)

        # KRA SEQ LEVELS
        kra_levels_input = Input(shape=(None, ), name="KRA_SEQ_LEVELS_INPUT")
        kra_levels = Embedding(self.configs["kra_levels_features"],
                            self.configs["kra_levels_embedding"],
                            input_length=None)(kra_levels_input)
        cnn_layer_CAR_L = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_L = cnn_layer_CAR_L(kra_levels)

        value_seq_CAR = tf.keras.layers.Concatenate(axis=1)([value_seq_CAR_L, value_seq_CAR_O])


        # Bug Branch ___________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(self.configs['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(self.configs["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        cnn_layer_Bug = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        query_seq_Bug = cnn_layer_Bug(bug_input)

        # Attention ____________________________________________________________________________________________________
        seq_CAR_O_with_Bug = tf.keras.layers.Attention()([query_seq_Bug, value_seq_CAR_O])
        seq_CAR_L_with_Bug = tf.keras.layers.Attention()([query_seq_Bug, value_seq_CAR_L])
        seq_Bug_with_CAR_L = tf.keras.layers.Attention()([value_seq_CAR_L, query_seq_Bug])
        seq_Bug_with_CAR_O = tf.keras.layers.Attention()([value_seq_CAR_O, query_seq_Bug])

        # seq_Bug_with_CAR = tf.keras.layers.Attention()([value_seq_CAR, query_seq_Bug])
        encoded_Bug = tf.keras.layers.GlobalAveragePooling1D()(query_seq_Bug)
        encoded_CAR_O = tf.keras.layers.GlobalAveragePooling1D()(value_seq_CAR_O)
        encoded_CAR_L = tf.keras.layers.GlobalAveragePooling1D()(value_seq_CAR_L)

        encoded_CAR_O_with_Bug = tf.keras.layers.GlobalAveragePooling1D()(seq_CAR_O_with_Bug)
        encoded_CAR_L_with_Bug = tf.keras.layers.GlobalAveragePooling1D()(seq_CAR_L_with_Bug)
        encoded_Bug_with_CAR_O = tf.keras.layers.GlobalAveragePooling1D()(seq_Bug_with_CAR_O)
        encoded_Bug_with_CAR_L = tf.keras.layers.GlobalAveragePooling1D()(seq_Bug_with_CAR_L)
        modulated = tf.keras.layers.Concatenate()([encoded_Bug, encoded_CAR_O, encoded_CAR_L, encoded_CAR_O_with_Bug, encoded_CAR_L_with_Bug, encoded_Bug_with_CAR_O, encoded_Bug_with_CAR_L])

        # DNN ____________________________________________________________________________________________________
        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(modulated)

        # Model: _______________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.configs["loss"], metrics=['accuracy'])  ####, tp_rate, tn_rate, fp_rate, fn_rate])
        printout(["Done Building Model", self.__class__.__name__])

    def save_model_summary(self, path):
        '''
        https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
        https://dzone.com/articles/python-101-redirecting-stdout
        :param path:
        :return:
        '''
        original = sys.stdout
        sys.stdout = open(path+"model_summary.txt", 'w')
        self.model.summary()
        sys.stdout = original

class CAR_variant_36:

    def __init__(self, configs):

        self.configs = configs
        choose = Feed_Type()
        self.feed_flag = choose.feed_CAR

        # KRA SEQ OBJECTS
        gru_units = self.configs["gru_units"]
        gru_units = 32
        kra_objects_input = Input(shape=(None, ), name="KRA_SEQ_OBJECTS_INPUT")
        #kra_objects_input = tf.keras.preprocessing.sequence.pad_sequences(kra_objects_input, padding='post')
        kra_objects = Embedding(self.configs["kra_objects_features"],
                            self.configs["kra_objects_embedding"],
                            input_length=None)(kra_objects_input)
        cnn_layer_CAR_O = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_O = cnn_layer_CAR_O(kra_objects)

        # KRA SEQ LEVELS
        kra_levels_input = Input(shape=(None, ), name="KRA_SEQ_LEVELS_INPUT")
        kra_levels = Embedding(self.configs["kra_levels_features"],
                            self.configs["kra_levels_embedding"],
                            input_length=None)(kra_levels_input)
        cnn_layer_CAR_L = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_L = cnn_layer_CAR_L(kra_levels)

        value_seq_CAR = tf.keras.layers.Concatenate(axis=1)([value_seq_CAR_L, value_seq_CAR_O])


        # Bug Branch ___________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(self.configs['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(self.configs["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        cnn_layer_Bug = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        query_seq_Bug = cnn_layer_Bug(bug_input)

        # Attention ____________________________________________________________________________________________________
        seq_CAR_O_with_Bug = tf.keras.layers.Attention()([query_seq_Bug, value_seq_CAR_O])
        seq_CAR_L_with_Bug = tf.keras.layers.Attention()([query_seq_Bug, value_seq_CAR_L])
        seq_Bug_with_CAR_L = tf.keras.layers.Attention()([value_seq_CAR_L, query_seq_Bug])
        seq_Bug_with_CAR_O = tf.keras.layers.Attention()([value_seq_CAR_O, query_seq_Bug])

        # seq_Bug_with_CAR = tf.keras.layers.Attention()([value_seq_CAR, query_seq_Bug])
        encoded_Bug = tf.keras.layers.GlobalAveragePooling1D()(query_seq_Bug)
        encoded_CAR_O = tf.keras.layers.GlobalAveragePooling1D()(value_seq_CAR_O)
        encoded_CAR_L = tf.keras.layers.GlobalAveragePooling1D()(value_seq_CAR_L)

        encoded_CAR_O_with_Bug = tf.keras.layers.GlobalAveragePooling1D()(seq_CAR_O_with_Bug)
        encoded_CAR_L_with_Bug = tf.keras.layers.GlobalAveragePooling1D()(seq_CAR_L_with_Bug)
        encoded_Bug_with_CAR_O = tf.keras.layers.GlobalAveragePooling1D()(seq_Bug_with_CAR_O)
        encoded_Bug_with_CAR_L = tf.keras.layers.GlobalAveragePooling1D()(seq_Bug_with_CAR_L)
        modulated = tf.keras.layers.Concatenate()([encoded_Bug, encoded_CAR_O, encoded_CAR_L, encoded_CAR_O_with_Bug, encoded_CAR_L_with_Bug, encoded_Bug_with_CAR_O, encoded_Bug_with_CAR_L])

        # DNN ____________________________________________________________________________________________________
        modulated = Dense(300, activation='sigmoid', name='FC')(modulated)
        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(modulated)

        # Model: _______________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.configs["loss"], metrics=['accuracy'])  ####, tp_rate, tn_rate, fp_rate, fn_rate])
        printout(["Done Building Model", self.__class__.__name__])

    def save_model_summary(self, path):
        '''
        https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
        https://dzone.com/articles/python-101-redirecting-stdout
        :param path:
        :return:
        '''
        original = sys.stdout
        sys.stdout = open(path+"model_summary.txt", 'w')
        self.model.summary()
        sys.stdout = original

class CAR_variant_37:

    def __init__(self, configs):

        self.configs = configs
        choose = Feed_Type()
        self.feed_flag = choose.feed_CAR

        # KRA SEQ OBJECTS
        gru_units = self.configs["gru_units"]
        gru_units = 32
        kra_objects_input = Input(shape=(None, ), name="KRA_SEQ_OBJECTS_INPUT")
        #kra_objects_input = tf.keras.preprocessing.sequence.pad_sequences(kra_objects_input, padding='post')
        kra_objects = Embedding(self.configs["kra_objects_features"],
                            self.configs["kra_objects_embedding"],
                            input_length=None)(kra_objects_input)
        cnn_layer_CAR_O = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_O = cnn_layer_CAR_O(kra_objects)

        # KRA SEQ LEVELS
        kra_levels_input = Input(shape=(None, ), name="KRA_SEQ_LEVELS_INPUT")
        kra_levels = Embedding(self.configs["kra_levels_features"],
                            self.configs["kra_levels_embedding"],
                            input_length=None)(kra_levels_input)
        cnn_layer_CAR_L = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        value_seq_CAR_L = cnn_layer_CAR_L(kra_levels)

        value_seq_CAR = tf.keras.layers.Concatenate(axis=1)([value_seq_CAR_L, value_seq_CAR_O])


        # Bug Branch ___________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(self.configs['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(self.configs["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        cnn_layer_Bug = tf.keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')
        query_seq_Bug = cnn_layer_Bug(bug_input)

        # Attention ____________________________________________________________________________________________________
        seq_CAR_O_with_CAR_L = tf.keras.layers.Attention()([value_seq_CAR_L, value_seq_CAR_O])
        seq_Bug_with_CAR_O_with_CAR_L = tf.keras.layers.Attention()([seq_CAR_O_with_CAR_L, query_seq_Bug])
        seq_CAR_O_with_CAR_L_with_Bug = tf.keras.layers.Attention()([query_seq_Bug, seq_CAR_O_with_CAR_L])

        # seq_Bug_with_CAR = tf.keras.layers.Attention()([value_seq_CAR, query_seq_Bug])
        encoded_Bug = tf.keras.layers.GlobalAveragePooling1D()(query_seq_Bug)
        encoded_CAR_O = tf.keras.layers.GlobalAveragePooling1D()(value_seq_CAR_O)
        encoded_CAR_L = tf.keras.layers.GlobalAveragePooling1D()(value_seq_CAR_L)

        encoded_Bug_with_CAR_O_with_CAR_L = tf.keras.layers.GlobalAveragePooling1D()(seq_Bug_with_CAR_O_with_CAR_L)
        encoded_CAR_O_with_CAR_L_with_Bug = tf.keras.layers.GlobalAveragePooling1D()(seq_CAR_O_with_CAR_L_with_Bug)

        modulated = tf.keras.layers.Concatenate()([encoded_Bug, encoded_CAR_O, encoded_CAR_L, encoded_Bug_with_CAR_O_with_CAR_L, encoded_CAR_O_with_CAR_L_with_Bug])

        # DNN ____________________________________________________________________________________________________
        modulated = Dense(300, activation='sigmoid', name='FC')(modulated)
        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(modulated)

        # Model: _______________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.configs["loss"], metrics=['accuracy'])  ####, tp_rate, tn_rate, fp_rate, fn_rate])
        printout(["Done Building Model", self.__class__.__name__])

    def save_model_summary(self, path):
        '''
        https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
        https://dzone.com/articles/python-101-redirecting-stdout
        :param path:
        :return:
        '''
        original = sys.stdout
        sys.stdout = open(path+"model_summary.txt", 'w')
        self.model.summary()
        sys.stdout = original