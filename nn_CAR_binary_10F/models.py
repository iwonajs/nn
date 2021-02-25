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

def tp_rate(y_true, y_pred):
    #https://datascience.stackexchange.com/questions/33587/keras-custom-loss-function-as-true-negatives-by-true-negatives-plus-false-posit
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

class Model_Img_Attention_Original:

    def __init__(self, config_file):

        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())

        # Code Image Branch ____________________________________________________________________________________________________
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        code_input_img = Input(shape=(128, 128, 3), name="CODE_IMG")
        self.img_units = config["img_units"]

        conv1 = Conv2D(32, (3, 3))(code_input_img)
        conv1 = Activation('relu')(conv1)
        conv1 = MaxPool2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(32, (3, 3))(conv1)
        conv2 = Activation('relu')(conv2)
        conv2 = MaxPool2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(64, (3, 3))(conv2)
        conv3 = Activation('relu')(conv3)
        conv3 = MaxPool2D(pool_size=(2, 2))(conv3)
        flat_img = Flatten()(conv3)
        dense_img = Dense(self.img_units)(flat_img)
        img = img_attention(dense_img, self.img_units)

        # code_img = Activation('sigmoid')(dense_img)
        # print("code_img", flat_img.shape)
        # img = RepeatVector(1)(code_img)
        # img = Reshape((1,64))(code_img)

        # print("code_img", img.shape)

        # Code Branch __________________________________________________________________________________________________________
        self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        self.hp_code_embedding_dim = int(config["code_embedding"])
        self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        code_input_f2_xyz = Input(shape=(None,), name="CODE_TOKEN")

        code_embedding_f2_xyz = Embedding(
            input_dim=self.hp_code_dim2_max_word_features,
            output_dim=self.hp_code_embedding_dim)(code_input_f2_xyz)
        ###input_length=hp_code_time_steps)(code_input_f2_xyz)

        code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        code_gru = \
            Bidirectional(
                GRU(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]), name='CODE_GRU_1'
            )(code_input)

        # code_gru = \
        #    Bidirectional(
        #        GRU(units=hp_code_gru_latent_dim, return_sequences=True, dropout=config["dropout"]), name='CODE_GRU_2'
        #    )(code_gru)

        code = gru_attention_last_step(code_gru, self.hp_code_gru_latent_dim)
        print("code", code.shape)

        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug_gru = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)

        # bug_gru = \
        #    Bidirectional(
        #        GRU(hp_bug_gru_latent_dim, return_sequences=True, dropout=config["dropout"], name='BUG_GRU_2')
        #    )(bug_gru)

        bug = gru_attention_last_step(bug_gru, self.hp_bug_gru_latent_dim)

        # Combined _____________________________________________________________________________________________________________
        print("test code", code.shape)
        print("test bug", bug.shape)
        # print("test img", dense_img.shape)
        combined = Concatenate(axis=-1)([code, img, bug])
        print("combined", combined.shape)

        # combo = Concatenate(axis=-1)([code, bug])
        # combo = combined_attention(combo)

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT', W_regularizer=l2(0.01))(combined)
        print("binary", binary.shape)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz, code_input_img], outputs=binary)
        #self.model.summary()

        #return model
        print("Done Building Model.")

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

class Base_Model_GPU:

    def __init__(self, config_file):

        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())

        mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
        with mirrored_strategy.scope():
    #        # Code Image Branch ____________________________________________________________________________________________________
    #        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    #        code_input_img = Input(shape=(128, 128, 3), name="CODE_IMG")
    #        self.img_units = config["img_units"]
    #
    #        conv1 = Conv2D(32, (3, 3))(code_input_img)
    #        conv1 = Activation('relu')(conv1)
    #        conv1 = MaxPool2D(pool_size=(2, 2))(conv1)
    #        conv2 = Conv2D(32, (3, 3))(conv1)
    #        conv2 = Activation('relu')(conv2)
    #        conv2 = MaxPool2D(pool_size=(2, 2))(conv2)
    #        conv3 = Conv2D(64, (3, 3))(conv2)
    #        conv3 = Activation('relu')(conv3)
    #        conv3 = MaxPool2D(pool_size=(2, 2))(conv3)
    #        flat_img = Flatten()(conv3)
    #        dense_img = Dense(self.img_units)(flat_img)
    #        img = img_attention(dense_img, self.img_units)

            # code_img = Activation('sigmoid')(dense_img)
            # print("code_img", flat_img.shape)
            # img = RepeatVector(1)(code_img)
            # img = Reshape((1,64))(code_img)

            # print("code_img", img.shape)

            # Code Branch __________________________________________________________________________________________________________
            self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
            self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
            self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
            self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
            self.hp_code_gru_latent_dim = int(config["code_gru_units"])

            # Variable Sequence Length, shape=(None...
            # Code Dim 1 is one-hot-encoded
            # Code Dim 2 is vectorized, (NOT one-hot encoded)
            #print("iwona:",)
            code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
            code_input_f2_xyz = Input(shape=(None, ), name="CODE_TOKEN")

            #code_embedding_f1_cat = Embedding(self.hp_code_dim1_max_word_features, self.hp_code_embedding_dim_1,
            #    input_length=None
            #    )(code_input_f1_cat)
            #print(code_embedding_f1_cat.shape)


            code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
                input_length=None
                )(code_input_f2_xyz)
            #input_length=hp_code_time_steps)(code_input_f2_xyz)
            #print(code_embedding_f2_xyz.shape)
            #code_input = Concatenate()([code_embedding_f1_cat, code_embedding_f2_xyz])
            code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

            # Encode the code information (dim1+dim2)
            # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
            code_gru = \
                Bidirectional(
                    GRU(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
                    name='CODE_GRU_1'
                )(code_input)

            # code_gru = \
            #    Bidirectional(
            #        GRU(units=hp_code_gru_latent_dim, return_sequences=True, dropout=config["dropout"]), name='CODE_GRU_2'
            #    )(code_gru)

            code = gru_attention_last_step(code_gru, self.hp_code_gru_latent_dim)
            print("code", code.shape)

            # Bug Branch ___________________________________________________________________________________________________________
            self.hp_bug_embedding_dim = int(config['bug_embedding'])
            self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

            # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
            bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
            bug_gru = \
                Bidirectional(
                    GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
                )(bug_input)

            # bug_gru = \
            #    Bidirectional(
            #        GRU(hp_bug_gru_latent_dim, return_sequences=True, dropout=config["dropout"], name='BUG_GRU_2')
            #    )(bug_gru)

            bug = gru_attention_last_step(bug_gru, self.hp_bug_gru_latent_dim)

            # Combined _____________________________________________________________________________________________________________
            print("test code", code.shape)
            print("test bug", bug.shape)
            # print("test img", dense_img.shape)
            combined = Concatenate(axis=-1)([code, bug])
            print("combined", combined.shape)

            # combo = Concatenate(axis=-1)([code, bug])
            # combo = combined_attention(combo)

            binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)  ##############, W_regularizer=l2(0.01))(combined)
            print("binary", binary.shape)

            # Model: _______________________________________________________________________________________________________________
            self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz], outputs=binary)
            self.model.compile(optimizer=RMSprop(lr=0.001), loss=config["loss"], metrics=['accuracy', tp_rate, tn_rate])
            #self.model.summary()

            #return model
            print("Done Building Model.")

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

class Base_Model:

    def __init__(self, config_file):

        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())

        #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2"])
        #with mirrored_strategy.scope():
#        # Code Image Branch ____________________________________________________________________________________________________
#        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
#        code_input_img = Input(shape=(128, 128, 3), name="CODE_IMG")
#        self.img_units = config["img_units"]
#
#        conv1 = Conv2D(32, (3, 3))(code_input_img)
#        conv1 = Activation('relu')(conv1)
#        conv1 = MaxPool2D(pool_size=(2, 2))(conv1)
#        conv2 = Conv2D(32, (3, 3))(conv1)
#        conv2 = Activation('relu')(conv2)
#        conv2 = MaxPool2D(pool_size=(2, 2))(conv2)
#        conv3 = Conv2D(64, (3, 3))(conv2)
#        conv3 = Activation('relu')(conv3)
#        conv3 = MaxPool2D(pool_size=(2, 2))(conv3)
#        flat_img = Flatten()(conv3)
#        dense_img = Dense(self.img_units)(flat_img)
#        img = img_attention(dense_img, self.img_units)

        # code_img = Activation('sigmoid')(dense_img)
        # print("code_img", flat_img.shape)
        # img = RepeatVector(1)(code_img)
        # img = Reshape((1,64))(code_img)

        # print("code_img", img.shape)

        # Code Branch __________________________________________________________________________________________________________
        self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        #print("iwona:",)
        code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        code_input_f2_xyz = Input(shape=(None, ), name="CODE_TOKEN")

        #code_embedding_f1_cat = Embedding(self.hp_code_dim1_max_word_features, self.hp_code_embedding_dim_1,
        #    input_length=None
        #    )(code_input_f1_cat)
        #print(code_embedding_f1_cat.shape)


        code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
            input_length=None
            )(code_input_f2_xyz)
        #input_length=hp_code_time_steps)(code_input_f2_xyz)
        #print(code_embedding_f2_xyz.shape)
        #code_input = Concatenate()([code_embedding_f1_cat, code_embedding_f2_xyz])
        code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        code_gru = \
            Bidirectional(
                GRU(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
                name='CODE_GRU_1'
            )(code_input)

        # code_gru = \
        #    Bidirectional(
        #        GRU(units=hp_code_gru_latent_dim, return_sequences=True, dropout=config["dropout"]), name='CODE_GRU_2'
        #    )(code_gru)

        code = gru_attention_last_step(code_gru, self.hp_code_gru_latent_dim)
        print("code", code.shape)

        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug_gru = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)

        # bug_gru = \
        #    Bidirectional(
        #        GRU(hp_bug_gru_latent_dim, return_sequences=True, dropout=config["dropout"], name='BUG_GRU_2')
        #    )(bug_gru)

        bug = gru_attention_last_step(bug_gru, self.hp_bug_gru_latent_dim)

        # Combined _____________________________________________________________________________________________________________
        print("test code", code.shape)
        print("test bug", bug.shape)
        # print("test img", dense_img.shape)
        combined = Concatenate(axis=-1)([code, bug])
        print("combined", combined.shape)

        # combo = Concatenate(axis=-1)([code, bug])
        # combo = combined_attention(combo)

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)  ##############, W_regularizer=l2(0.01))(combined)
        print("binary", binary.shape)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.001), loss=config["loss"], metrics=['accuracy', tp_rate, tn_rate])
        #self.model.summary()

        #return model
        print("Done Building Model.")

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

class IMG_Model:

    def __init__(self, config_file):

        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())

        #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2"])
        #with mirrored_strategy.scope():
#        # Code Image Branch ____________________________________________________________________________________________________
#        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        code_input_img = Input(shape=(None, None, 3), name="CODE_IMG")
        ################## ?????????????????? self.img_units = config["img_units"]

        conv1 = Conv2D(32, (3, 3))(code_input_img)
        conv1 = Activation('relu')(conv1)
        conv1 = MaxPooling2D(pool_size=(2,2))(conv1)

        conv2 = Conv2D(32, (3, 3))(conv1)
        conv2 = Activation('relu')(conv2)
        conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        #conv3 = Conv2D(64, (3, 3))(conv2)
        #conv3 = Activation('relu')(conv3)
        #conv3 = MaxPooling2D(pool_size=(2,2))(conv3)

        # SPATIAL PYRAMID POOLING
        conv3 = Conv2D(32, (3, 3))(conv2)
        img = spp.SpatialPyramidPooling([1, 2, 3])(conv3)
        img.set_shape((1, 32+(32*2*2)+(32*3*3)))
        img = img_attention(img, 32 + (32 * 2 * 2) + (32 * 3 * 3))



        # code_img = Activation('sigmoid')(dense_img)
        # print("code_img", flat_img.shape)
        # img = RepeatVector(1)(code_img)
        # img = Reshape((1,64))(code_img)

        # print("code_img", img.shape)

        # Code Branch __________________________________________________________________________________________________________
        self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        #print("iwona:",)
        code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        code_input_f2_xyz = Input(shape=(None, ), name="CODE_TOKEN")

        #code_embedding_f1_cat = Embedding(self.hp_code_dim1_max_word_features, self.hp_code_embedding_dim_1,
        #    input_length=None
        #    )(code_input_f1_cat)
        #print(code_embedding_f1_cat.shape)


        code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
            input_length=None
            )(code_input_f2_xyz)
        #input_length=hp_code_time_steps)(code_input_f2_xyz)
        #print(code_embedding_f2_xyz.shape)
        #code_input = Concatenate()([code_embedding_f1_cat, code_embedding_f2_xyz])
        code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        code_gru = \
            Bidirectional(
                GRU(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
                name='CODE_GRU_1'
            )(code_input)

        # code_gru = \
        #    Bidirectional(
        #        GRU(units=hp_code_gru_latent_dim, return_sequences=True, dropout=config["dropout"]), name='CODE_GRU_2'
        #    )(code_gru)

        code = gru_attention_last_step(code_gru, self.hp_code_gru_latent_dim)
        print("code", code.shape)

        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug_gru = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)

        # bug_gru = \
        #    Bidirectional(
        #        GRU(hp_bug_gru_latent_dim, return_sequences=True, dropout=config["dropout"], name='BUG_GRU_2')
        #    )(bug_gru)

        bug = gru_attention_last_step(bug_gru, self.hp_bug_gru_latent_dim)

        # Combined _____________________________________________________________________________________________________________
        print("test code", code.shape)
        print("test bug", bug.shape)
        # print("test img", dense_img.shape)
        code.set_shape((1,64))
        bug.set_shape((1,64))
        #print("$$$$$$$$$$$$$$$$$$$$$$$$ ", K.shape(code))
        combined = Concatenate(axis=-1)([code, bug, img])
        #combined = Concatenate(axis=-1)([combined, img])
        print("combined", combined.shape)

        # combo = Concatenate(axis=-1)([code, bug])
        # combo = combined_attention(combo)

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)  ##############, W_regularizer=l2(0.01))(combined)
        print("binary", binary.shape)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz, code_input_img], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0001), loss=config["loss"], metrics=['accuracy'])  #############3, tp_rate, tn_rate])
        #self.model.summary()

        #return model
        print("Done Building Model.")

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

class IMG_Model_Attention_SPP:

    def __init__(self, config_file):

        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())


        #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2"])
        #with mirrored_strategy.scope():


        # Code Image Branch ____________________________________________________________________________________________________
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        code_input_img = Input(shape=(None, None, 3), name="CODE_IMG")

        conv = Conv2D(32, (3, 3))(code_input_img)
        conv = Activation('relu')(conv)
        conv = MaxPooling2D(pool_size=(2, 2))(conv)

        conv = Conv2D(32, (3, 3))(conv)
        conv = Activation('relu')(conv)
        conv = MaxPooling2D(pool_size=(2, 2))(conv)

        #conv = Conv2D(32, (3, 3))(conv)
        #conv = Activation('relu')(conv)
        #conv = MaxPooling2D(pool_size=(2, 2))(conv)

        #conv = Conv2D(32, (7, 7))(conv)
        #conv = Activation('relu')(conv)
        #conv = MaxPooling2D(pool_size=(4, 4))(conv)

        # SPATIAL PYRAMID POOLING
        img = Conv2D(64, (3, 3))(conv)
        #img = spp.SpatialPyramidPooling([1, 2, 3])(conv3)
        #img.set_shape((1, 32+(32*2*2)+(32*3*3)))
        print("oooooooooooo shape:", img.shape)
        img = spp.SpatialPyramidPooling([10])(img)
        print("oooooooooooo shape:", img.shape)
        img = Reshape((10*10, 64))(img)  ## ADDS the BATCH DIM, DO WE NEED IT?
        print("oooooooooooo shape:", img.shape)
        #img.set_shape((1, (32 * 8 * 8)))
        #img.set_shape(((8*8), 32))
        #print("oooooooooooo shape:", img.shape)
        #img = Permute((2, 1), input_shape=(8*8, 64))(img)
        #print("oooooooooooo shape:", img.shape)
        # dense_img = Dense(self.img_units)(flat_img)
        # XXXXXXXXXXXXXXXXXXXXXXXXXXX img = img_attention(img, 32 + (32 * 2 * 2) + (32 * 3 * 3))
        # XXXXXXXXXXXXXXXXXXXXXXXXX img = Dense(64)(img)


        # Code Branch __________________________________________________________________________________________________________
        self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        code_input_f2_xyz = Input(shape=(None, ), name="CODE_TOKEN")

        #code_embedding_f1_cat = Embedding(self.hp_code_dim1_max_word_features, self.hp_code_embedding_dim_1,
        #    input_length=None
        #    )(code_input_f1_cat)
        #print(code_embedding_f1_cat.shape)


        code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
            input_length=None
            )(code_input_f2_xyz)
        #input_length=hp_code_time_steps)(code_input_f2_xyz)
        #print(code_embedding_f2_xyz.shape)
        #code_input = Concatenate()([code_embedding_f1_cat, code_embedding_f2_xyz])
        code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        code = \
            Bidirectional(
                GRU(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
                name='CODE_GRU_1'
            )(code_input)

        # code_gru = \
        #    Bidirectional(
        #        GRU(units=hp_code_gru_latent_dim, return_sequences=True, dropout=config["dropout"]), name='CODE_GRU_2'
        #    )(code_gru)

        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXx code = gru_attention_last_step(code, self.hp_code_gru_latent_dim)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)

        # bug_gru = \
        #    Bidirectional(
        #        GRU(hp_bug_gru_latent_dim, return_sequences=True, dropout=config["dropout"], name='BUG_GRU_2')
        #    )(bug_gru)

        # XXXXXXXXXXXXXXXXX bug = gru_attention_last_step(bug, self.hp_bug_gru_latent_dim)

        # Combined _____________________________________________________________________________________________________________
        print("test code", code.shape)
        print("test bug", bug.shape)
        code = Reshape((1, 64))(code)
        bug = Reshape((1, 64))(bug)
        #code.set_shape((1, 64))
        #bug.set_shape((1, 64))


        #attCode = Dot(axes=1, normalize=True)([bug, code])
        #attCode = Flatten()(attCode)
        #attCode = Dense(64)(attCode)
        attCode = Attention()([bug, code])
        print("att code", attCode.shape, code.shape)
        sim_bug_code = Multiply()([attCode, code])
        #????????????????????????????????
        #sim_bug_code = attCode
        print("att code", sim_bug_code.shape, attCode.shape, code.shape)


        #attImg = Dot(axes=1, normalize=True)([bug, img])
        #attImg = Flatten()(attImg)
        #attImg = Dense(64)(attImg)
        img = GlobalAveragePooling1D()(img)
        attImg = Attention()([bug, img])
        print("att img", attImg.shape, img.shape)
        sim_bug_img = Multiply()([attImg, img])
        print("att img", sim_bug_img.shape, attImg.shape, img.shape)
        #sim_bug_img = GlobalAveragePooling1D()(sim_bug_img)
        #sim_bug_img = GlobalMaxPooling1D()(sim_bug_img)
        print("att img", sim_bug_img.shape)
        #sim_bug_img = Reshape((1, 64))(sim_bug_img)
        #print("att img", sim_bug_img.shape)

        print("sim_bug_code:", sim_bug_code.shape)
        print("sim_bug_img:", sim_bug_img.shape)
        combined = Add()([sim_bug_code, sim_bug_img])
        print("combined add:", combined.shape)
        print("bug:", bug.shape)
        print("combined:", combined.shape)
        combined = Concatenate(axis=-2)([bug, sim_bug_code])  # sim_bug_img])  #combined])
        print("iwona")
        combined = Flatten()(combined)
        print("xCombined", combined.shape)

        sim_bug_img = Flatten()(sim_bug_img)
        sim_bug_img = Activation('softmax')(sim_bug_img)
        combined = Concatenate()([combined, sim_bug_img])
        print("yCombined", combined.shape)

        #att = Attention()([combined, combined])
        #combined = Multiply()([att, combined])
        # combo = Concatenate(axis=-1)([code, bddddi
        # combo = combined_attention(combo)
        #combined = Reshape((132, 1))(combined)
        ##combined = Conv1D(filters=32, kernel_size=7, activation='relu')(combined)
        #combined = MaxPooling1D(pool_size=4, strides=3)(combined)
        #combined = Dropout(rate=0.1)(combined)
        #combined = Conv1D(filters=1, kernel_size=1, activation='relu')(combined)
        #combined = GlobalAveragePooling1D()(combined)
        #binary = Activation('sigmoid')(combined)


        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)  ##############, W_regularizer=l2(0.01))(combined)
        #print("binary", binary.shape)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz, code_input_img], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0001), loss=config["loss"], metrics=['accuracy'])  #############3, tp_rate, tn_rate])
        #self.model.summary()

        #return model
        print("Done Building Model.")

        # BaseModel, RMSprop=(lr=0.0001), TF.ATTENTION (better?) epoch had slightly higher accuracy but also higher loss
        # Epoch 2/8
        # 7152/7152 [==============================] - 479s 67ms/step - loss: 0.6800 - accuracy: 0.5678 - val_loss: 0.6877 - val_accuracy: 0.5829
        # Epoch 2/3
        # 7152/7152 [==============================] - 450s 63ms/step - loss: 0.6801 - accuracy: 0.5677 - val_loss: 0.6873 - val_accuracy: 0.6007
        # When I take out the MATMUL > BAD THINGS HAPPEN
        # 7152/7152 [==============================] - 487s 68ms/step - loss: 0.6899 - accuracy: 0.5306 - val_loss: 0.6934 - val_accuracy: 0.5053
        # Epoch 2/3
        # 7152/7152 [==============================] - 488s 68ms/step - loss: 0.6771 - accuracy: 0.5765 - val_loss: 0.7017 - val_accuracy: 0.4890
        # BaseModel, RMSprop=(lr=0.0001), DOT ATTENTION
        # Epoch 2/8
        # 7152/7152 [==============================] - 491s 69ms/step - loss: 0.6772 - accuracy: 0.5699 - val_loss: 0.6892 - val_accuracy: 0.5829
        # BaseModel, RMSprop=(lr=0.0001), WITHOUT ATTENTION
        # 7152/7152 [==============================] - 443s 62ms/step - loss: 0.6906 - accuracy: 0.5284 - val_loss: 0.6947 - val_accuracy: 0.4823
        # Epoch 2/3
        # 7152/7152 [==============================] - 443s 62ms/step - loss: 0.6778 - accuracy: 0.5689 - val_loss: 0.6989 - val_accuracy: 0.4971 same
        # Epoch 3/3
        # 7152/7152 [==============================] - 442s 62ms/step - loss: 0.6678 - accuracy: 0.5872 - val_loss: 0.7194 - val_accuracy: 0.4626 same


        # IMG-TEST MODEL
        # SPP=8, RMSprop=(lr=0.0001), TF.ATTENTION

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

class IMG_Model_Attention_Conv32:

    def __init__(self, config_file):

        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())


        #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2"])
        #with mirrored_strategy.scope():


        # Code Image Branch ____________________________________________________________________________________________________
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        code_input_img = Input(shape=(None, None, 3), name="CODE_IMG")

        conv = Conv2D(32, (3, 3))(code_input_img)
        conv = Activation('relu')(conv)
        conv = MaxPooling2D(pool_size=(2, 2))(conv)

        conv = Conv2D(32, (3, 3))(conv)
        conv = Activation('relu')(conv)
        conv = MaxPooling2D(pool_size=(2, 2))(conv)

        conv = Conv2D(32, (3, 3))(conv)
        conv = Activation('relu')(conv)
        img = MaxPooling2D(pool_size=(2, 2))(conv)


        # Code Branch __________________________________________________________________________________________________________
        self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        code_input_f2_xyz = Input(shape=(None, ), name="CODE_TOKEN")

        code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
            input_length=None
            )(code_input_f2_xyz)
        code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        code = \
            Bidirectional(
                GRU(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
                name='CODE_GRU_1'
            )(code_input)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)


        # Combined _____________________________________________________________________________________________________________
        print("test code", code.shape)
        print("test bug", bug.shape)
        code = Reshape((1, 64))(code)
        bug = Reshape((1, 64))(bug)

        print("*******************************************************")
        attCode = Attention()([bug, code])
        print("att code", attCode.shape, code.shape)
        sim_bug_code = Multiply()([attCode, code])
        print("att code", sim_bug_code.shape, attCode.shape, code.shape)

        print("*******************************************************")
        # CONV2 OUT: (B=1, W*, H*, F=64)
        print("img:", img.shape)
        img = Permute((3,1,2))(img)
        print("img permute:", img.shape)
        #imgx = Reshape((tf.shape(img)[1]*tf.shape(img)[2], tf.shape(img)[3]))(img)
        img = TimeDistributed(Flatten())(img)
        print("FLATTEN FILTER: ", img.shape)
        # Global Pooling: (B=1, 1, F=64)
        img = Permute((2,1))(img)
        img = GlobalAveragePooling1D()(img)
        print("GLOBAL POOLING:", img.shape)
        img = Reshape((1, 32))(img)
        print("IMG 1F, 32E:", img.shape)
        # DOUBLE UP
        img = Concatenate(axis=-1)([img, img])
        print("FINAL DOUBLED IMG 1F, 64F:", img.shape)
        # NORMALIZE?
        img = Activation('softmax')(img) ## ???????????????????? <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        attImg = Attention()([bug, img])
        print("att img", attImg.shape, img.shape)
        sim_bug_img = Multiply()([attImg, img])
        print("att img", sim_bug_img.shape, attImg.shape, img.shape)

        print("*******************************************************")
        print("sim_bug_code:", sim_bug_code.shape)
        print("sim_bug_img:", sim_bug_img.shape)
        sim_combined = Add()([sim_bug_code, sim_bug_img])
        sim_combined = Activation('softmax')(sim_combined) ############################ ????????????????????????? <<<<<<
        combined = Concatenate(axis=-2)([bug, sim_combined])  # sim_bug_img])  #combined])
        combined = Flatten()(combined)
        print("xCombined", combined.shape)
        print("*******************************************************")
        #sim_bug_img = Flatten()(sim_bug_img)
        #sim_bug_img = Activation('softmax')(sim_bug_img)
        #combined = Concatenate()([combined, sim_bug_img])
        #print("yCombined", combined.shape)

        #att = Attention()([combined, combined])
        #combined = Multiply()([att, combined])
        # combo = Concatenate(axis=-1)([code, bddddi
        # combo = combined_attention(combo)
        #combined = Reshape((132, 1))(combined)
        ##combined = Conv1D(filters=32, kernel_size=7, activation='relu')(combined)
        #combined = MaxPooling1D(pool_size=4, strides=3)(combined)
        #combined = Dropout(rate=0.1)(combined)
        #combined = Conv1D(filters=1, kernel_size=1, activation='relu')(combined)
        #combined = GlobalAveragePooling1D()(combined)
        #binary = Activation('sigmoid')(combined)


        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)  ##############, W_regularizer=l2(0.01))(combined)
        #print("binary", binary.shape)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz, code_input_img], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0001), loss=config["loss"], metrics=['accuracy'])  #############3, tp_rate, tn_rate])
        #self.model.summary()

        #return model
        print("Done Building Model.")

        # 64 Conv BaseModel, RMSprop=(lr=0.0001), TF.ATTENTION (better?) epoch had slightly higher accuracy but also higher loss
        # Epoch 2/8
        # 7152/7152 [==============================] - 479s 67ms/step - loss: 0.6800 - accuracy: 0.5678 - val_loss: 0.6877 - val_accuracy: 0.5829
        # Epoch 2/3
        # 7152/7152 [==============================] - 450s 63ms/step - loss: 0.6801 - accuracy: 0.5677 - val_loss: 0.6873 - val_accuracy: 0.6007

        # 32 DOUBLED (3-Layer)
        # 7152/7152 [==============================] - 797s 111ms/step - loss: 0.6892 - accuracy: 0.5354 - val_loss: 0.6911 - val_accuracy: 0.5499
        # Epoch 2/3
        # 7152/7152 [==============================] - 863s 121ms/step - loss: 0.6788 - accuracy: 0.5666 - val_loss: 0.6923 - val_accuracy: 0.5542
        # Epoch 3/3
        # 7152/7152 [==============================] - 936s 131ms/step - loss: 0.6683 - accuracy: 0.5940 - val_loss: 0.6992 - val_accuracy: 0.5192

        # 32 DOUBLED (4-Layer)
        # Error



        #######################################################################################################################################
        #######################################################################################################################################
        # IMG-TEST MODEL
        # Sigmoid x 2 (GlobalMax, sim_combined)
        # RMSprop(lr=0.0001)
        # [bug, sim_combined]
        # 7152/7152 [==============================] - 805s 113ms/step - loss: 0.6902 - accuracy: 0.5306 - val_loss: 0.6885 - val_accuracy: 0.5489
        # Epoch 2/3
        # 7152/7152 [==============================] - 853s 119ms/step - loss: 0.6804 - accuracy: 0.5687 - val_loss: 0.6853 - val_accuracy: 0.5690
        # Epoch 3/3
        # 7152/7152 [==============================] - 909s 127ms/step - loss: 0.6711 - accuracy: 0.5854 - val_loss: 0.6909 - val_accuracy: 0.5762

        # try last Conv2 to be 32 and double it instead
        # try changing
        # try combined = [bug, sim_combined]

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

class IMG_Model_Attention:

    def __init__(self, config_file):

        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())


        #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2"])
        #with mirrored_strategy.scope():


        # Code Image Branch ____________________________________________________________________________________________________
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        code_input_img = Input(shape=(None, None, 3), name="CODE_IMG")

        conv1 = Conv2D(32, (3, 3))(code_input_img)
        conv1 = Activation('relu')(conv1)
        img1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3))(img1)
        conv2 = Activation('relu')(conv2)
        img2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        img2 = img_features(img2)

        conv23 = Conv2D(32, (3, 3))(img1)
        conv23 = Activation('relu')(conv23)
        img23 = MaxPooling2D(pool_size=(2, 2))(conv23)

        conv3 = Conv2D(64, (3, 3))(img23)
        conv3 = Activation('relu')(conv3)
        img3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        img3 = img_features(img3)

        #conv4 = Conv2D(64, (3, 3))(img3)
        #conv4 = Activation('relu')(conv4)
        #img4 = MaxPooling2D(pool_size=(2, 2))(conv4)


        # Code Branch __________________________________________________________________________________________________________
        self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        code_input_f2_xyz = Input(shape=(None, ), name="CODE_TOKEN")

        #code_embedding_f1_cat = Embedding(self.hp_code_dim1_max_word_features, self.hp_code_embedding_dim_1,
        #    input_length=None
        #    )(code_input_f1_cat)
        #print(code_embedding_f1_cat.shape)


        code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
            input_length=None
            )(code_input_f2_xyz)
        #input_length=hp_code_time_steps)(code_input_f2_xyz)
        #print(code_embedding_f2_xyz.shape)
        #code_input = Concatenate()([code_embedding_f1_cat, code_embedding_f2_xyz])
        code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        code = \
            Bidirectional(
                GRU(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
                name='CODE_GRU_1'
            )(code_input)

        # code_gru = \
        #    Bidirectional(
        #        GRU(units=hp_code_gru_latent_dim, return_sequences=True, dropout=config["dropout"]), name='CODE_GRU_2'
        #    )(code_gru)

        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXx code = gru_attention_last_step(code, self.hp_code_gru_latent_dim)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)

        # bug_gru = \
        #    Bidirectional(
        #        GRU(hp_bug_gru_latent_dim, return_sequences=True, dropout=config["dropout"], name='BUG_GRU_2')
        #    )(bug_gru)

        # XXXXXXXXXXXXXXXXX bug = gru_attention_last_step(bug, self.hp_bug_gru_latent_dim)

        # Combined _____________________________________________________________________________________________________________
        print("test code", code.shape)
        print("test bug", bug.shape)
        code = Reshape((1, 64))(code)
        bug = Reshape((1, 64))(bug)

        print("NORM BEFORE: *******************************************************")
        ##bug = Activation('softmax')(bug)
        #code = Activation('softmax')(code)
        #img2 = Activation('softmax')(img2)
        #img3 = Activation('softmax')(img3)

        print("CODE: *******************************************************")
        #attCode = Attention()([bug, code])
        attCode = attention_distribution(bug, code)
        print("att code", attCode.shape, code.shape)
        sim_bug_code = Multiply()([attCode, code])
        print("att code", sim_bug_code.shape, attCode.shape, code.shape)

        print("IMG 2: *******************************************************")
        #attImg2 = Attention()([bug, img2])
        attImg2 = attention_distribution(bug, img2)
        print("att img", attImg2.shape, img2.shape)
        sim_bug_img2 = Multiply()([attImg2, img2])
        print("att img", sim_bug_img2.shape, attImg2.shape, img2.shape)
        print("sim_bug_code:", sim_bug_code.shape)
        print("sim_bug_img:", sim_bug_img2.shape)

        print("IMG 3: *******************************************************")
        #attImg3 = Attention()([bug, img3])
        attImg3 = attention_distribution(bug, img3)
        print("att img", attImg3.shape, img3.shape)
        sim_bug_img3 = Multiply()([attImg3, img3])
        print("att img", sim_bug_img3.shape, attImg3.shape, img3.shape)
        print("sim_bug_code:", sim_bug_code.shape)
        print("sim_bug_img:", sim_bug_img3.shape)

        print("NORM AFTER: *******************************************************")
        sim_bug_code = Activation('softmax')(sim_bug_code)
        sim_bug_img2 = Activation('softmax')(sim_bug_img2)
        sim_bug_img3 = Activation('softmax')(sim_bug_img3)

        # loss: 0.6839 - accuracy: 0.5526 - val_loss: 0.6933 - val_accuracy: 0.5058
        #sim_img_combined = Add()([sim_bug_code, sim_bug_code, sim_bug_img2, sim_bug_img3])
        # loss: 0.6838 - accuracy: 0.5475 - val_loss: 0.6823 - val_accuracy: 0.6002 > RMSprop(lr=0.0005) norm img2
        # loss: 0.6827 - accuracy: 0.5551 - val_loss: 0.6944 - val_accuracy: 0.5225
        # loss: 0.6847 - accuracy: 0.5501 - val_loss: 0.6854 - val_accuracy: 0.5896
        # code = Activation('softmax')(code) ## NORMALIZE?
        # loss: 0.6849 - accuracy: 0.5506 - val_loss: 0.6841 - val_accuracy: 0.6141 > RMSprop(lr=0.0005) normB img2+code
        # loss: 0.6837 - accuracy: 0.5520 - val_loss: 0.6892 - val_accuracy: 0.5772
        # loss: 0.6831 - accuracy: 0.5491 - val_loss: 0.6913 - val_accuracy: 0.5695
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # BASE MODEL + codeNorm
        # loss: 0.6843 - accuracy: 0.5556 - val_loss: 0.6830 - val_accuracy: 0.5757 58 > RMSprop(lr=0.0005) normB code
        # loss: 0.6831 - accuracy: 0.5485 - val_loss: 0.6997 - val_accuracy: 0.5575 56
        # loss: 0.6848 - accuracy: 0.5474 - val_loss: 0.6802 - val_accuracy: 0.5820 58
        # loss: 0.6839 - accuracy: 0.5572 - val_loss: 0.7025 - val_accuracy: 0.5058 e1
        # loss: 0.6649 - accuracy: 0.5976 - val_loss: 0.7109 - val_accuracy: 0.5561 e2
        # loss: 0.6842 - accuracy: 0.5522 - val_loss: 0.6872 - val_accuracy: 0.5537
        # BASE MODEL Norm Code After
        # loss: 0.6832 - accuracy: 0.5548 - val_loss: 0.6833 - val_accuracy: 0.5733
        # loss: 0.6837 - accuracy: 0.5531 - val_loss: 0.6852 - val_accuracy: 0.5829
        # loss: 0.6835 - accuracy: 0.5501 - val_loss: 0.6962 - val_accuracy: 0.5038
        # loss: 0.6827 - accuracy: 0.5636 - val_loss: 0.6912 - val_accuracy: 0.5666
        # loss: 0.6828 - accuracy: 0.5591 - val_loss: 0.7059 - val_accuracy: 0.5000
        # loss: 0.6843 - accuracy: 0.5474 - val_loss: 0.6924 - val_accuracy: 0.5590
        # loss: 0.6842 - accuracy: 0.5503 - val_loss: 0.6844 - val_accuracy: 0.5978
        # loss: 0.6841 - accuracy: 0.5487 - val_loss: 0.6894 - val_accuracy: 0.5935
        # loss: 0.6835 - accuracy: 0.5584 - val_loss: 0.6964 - val_accuracy: 0.5163
        # loss: 0.6843 - accuracy: 0.5536 - val_loss: 0.6794 - val_accuracy: 0.5940
        # loss: 0.6838 - accuracy: 0.5554 - val_loss: 0.6835 - val_accuracy: 0.5729
        #
        # BASE MODEL + codeNorm + bugNorm
        # loss: 0.6924 - accuracy: 0.5203 - val_loss: 0.6917 - val_accuracy: 0.5594  56 > RMSprop(lr=0.0005) normB bug+code
        # loss: 0.6927 - accuracy: 0.5110 - val_loss: 0.6909 - val_accuracy: 0.5796  58
        # loss: 0.6926 - accuracy: 0.5154 - val_loss: 0.6915 - val_accuracy: 0.5479  55
        # loss: 0.6924 - accuracy: 0.5136 - val_loss: 0.6902 - val_accuracy: 0.5647  56
        # BASE MODLE without normalization
        # loss: 0.6844 - accuracy: 0.5362 - val_loss: 0.6811 - val_accuracy: 0.5791 58 > RMSprop(lr=0.0005) no norm
        # loss: 0.6854 - accuracy: 0.5492 - val_loss: 0.6905 - val_accuracy: 0.5494 55 > e1
        # loss: 0.6637 - accuracy: 0.6008 - val_loss: 0.7975 - val_accuracy: 0.4818 > e2
        # loss: 0.6826 - accuracy: 0.5513 - val_loss: 0.6956 - val_accuracy: 0.5638 56 > e1
        # loss: 0.6633 - accuracy: 0.5944 - val_loss: 0.7431 - val_accuracy: 0.4813 > e2
        sim_img_combined = Add()([sim_bug_code, sim_bug_img2])

        sim_img_combined = Activation('softmax')(sim_img_combined)

        combined = Concatenate(axis=-2)([bug, sim_bug_code, sim_img_combined])  #, sim_combined3])  #
        combined = Flatten()(combined)
        # >>>>>>>>>>>> combined = Dropout(rate=.5)(combined)
        print("All Combined:", combined.shape)


        #binary2 = Dense(64, activation='sigmoid', name="binary_2")(combined2)
        #binary3 = Dense(64, activation='sigmoid', name="binary_3")(combined3)
        #combined = Concatenate()([binary2, binary3])
        #combined = Activation('softmax')(combined)

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)  ##############, W_regularizer=l2(0.01))(combined)
        #print("binary", binary.shape)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz, code_input_img], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=config["loss"], metrics=['accuracy'])  #############3, tp_rate, tn_rate])
        #self.model.summary()

        #return model
        print("Done Building Model.")

        # BaseModel, RMSprop=(lr=0.0001), TF.ATTENTION (better?) epoch had slightly higher accuracy but also higher loss
        # Epoch 2/8
        # 7152/7152 [==============================] - 479s 67ms/step - loss: 0.6800 - accuracy: 0.5678 - val_loss: 0.6877 - val_accuracy: 0.5829
        # Epoch 2/3
        # 7152/7152 [==============================] - 450s 63ms/step - loss: 0.6801 - accuracy: 0.5677 - val_loss: 0.6873 - val_accuracy: 0.6007



        #######################################################################################################################################
        #######################################################################################################################################
        # IMG-TEST MODEL
        # Sigmoid x 2 (GlobalMax, sim_combined)
        # RMSprop(lr=0.0001)
        # Conv2 x 3 (32, 32, 64)
        # 7152/7152 [==============================] - 725s 101ms/step - loss: 0.6893 - accuracy: 0.5443 - val_loss: 0.6882 - val_accuracy: 0.5748
        # Epoch 2/3
        # 7152/7152 [==============================] - 806s 113ms/step - loss: 0.6794 - accuracy: 0.5713 - val_loss: 0.6862 - val_accuracy: 0.5676
        # Epoch 3/3
        # 7152/7152 [==============================] - 863s 121ms/step - loss: 0.6721 - accuracy: 0.5853 - val_loss: 0.6866 - val_accuracy: 0.5695

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # try not using sigmoid (BAD)
        # 7152/7152 [==============================] - 760s 106ms/step - loss: 7.7125 - accuracy: 0.5000 - val_loss: 7.7125 - val_accuracy: 0.5000
        # Epoch 2/3
        # 7152/7152 [==============================] - 872s 122ms/step - loss: 7.7125 - accuracy: 0.5000 - val_loss: 7.7125 - val_accuracy: 0.5000
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # One Sigmoid on img
        # 7152/7152 [==============================] - 721s 101ms/step - loss: 0.6907 - accuracy: 0.5224 - val_loss: 0.6890 - val_accuracy: 0.5393
        # Epoch 2/3
        # 7152/7152 [==============================] - 801s 112ms/step - loss: 0.6807 - accuracy: 0.5705 - val_loss: 0.6862 - val_accuracy: 0.5772
        # Epoch 3/3
        # 7152/7152 [==============================] - 882s 123ms/step - loss: 0.6693 - accuracy: 0.5899 - val_loss: 0.6911 - val_accuracy: 0.5700
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # One Sigmoid on sim_combined
        # 7152/7152 [==============================] - 733s 102ms/step - loss: 0.6891 - accuracy: 0.5394 - val_loss: 0.6900 - val_accuracy: 0.5705
        # Epoch 2/3
        # 7152/7152 [==============================] - 934s 131ms/step - loss: 0.6789 - accuracy: 0.5702 - val_loss: 0.6899 - val_accuracy: 0.5623
        # Epoch 3/3
        # 7152/7152 [==============================] - 1037s 145ms/step - loss: 0.6694 - accuracy: 0.5893 - val_loss: 0.6991 - val_accuracy: 0.5614

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # IMG-TEST MODEL
        # Sigmoid x 2 (GlobalAVG, sim_combined)
        # RMSprop(lr=0.0001)
        # Conv2 x 2 (32, 64)
        # [bug, sim_bug_code, sim_combined]
        # 7152/7152 [==============================] - 779s 109ms/step - loss: 0.6897 - accuracy: 0.5320 - val_loss: 0.6881 - val_accuracy: 0.5623
        # Epoch 2/3
        # 7152/7152 [==============================] - 832s 116ms/step - loss: 0.6807 - accuracy: 0.5762 - val_loss: 0.6860 - val_accuracy: 0.5858
        # Epoch 3/3
        # 7152/7152 [==============================] - 936s 131ms/step - loss: 0.6700 - accuracy: 0.5874 - val_loss: 0.6896 - val_accuracy: 0.5244

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # IMG-TEST MODEL
        # Sigmoid x 2 (GlobalAVG, sim_combined)
        # RMSprop(lr=0.0001)
        # Conv2 x 2 (32, 64)
        # [bug, sim_bug_code, sim_bug_img, sim_combined] >> NORMALIZED
        # 7152/7152 [==============================] - 526s 74ms/step - loss: 0.6932 - accuracy: 0.4966 - val_loss: 0.6930 - val_accuracy: 0.5000
        # Epoch 2/3
        # 7152/7152 [==============================] - 583s 82ms/step - loss: 0.6929 - accuracy: 0.5159 - val_loss: 0.6928 - val_accuracy: 0.5585
        # Epoch 3/3
        # 7152/7152 [==============================] - 559s 78ms/step - loss: 0.6920 - accuracy: 0.5438 - val_loss: 0.6902 - val_accuracy: 0.5738




        # L1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # IMG-TEST MODEL
        # Sigmoid x 2 (GlobalAVG, sim_combined)
        # RMSprop(lr=0.0001)
        # Conv2 x 2 (32, 64)
        # [bug, sim_combined] [x, normalized]
        #

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # IMG-TEST MODEL
        # Sigmoid x 2 (GlobalAVG, sim_combined)
        # RMSprop(lr=0.0001)
        # Conv2 x 2 (64, 64)
        # [bug, sim_bug_code, bug, sim_combined]
        # 7152/7152 [==============================] - 710s 99ms/step - loss: 0.6895 - accuracy: 0.5310 - val_loss: 0.6857 - val_accuracy: 0.5748
        # Epoch 2/3
        # 7152/7152 [==============================] - 761s 106ms/step - loss: 0.6789 - accuracy: 0.5687 - val_loss: 0.6842 - val_accuracy: 0.5868
        # Epoch 3/3
        # 7152/7152 [==============================] - 821s 115ms/step - loss: 0.6689 - accuracy: 0.5824 - val_loss: 0.6958 - val_accuracy: 0.5134



        # CRASHED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # IMG-TEST MODEL
        # Sigmoid x 2 (GlobalAVG, sim_combined)
        # RMSprop(lr=0.0001)
        # Conv2 x 2 (32, 64)
        # [bug, sim_combined] >> NORMALIZED
        # CRASH

        # CRASHED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # IMG-TEST MODEL
        # Sigmoid x 2 (GlobalAVG, sim_combined)
        # RMSprop(lr=0.0001)
        # Conv2 x 2 (32, 64)
        # [sim_combined] >> NORMALIZED
        # CRASH




        # try sigmoid on bug...
        # try last Conv2 to be 32 and double it instead
        # try Conv2 changing levels
        # try changing Filter size for non-last Conv2
        # GlobalAverage vs Max
        # try combined = [bug, sim_combined]
        # LR, Optimizer
        # try Dense, Dropout


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

class IMG_Model_Attention_After:

    def __init__(self, config_file):

        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())


        #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2"])
        #with mirrored_strategy.scope():


        # Code Image Branch ____________________________________________________________________________________________________
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        code_input_img = Input(shape=(None, None, 3), name="CODE_IMG")

        conv1 = Conv2D(32, (3, 3))(code_input_img)
        conv1 = Activation('relu')(conv1)
        img1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3))(img1)
        conv2 = Activation('relu')(conv2)
        img2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        img2 = img_features(img2)

        conv23 = Conv2D(32, (3, 3))(img1)
        conv23 = Activation('relu')(conv23)
        img23 = MaxPooling2D(pool_size=(2, 2))(conv23)

        conv3 = Conv2D(64, (3, 3))(img23)
        conv3 = Activation('relu')(conv3)
        img3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        img3 = img_features(img3)

        #conv4 = Conv2D(64, (3, 3))(img3)
        #conv4 = Activation('relu')(conv4)
        #img4 = MaxPooling2D(pool_size=(2, 2))(conv4)


        # Code Branch __________________________________________________________________________________________________________
        self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        code_input_f2_xyz = Input(shape=(None, ), name="CODE_TOKEN")

        #code_embedding_f1_cat = Embedding(self.hp_code_dim1_max_word_features, self.hp_code_embedding_dim_1,
        #    input_length=None
        #    )(code_input_f1_cat)
        #print(code_embedding_f1_cat.shape)


        code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
            input_length=None
            )(code_input_f2_xyz)
        #input_length=hp_code_time_steps)(code_input_f2_xyz)
        #print(code_embedding_f2_xyz.shape)
        #code_input = Concatenate()([code_embedding_f1_cat, code_embedding_f2_xyz])
        code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        code = \
            Bidirectional(
                GRU(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
                name='CODE_GRU_1'
            )(code_input)

        # code_gru = \
        #    Bidirectional(
        #        GRU(units=hp_code_gru_latent_dim, return_sequences=True, dropout=config["dropout"]), name='CODE_GRU_2'
        #    )(code_gru)

        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXx code = gru_attention_last_step(code, self.hp_code_gru_latent_dim)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)

        # bug_gru = \
        #    Bidirectional(
        #        GRU(hp_bug_gru_latent_dim, return_sequences=True, dropout=config["dropout"], name='BUG_GRU_2')
        #    )(bug_gru)

        # XXXXXXXXXXXXXXXXX bug = gru_attention_last_step(bug, self.hp_bug_gru_latent_dim)

        # Combined _____________________________________________________________________________________________________________
        print("test code", code.shape)
        print("test bug", bug.shape)
        code = Reshape((1, 64))(code)
        bug = Reshape((1, 64))(bug)

        print("NORM BEFORE: *******************************************************")
        ##bug = Activation('softmax')(bug)
        #code = Activation('softmax')(code)
        #img2 = Activation('softmax')(img2)
        #img3 = Activation('softmax')(img3)

        print("CODE: *******************************************************")
        #attCode = Attention()([bug, code])
        attCode = attention_distribution(bug, code)
        print("att code", attCode.shape, code.shape)
        sim_bug_code = Multiply()([attCode, code])
        print("att code", sim_bug_code.shape, attCode.shape, code.shape)

        print("IMG 2: *******************************************************")
        #attImg2 = Attention()([bug, img2])
        attImg2 = attention_distribution(bug, img2)
        print("att img", attImg2.shape, img2.shape)
        sim_bug_img2 = Multiply()([attImg2, img2])
        print("att img", sim_bug_img2.shape, attImg2.shape, img2.shape)
        print("sim_bug_code:", sim_bug_code.shape)
        print("sim_bug_img:", sim_bug_img2.shape)

        print("IMG 3: *******************************************************")
        #attImg3 = Attention()([bug, img3])
        attImg3 = attention_distribution(bug, img3)
        print("att img", attImg3.shape, img3.shape)
        sim_bug_img3 = Multiply()([attImg3, img3])
        print("att img", sim_bug_img3.shape, attImg3.shape, img3.shape)
        print("sim_bug_code:", sim_bug_code.shape)
        print("sim_bug_img:", sim_bug_img3.shape)

        print("NORM AFTER: *******************************************************")
        sim_bug_code = Activation('softmax')(sim_bug_code)
        sim_bug_img2 = Activation('softmax')(sim_bug_img2)
        sim_bug_img3 = Activation('softmax')(sim_bug_img3)

        # loss: 0.6839 - accuracy: 0.5526 - val_loss: 0.6933 - val_accuracy: 0.5058
        #sim_img_combined = Add()([sim_bug_code, sim_bug_code, sim_bug_img2, sim_bug_img3])
        # loss: 0.6838 - accuracy: 0.5475 - val_loss: 0.6823 - val_accuracy: 0.6002 > RMSprop(lr=0.0005) norm img2
        # loss: 0.6827 - accuracy: 0.5551 - val_loss: 0.6944 - val_accuracy: 0.5225
        # loss: 0.6847 - accuracy: 0.5501 - val_loss: 0.6854 - val_accuracy: 0.5896
        # code = Activation('softmax')(code) ## NORMALIZE?
        # loss: 0.6849 - accuracy: 0.5506 - val_loss: 0.6841 - val_accuracy: 0.6141 > RMSprop(lr=0.0005) normB img2+code
        # loss: 0.6837 - accuracy: 0.5520 - val_loss: 0.6892 - val_accuracy: 0.5772
        # loss: 0.6831 - accuracy: 0.5491 - val_loss: 0.6913 - val_accuracy: 0.5695
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # BASE MODEL + codeNorm
        # loss: 0.6843 - accuracy: 0.5556 - val_loss: 0.6830 - val_accuracy: 0.5757 58 > RMSprop(lr=0.0005) normB code
        # loss: 0.6831 - accuracy: 0.5485 - val_loss: 0.6997 - val_accuracy: 0.5575 56
        # loss: 0.6848 - accuracy: 0.5474 - val_loss: 0.6802 - val_accuracy: 0.5820 58
        # loss: 0.6839 - accuracy: 0.5572 - val_loss: 0.7025 - val_accuracy: 0.5058 e1
        # loss: 0.6649 - accuracy: 0.5976 - val_loss: 0.7109 - val_accuracy: 0.5561 e2
        # loss: 0.6842 - accuracy: 0.5522 - val_loss: 0.6872 - val_accuracy: 0.5537
        # BASE MODEL Norm Code After
        # loss: 0.6832 - accuracy: 0.5548 - val_loss: 0.6833 - val_accuracy: 0.5733
        # loss: 0.6837 - accuracy: 0.5531 - val_loss: 0.6852 - val_accuracy: 0.5829
        # loss: 0.6835 - accuracy: 0.5501 - val_loss: 0.6962 - val_accuracy: 0.5038
        # loss: 0.6827 - accuracy: 0.5636 - val_loss: 0.6912 - val_accuracy: 0.5666
        # loss: 0.6828 - accuracy: 0.5591 - val_loss: 0.7059 - val_accuracy: 0.5000
        # loss: 0.6843 - accuracy: 0.5474 - val_loss: 0.6924 - val_accuracy: 0.5590
        # loss: 0.6842 - accuracy: 0.5503 - val_loss: 0.6844 - val_accuracy: 0.5978
        # loss: 0.6841 - accuracy: 0.5487 - val_loss: 0.6894 - val_accuracy: 0.5935
        # loss: 0.6835 - accuracy: 0.5584 - val_loss: 0.6964 - val_accuracy: 0.5163
        # loss: 0.6843 - accuracy: 0.5536 - val_loss: 0.6794 - val_accuracy: 0.5940
        # loss: 0.6838 - accuracy: 0.5554 - val_loss: 0.6835 - val_accuracy: 0.5729
        #
        # BASE MODEL + codeNorm + bugNorm
        # loss: 0.6924 - accuracy: 0.5203 - val_loss: 0.6917 - val_accuracy: 0.5594  56 > RMSprop(lr=0.0005) normB bug+code
        # loss: 0.6927 - accuracy: 0.5110 - val_loss: 0.6909 - val_accuracy: 0.5796  58
        # loss: 0.6926 - accuracy: 0.5154 - val_loss: 0.6915 - val_accuracy: 0.5479  55
        # loss: 0.6924 - accuracy: 0.5136 - val_loss: 0.6902 - val_accuracy: 0.5647  56
        # BASE MODLE without normalization
        # loss: 0.6844 - accuracy: 0.5362 - val_loss: 0.6811 - val_accuracy: 0.5791 58 > RMSprop(lr=0.0005) no norm
        # loss: 0.6854 - accuracy: 0.5492 - val_loss: 0.6905 - val_accuracy: 0.5494 55 > e1
        # loss: 0.6637 - accuracy: 0.6008 - val_loss: 0.7975 - val_accuracy: 0.4818 > e2
        # loss: 0.6826 - accuracy: 0.5513 - val_loss: 0.6956 - val_accuracy: 0.5638 56 > e1
        # loss: 0.6633 - accuracy: 0.5944 - val_loss: 0.7431 - val_accuracy: 0.4813 > e2
        sim_img_combined = Add()([sim_bug_code, sim_bug_img2])

        sim_img_combined = Activation('softmax')(sim_img_combined)

        combined = Concatenate(axis=-2)([bug, sim_bug_code])  #, sim_img_combined])  #, sim_combined3])  #
        combined = Flatten()(combined)
        # >>>>>>>>>>>> combined = Dropout(rate=.5)(combined)
        print("All Combined:", combined.shape)


        #binary2 = Dense(64, activation='sigmoid', name="binary_2")(combined2)
        #binary3 = Dense(64, activation='sigmoid', name="binary_3")(combined3)
        #combined = Concatenate()([binary2, binary3])
        #combined = Activation('softmax')(combined)

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)  ##############, W_regularizer=l2(0.01))(combined)
        #print("binary", binary.shape)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz, code_input_img], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=config["loss"], metrics=['accuracy'])  #############3, tp_rate, tn_rate])
        #self.model.summary()

        #return model
        print("Done Building Model.")

        # BaseModel, RMSprop=(lr=0.0001), TF.ATTENTION (better?) epoch had slightly higher accuracy but also higher loss
        # Epoch 2/8
        # 7152/7152 [==============================] - 479s 67ms/step - loss: 0.6800 - accuracy: 0.5678 - val_loss: 0.6877 - val_accuracy: 0.5829
        # Epoch 2/3
        # 7152/7152 [==============================] - 450s 63ms/step - loss: 0.6801 - accuracy: 0.5677 - val_loss: 0.6873 - val_accuracy: 0.6007



        #######################################################################################################################################
        #######################################################################################################################################
        # IMG-TEST MODEL
        # Sigmoid x 2 (GlobalMax, sim_combined)
        # RMSprop(lr=0.0001)
        # Conv2 x 3 (32, 32, 64)
        # 7152/7152 [==============================] - 725s 101ms/step - loss: 0.6893 - accuracy: 0.5443 - val_loss: 0.6882 - val_accuracy: 0.5748
        # Epoch 2/3
        # 7152/7152 [==============================] - 806s 113ms/step - loss: 0.6794 - accuracy: 0.5713 - val_loss: 0.6862 - val_accuracy: 0.5676
        # Epoch 3/3
        # 7152/7152 [==============================] - 863s 121ms/step - loss: 0.6721 - accuracy: 0.5853 - val_loss: 0.6866 - val_accuracy: 0.5695

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # try not using sigmoid (BAD)
        # 7152/7152 [==============================] - 760s 106ms/step - loss: 7.7125 - accuracy: 0.5000 - val_loss: 7.7125 - val_accuracy: 0.5000
        # Epoch 2/3
        # 7152/7152 [==============================] - 872s 122ms/step - loss: 7.7125 - accuracy: 0.5000 - val_loss: 7.7125 - val_accuracy: 0.5000
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # One Sigmoid on img
        # 7152/7152 [==============================] - 721s 101ms/step - loss: 0.6907 - accuracy: 0.5224 - val_loss: 0.6890 - val_accuracy: 0.5393
        # Epoch 2/3
        # 7152/7152 [==============================] - 801s 112ms/step - loss: 0.6807 - accuracy: 0.5705 - val_loss: 0.6862 - val_accuracy: 0.5772
        # Epoch 3/3
        # 7152/7152 [==============================] - 882s 123ms/step - loss: 0.6693 - accuracy: 0.5899 - val_loss: 0.6911 - val_accuracy: 0.5700
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # One Sigmoid on sim_combined
        # 7152/7152 [==============================] - 733s 102ms/step - loss: 0.6891 - accuracy: 0.5394 - val_loss: 0.6900 - val_accuracy: 0.5705
        # Epoch 2/3
        # 7152/7152 [==============================] - 934s 131ms/step - loss: 0.6789 - accuracy: 0.5702 - val_loss: 0.6899 - val_accuracy: 0.5623
        # Epoch 3/3
        # 7152/7152 [==============================] - 1037s 145ms/step - loss: 0.6694 - accuracy: 0.5893 - val_loss: 0.6991 - val_accuracy: 0.5614

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # IMG-TEST MODEL
        # Sigmoid x 2 (GlobalAVG, sim_combined)
        # RMSprop(lr=0.0001)
        # Conv2 x 2 (32, 64)
        # [bug, sim_bug_code, sim_combined]
        # 7152/7152 [==============================] - 779s 109ms/step - loss: 0.6897 - accuracy: 0.5320 - val_loss: 0.6881 - val_accuracy: 0.5623
        # Epoch 2/3
        # 7152/7152 [==============================] - 832s 116ms/step - loss: 0.6807 - accuracy: 0.5762 - val_loss: 0.6860 - val_accuracy: 0.5858
        # Epoch 3/3
        # 7152/7152 [==============================] - 936s 131ms/step - loss: 0.6700 - accuracy: 0.5874 - val_loss: 0.6896 - val_accuracy: 0.5244

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # IMG-TEST MODEL
        # Sigmoid x 2 (GlobalAVG, sim_combined)
        # RMSprop(lr=0.0001)
        # Conv2 x 2 (32, 64)
        # [bug, sim_bug_code, sim_bug_img, sim_combined] >> NORMALIZED
        # 7152/7152 [==============================] - 526s 74ms/step - loss: 0.6932 - accuracy: 0.4966 - val_loss: 0.6930 - val_accuracy: 0.5000
        # Epoch 2/3
        # 7152/7152 [==============================] - 583s 82ms/step - loss: 0.6929 - accuracy: 0.5159 - val_loss: 0.6928 - val_accuracy: 0.5585
        # Epoch 3/3
        # 7152/7152 [==============================] - 559s 78ms/step - loss: 0.6920 - accuracy: 0.5438 - val_loss: 0.6902 - val_accuracy: 0.5738




        # L1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # IMG-TEST MODEL
        # Sigmoid x 2 (GlobalAVG, sim_combined)
        # RMSprop(lr=0.0001)
        # Conv2 x 2 (32, 64)
        # [bug, sim_combined] [x, normalized]
        #

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # IMG-TEST MODEL
        # Sigmoid x 2 (GlobalAVG, sim_combined)
        # RMSprop(lr=0.0001)
        # Conv2 x 2 (64, 64)
        # [bug, sim_bug_code, bug, sim_combined]
        # 7152/7152 [==============================] - 710s 99ms/step - loss: 0.6895 - accuracy: 0.5310 - val_loss: 0.6857 - val_accuracy: 0.5748
        # Epoch 2/3
        # 7152/7152 [==============================] - 761s 106ms/step - loss: 0.6789 - accuracy: 0.5687 - val_loss: 0.6842 - val_accuracy: 0.5868
        # Epoch 3/3
        # 7152/7152 [==============================] - 821s 115ms/step - loss: 0.6689 - accuracy: 0.5824 - val_loss: 0.6958 - val_accuracy: 0.5134



        # CRASHED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # IMG-TEST MODEL
        # Sigmoid x 2 (GlobalAVG, sim_combined)
        # RMSprop(lr=0.0001)
        # Conv2 x 2 (32, 64)
        # [bug, sim_combined] >> NORMALIZED
        # CRASH

        # CRASHED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # IMG-TEST MODEL
        # Sigmoid x 2 (GlobalAVG, sim_combined)
        # RMSprop(lr=0.0001)
        # Conv2 x 2 (32, 64)
        # [sim_combined] >> NORMALIZED
        # CRASH




        # try sigmoid on bug...
        # try last Conv2 to be 32 and double it instead
        # try Conv2 changing levels
        # try changing Filter size for non-last Conv2
        # GlobalAverage vs Max
        # try combined = [bug, sim_combined]
        # LR, Optimizer
        # try Dense, Dropout


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

class Base_Model_Attention_Before:

    def __init__(self, configs):
        choose = Feed_Type()
        self.feed_flag = choose.feed_CAR
        self.config = configs
        #if not os.path.exists(config_file):
        #    print("Error, Experiment config file: " + config_file + " does not exist.")
        #    sys.exit(1)
        #with open(self.config_file) as f:
        #    config = json.loads(f.read())


        #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2"])
        #with mirrored_strategy.scope():


        # Code Image Branch ____________________________________________________________________________________________________
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        code_input_img = Input(shape=(None, None, 3), name="CODE_IMG")

        conv1 = Conv2D(32, (3, 3))(code_input_img)
        conv1 = Activation('relu')(conv1)
        img1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3))(img1)
        conv2 = Activation('relu')(conv2)
        img2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        img2 = img_features(img2)

        conv23 = Conv2D(32, (3, 3))(img1)
        conv23 = Activation('relu')(conv23)
        img23 = MaxPooling2D(pool_size=(2, 2))(conv23)

        conv3 = Conv2D(64, (3, 3))(img23)
        conv3 = Activation('relu')(conv3)
        img3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        img3 = img_features(img3)


        # Code Branch __________________________________________________________________________________________________________
        self.hp_code_dim1_max_word_features = int(self.config['code_dim1_max_word_features'])
        self.hp_code_dim2_max_word_features = int(self.config['code_dim2_max_word_features'])
        self.hp_code_embedding_dim_1 = int(self.config["code_embedding_d1_cat"])
        self.hp_code_embedding_dim_2 = int(self.config["code_embedding_d2_token"])
        self.hp_code_gru_latent_dim = int(self.config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        code_input_f2_xyz = Input(shape=(None, ), name="CODE_TOKEN")



        code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
            input_length=None
            )(code_input_f2_xyz)

        code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        code = \
            Bidirectional(
                GRU(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=self.config["dropout"]),
                name='CODE_GRU_1'
            )(code_input)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(self.config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(self.config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=self.config["dropout"], name='BUG_GRU_1')
            )(bug_input)


        # Combined _____________________________________________________________________________________________________________
        print("test code", code.shape)
        print("test bug", bug.shape)
        code = Reshape((1, 64))(code)
        bug = Reshape((1, 64))(bug)

        print("NORM BEFORE: *******************************************************")
        ##bug = Activation('softmax')(bug)
        code = Activation('softmax')(code)
        img2 = Activation('softmax')(img2)
        img3 = Activation('softmax')(img3)

        print("CODE: *******************************************************")
        #attCode = Attention()([bug, code])
        attCode = attention_distribution(bug, code)
        print("att code", attCode.shape, code.shape)
        sim_bug_code = Multiply()([attCode, code])
        print("att code", sim_bug_code.shape, attCode.shape, code.shape)

        print("IMG 2: *******************************************************")
        #attImg2 = Attention()([bug, img2])
        attImg2 = attention_distribution(bug, img2)
        print("att img", attImg2.shape, img2.shape)
        sim_bug_img2 = Multiply()([attImg2, img2])
        print("att img", sim_bug_img2.shape, attImg2.shape, img2.shape)
        print("sim_bug_code:", sim_bug_code.shape)
        print("sim_bug_img:", sim_bug_img2.shape)

        print("IMG 3: *******************************************************")
        #attImg3 = Attention()([bug, img3])
        attImg3 = attention_distribution(bug, img3)
        print("att img", attImg3.shape, img3.shape)
        sim_bug_img3 = Multiply()([attImg3, img3])
        print("att img", sim_bug_img3.shape, attImg3.shape, img3.shape)
        print("sim_bug_code:", sim_bug_code.shape)
        print("sim_bug_img:", sim_bug_img3.shape)

        #print("NORM AFTER: *******************************************************")
        #sim_bug_code = Activation('softmax')(sim_bug_code)
        #sim_bug_img2 = Activation('softmax')(sim_bug_img2)
        #sim_bug_img3 = Activation('softmax')(sim_bug_img3)

        sim_img_combined = Add()([sim_bug_code, sim_bug_img2])

        sim_img_combined = Activation('softmax')(sim_img_combined)

        combined = Concatenate(axis=-2)([bug, sim_bug_code])  #, sim_img_combined])  #, sim_combined3])  #
        combined = Flatten()(combined)
        # >>>>>>>>>>>> combined = Dropout(rate=.5)(combined)
        print("All Combined:", combined.shape)

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)  ##############, W_regularizer=l2(0.01))(combined)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz, code_input_img], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.config["loss"], metrics=['accuracy'])  #############3, tp_rate, tn_rate])
        #self.model.summary()

        #return model
        print("Done Building Model")

    #def gen_mode(self):
    #    return "kra_seqO_seqL"

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

class Feed_Type:
    def __init__(self):
        self.feed_CAR = "CAR"

class Base_Model_LSTM:

    def __init__(self, config_file):

        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())


        #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2"])
        #with mirrored_strategy.scope():


        # Code Image Branch ____________________________________________________________________________________________________
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        code_input_img = Input(shape=(None, None, 3), name="CODE_IMG")

        #conv1 = Conv2D(32, (3, 3))(code_input_img)
        #conv1 = Activation('relu')(conv1)
        #img1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        #conv2 = Conv2D(64, (3, 3))(img1)
        #conv2 = Activation('relu')(conv2)
        #img2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        #img2 = img_features(img2)

        #conv23 = Conv2D(32, (3, 3))(img1)
        #conv23 = Activation('relu')(conv23)
        #img23 = MaxPooling2D(pool_size=(2, 2))(conv23)

        #conv3 = Conv2D(64, (3, 3))(img23)
        #conv3 = Activation('relu')(conv3)
        #img3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        #img3 = img_features(img3)


        # Code Branch __________________________________________________________________________________________________________
        self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        code_input_f2_xyz = Input(shape=(None, ), name="CODE_TOKEN")



        code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
            input_length=None
            )(code_input_f2_xyz)

        code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        code = \
            Bidirectional(
                LSTM(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
                name='CODE_GRU_1'
            )(code_input)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)


        # Combined _____________________________________________________________________________________________________________
        print("test code", code.shape)
        print("test bug", bug.shape)
        code = Reshape((1, 64))(code)
        bug = Reshape((1, 64))(bug)

        print("NORM BEFORE: *******************************************************")
        ##bug = Activation('softmax')(bug)
        code = Activation('softmax')(code)
        #img2 = Activation('softmax')(img2)
        #img3 = Activation('softmax')(img3)

        print("CODE: *******************************************************")
        #attCode = Attention()([bug, code])
        attCode = attention_distribution(bug, code)
        print("att code", attCode.shape, code.shape)
        sim_bug_code = Multiply()([attCode, code])
        print("att code", sim_bug_code.shape, attCode.shape, code.shape)

        #print("IMG 2: *******************************************************")
        #attImg2 = Attention()([bug, img2])
        #attImg2 = attention_distribution(bug, img2)
        #print("att img", attImg2.shape, img2.shape)
        #sim_bug_img2 = Multiply()([attImg2, img2])
        #print("att img", sim_bug_img2.shape, attImg2.shape, img2.shape)
        #print("sim_bug_code:", sim_bug_code.shape)
        #print("sim_bug_img:", sim_bug_img2.shape)

        #print("IMG 3: *******************************************************")
        #attImg3 = Attention()([bug, img3])
        #attImg3 = attention_distribution(bug, img3)
        #print("att img", attImg3.shape, img3.shape)
        #sim_bug_img3 = Multiply()([attImg3, img3])
        #print("att img", sim_bug_img3.shape, attImg3.shape, img3.shape)
        #print("sim_bug_code:", sim_bug_code.shape)
        #print("sim_bug_img:", sim_bug_img3.shape)

        #print("NORM AFTER: *******************************************************")
        #sim_bug_code = Activation('softmax')(sim_bug_code)
        #sim_bug_img2 = Activation('softmax')(sim_bug_img2)
        #sim_bug_img3 = Activation('softmax')(sim_bug_img3)

        #sim_img_combined = Add()([sim_bug_code, sim_bug_img2])

        #sim_img_combined = Activation('softmax')(sim_img_combined)

        combined = Concatenate(axis=-2)([bug, sim_bug_code])  #, sim_img_combined])  #, sim_combined3])  #
        combined = Flatten()(combined)
        # >>>>>>>>>>>> combined = Dropout(rate=.5)(combined)
        print("All Combined:", combined.shape)

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)  ##############, W_regularizer=l2(0.01))(combined)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz, code_input_img], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=config["loss"], metrics=['accuracy'])  #############3, tp_rate, tn_rate])
        #self.model.summary()

        #return model
        print("Done Building Model.")


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

class Base_Model_Bare:

    def __init__(self, config_file):

        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())


        #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2"])
        #with mirrored_strategy.scope():


        # Code Image Branch ____________________________________________________________________________________________________
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        code_input_img = Input(shape=(None, None, 3), name="CODE_IMG")

        conv1 = Conv2D(32, (3, 3))(code_input_img)
        conv1 = Activation('relu')(conv1)
        img1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3))(img1)
        conv2 = Activation('relu')(conv2)
        img2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        img2 = img_features(img2)

        conv23 = Conv2D(32, (3, 3))(img1)
        conv23 = Activation('relu')(conv23)
        img23 = MaxPooling2D(pool_size=(2, 2))(conv23)

        conv3 = Conv2D(64, (3, 3))(img23)
        conv3 = Activation('relu')(conv3)
        img3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        img3 = img_features(img3)


        # Code Branch __________________________________________________________________________________________________________
        self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        code_input_f2_xyz = Input(shape=(None, ), name="CODE_TOKEN")



        code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
            input_length=None
            )(code_input_f2_xyz)

        code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        code = \
            Bidirectional(
                GRU(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
                name='CODE_GRU_1'
            )(code_input)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)


        # Combined _____________________________________________________________________________________________________________
        print("test code", code.shape)
        print("test bug", bug.shape)
        code = Reshape((1, 64))(code)
        bug = Reshape((1, 64))(bug)

        print("NORM BEFORE: *******************************************************")
        ##bug = Activation('softmax')(bug)
        code = Activation('softmax')(code)
        img2 = Activation('softmax')(img2)
        img3 = Activation('softmax')(img3)

        print("CODE: *******************************************************")
        #attCode = Attention()([bug, code])
        attCode = attention_distribution(bug, code)
        print("att code", attCode.shape, code.shape)
        sim_bug_code = Multiply()([attCode, code])
        print("att code", sim_bug_code.shape, attCode.shape, code.shape)


        print("IMG 2: *******************************************************")
        #attImg2 = Attention()([bug, img2])
        attImg2 = attention_distribution(bug, img2)
        print("att img", attImg2.shape, img2.shape)
        sim_bug_img2 = Multiply()([attImg2, img2])
        print("att img", sim_bug_img2.shape, attImg2.shape, img2.shape)
        print("sim_bug_code:", sim_bug_code.shape)
        print("sim_bug_img:", sim_bug_img2.shape)

        print("IMG 3: *******************************************************")
        #attImg3 = Attention()([bug, img3])
        attImg3 = attention_distribution(bug, img3)
        print("att img", attImg3.shape, img3.shape)
        sim_bug_img3 = Multiply()([attImg3, img3])
        print("att img", sim_bug_img3.shape, attImg3.shape, img3.shape)
        print("sim_bug_code:", sim_bug_code.shape)
        print("sim_bug_img:", sim_bug_img3.shape)

        #print("NORM AFTER: *******************************************************")
        #sim_bug_code = Activation('softmax')(sim_bug_code)
        #sim_bug_img2 = Activation('softmax')(sim_bug_img2)
        #sim_bug_img3 = Activation('softmax')(sim_bug_img3)

        sim_img_combined = Add()([sim_bug_code, sim_bug_img2])

        sim_img_combined = Activation('softmax')(sim_img_combined)

        combined = Concatenate(axis=-2)([bug, code])  #sim_bug_code])  #, sim_img_combined])  #, sim_combined3])  #
        combined = Flatten()(combined)
        # >>>>>>>>>>>> combined = Dropout(rate=.5)(combined)
        print("All Combined:", combined.shape)

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)  ##############, W_regularizer=l2(0.01))(combined)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz, code_input_img], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=config["loss"], metrics=['accuracy'])  #############3, tp_rate, tn_rate])
        #self.model.summary()

        #return model
        print("Done Building Model.")


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

class IMG_Model_Attention_Before:

    def __init__(self, config_file):

        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())


        #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2"])
        #with mirrored_strategy.scope():


        # Code Image Branch ____________________________________________________________________________________________________
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        code_input_img = Input(shape=(None, None, 3), name="CODE_IMG")

        conv1 = Conv2D(32, (3, 3))(code_input_img)
        conv1 = Activation('relu')(conv1)
        img1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3))(img1)
        conv2 = Activation('relu')(conv2)
        img2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        img2 = img_features(img2)

        #conv23 = Conv2D(32, (3, 3))(img1)
        #conv23 = Activation('relu')(conv23)
        #img23 = MaxPooling2D(pool_size=(2, 2))(conv23)

        #conv3 = Conv2D(64, (3, 3))(img23)
        #conv3 = Activation('relu')(conv3)
        #img3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        #img3 = img_features(img3)


        # Code Branch __________________________________________________________________________________________________________
        self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        code_input_f2_xyz = Input(shape=(None, ), name="CODE_TOKEN")



        code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
            input_length=None
            )(code_input_f2_xyz)

        code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        code = \
            Bidirectional(
                GRU(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
                name='CODE_GRU_1'
            )(code_input)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)


        # Combined _____________________________________________________________________________________________________________
        print("test code", code.shape)
        print("test bug", bug.shape)
        code = Reshape((1, 64))(code)
        bug = Reshape((1, 64))(bug)

        print("NORM BEFORE: *******************************************************")
        ##bug = Activation('softmax')(bug)
        code = Activation('softmax')(code)
        img2 = Activation('softmax')(img2)
        #img3 = Activation('softmax')(img3)

        print("CODE: *******************************************************")
        #attCode = Attention()([bug, code])
        attCode = attention_distribution(bug, code)
        print("att code", attCode.shape, code.shape)
        sim_bug_code = Multiply()([attCode, code])
        print("att code", sim_bug_code.shape, attCode.shape, code.shape)

        print("IMG 2: *******************************************************")
        #attImg2 = Attention()([bug, img2])
        attImg2 = attention_distribution(bug, img2)
        print("att img", attImg2.shape, img2.shape)
        sim_bug_img2 = Multiply()([attImg2, img2])
        print("att img", sim_bug_img2.shape, attImg2.shape, img2.shape)
        print("sim_bug_code:", sim_bug_code.shape)
        print("sim_bug_img:", sim_bug_img2.shape)

        #print("IMG 3: *******************************************************")
        ##attImg3 = Attention()([bug, img3])
        #attImg3 = attention_distribution(bug, img3)
        #print("att img", attImg3.shape, img3.shape)
        #sim_bug_img3 = Multiply()([attImg3, img3])
        #print("att img", sim_bug_img3.shape, attImg3.shape, img3.shape)
        #print("sim_bug_code:", sim_bug_code.shape)
        #print("sim_bug_img:", sim_bug_img3.shape)

        #print("NORM AFTER: *******************************************************")
        #sim_bug_code = Activation('softmax')(sim_bug_code)
        #sim_bug_img2 = Activation('softmax')(sim_bug_img2)
        #sim_bug_img3 = Activation('softmax')(sim_bug_img3)

        sim_img_combined = Add()([sim_bug_code, sim_bug_img2])

        #sim_img_combined = Activation('softmax')(sim_img_combined)

        combined = Concatenate(axis=-2)([bug, sim_bug_code, sim_img_combined])  #, sim_combined3])  #
        combined = Flatten()(combined)
        # >>>>>>>>>>>> combined = Dropout(rate=.5)(combined)
        print("All Combined:", combined.shape)

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)  ##############, W_regularizer=l2(0.01))(combined)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz, code_input_img], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=config["loss"], metrics=['accuracy'])  #############3, tp_rate, tn_rate])
        #self.model.summary()

        #return model
        print("Done Building Model.")

    def gen_mode(self):
        return "base"

    def save_model_summary(self, path):
            '''
            https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
            https://dzone.com/articles/python-101-redirecting-stdout
            :param path:
            :return:
            '''


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



#class IMG_Model_Attention_Before_SoftmaxSim:
class IMG_Model_GRU2x1_CNN2_FC2:

    def __init__(self, config_file):
        print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())


        #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2"])
        #with mirrored_strategy.scope():


        # Code Image Branch ____________________________________________________________________________________________________
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        code_input_img = Input(shape=(None, None, 3), name="CODE_IMG")

        conv1 = Conv2D(32, (3, 3))(code_input_img)
        conv1 = Activation('relu')(conv1)
        img1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3))(img1)
        conv2 = Activation('relu')(conv2)
        img2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        img2 = img_features(img2)

        #conv23 = Conv2D(32, (3, 3))(img1)
        #conv23 = Activation('relu')(conv23)
        #img23 = MaxPooling2D(pool_size=(2, 2))(conv23)

        #conv3 = Conv2D(64, (3, 3))(img23)
        #conv3 = Activation('relu')(conv3)
        #img3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        #img3 = img_features(img3)


        # Code Branch __________________________________________________________________________________________________________
        self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        code_input_f2_xyz = Input(shape=(None, ), name="CODE_TOKEN")



        code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
            input_length=None
            )(code_input_f2_xyz)

        code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        code = \
            Bidirectional(
                GRU(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
                name='CODE_GRU_1'
            )(code_input)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)


        # Combined _____________________________________________________________________________________________________________
        print("test code", code.shape)
        print("test bug", bug.shape)
        code = Reshape((1, 64))(code)
        bug = Reshape((1, 64))(bug)

        print("NORM BEFORE: *******************************************************")
        ##bug = Activation('softmax')(bug)
        code = Activation('softmax')(code)
        img2 = Activation('softmax')(img2)
        #img3 = Activation('softmax')(img3)

        print("CODE: *******************************************************")
        #attCode = Attention()([bug, code])
        attCode = attention_distribution(bug, code)
        print("att code", attCode.shape, code.shape)
        sim_bug_code = Multiply()([attCode, code])
        print("att code", sim_bug_code.shape, attCode.shape, code.shape)

        print("IMG 2: *******************************************************")
        #attImg2 = Attention()([bug, img2])
        attImg2 = attention_distribution(bug, img2)
        print("att img", attImg2.shape, img2.shape)
        sim_bug_img2 = Multiply()([attImg2, img2])
        print("att img", sim_bug_img2.shape, attImg2.shape, img2.shape)
        print("sim_bug_code:", sim_bug_code.shape)
        print("sim_bug_img:", sim_bug_img2.shape)

        #print("IMG 3: *******************************************************")
        ##attImg3 = Attention()([bug, img3])
        #attImg3 = attention_distribution(bug, img3)
        #print("att img", attImg3.shape, img3.shape)
        #sim_bug_img3 = Multiply()([attImg3, img3])
        #print("att img", sim_bug_img3.shape, attImg3.shape, img3.shape)
        #print("sim_bug_code:", sim_bug_code.shape)
        #print("sim_bug_img:", sim_bug_img3.shape)

        #print("NORM AFTER: *******************************************************")
        #sim_bug_code = Activation('softmax')(sim_bug_code)
        #sim_bug_img2 = Activation('softmax')(sim_bug_img2)
        #sim_bug_img3 = Activation('softmax')(sim_bug_img3)

        sim_img_combined = Add()([sim_bug_code, sim_bug_img2])
        sim_img_combined = Activation('softmax')(sim_img_combined)

        combined = Concatenate(axis=-2)([bug, sim_bug_code, sim_img_combined])  #, sim_combined3])  #
        combined = Flatten()(combined)
        # >>>>>>>>>>>> combined = Dropout(rate=.5)(combined)
        print("All Combined:", combined.shape)

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)  ##############, W_regularizer=l2(0.01))(combined)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz, code_input_img], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=config["loss"], metrics=['accuracy'])  #############3, tp_rate, tn_rate])
        #self.model.summary()

        #return model
        print("Done Building Model.")


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


class IMG_Model_GRU2x1_CNN2_3:

    def __init__(self, config_file):
        print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())


        #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2"])
        #with mirrored_strategy.scope():


        # Code Image Branch ____________________________________________________________________________________________________
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        code_input_img = Input(shape=(None, None, 3), name="CODE_IMG")

        conv1 = Conv2D(32, (3, 3))(code_input_img)
        conv1 = Activation('relu')(conv1)
        img1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3))(img1)
        conv2 = Activation('relu')(conv2)
        img2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        img2 = img_features(img2)

        #conv23 = Conv2D(32, (3, 3))(img1)
        #conv23 = Activation('relu')(conv23)
        #img23 = MaxPooling2D(pool_size=(2, 2))(conv23)

        #conv3 = Conv2D(64, (3, 3))(img23)
        #conv3 = Activation('relu')(conv3)
        #img3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        #img3 = img_features(img3)


        # Code Branch __________________________________________________________________________________________________________
        self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        code_input_f2_xyz = Input(shape=(None, ), name="CODE_TOKEN")



        code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
            input_length=None
            )(code_input_f2_xyz)

        code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        code = \
            Bidirectional(
                GRU(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
                name='CODE_GRU_1'
            )(code_input)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)


        # Combined _____________________________________________________________________________________________________________
        print("test code", code.shape)
        print("test bug", bug.shape)
        code = Reshape((1, 64))(code)
        bug = Reshape((1, 64))(bug)

        print("NORM BEFORE: *******************************************************")
        ##bug = Activation('softmax')(bug)
        code = Activation('softmax')(code)
        img2 = Activation('softmax')(img2)
        #img3 = Activation('softmax')(img3)

        print("CODE: *******************************************************")
        #attCode = Attention()([bug, code])
        attCode = attention_distribution(bug, code)
        print("att code", attCode.shape, code.shape)
        sim_bug_code = Multiply()([attCode, code])
        print("att code", sim_bug_code.shape, attCode.shape, code.shape)

        print("IMG 2: *******************************************************")
        #attImg2 = Attention()([bug, img2])
        attImg2 = attention_distribution(bug, img2)
        print("att img", attImg2.shape, img2.shape)
        sim_bug_img2 = Multiply()([attImg2, img2])
        print("att img", sim_bug_img2.shape, attImg2.shape, img2.shape)
        print("sim_bug_code:", sim_bug_code.shape)
        print("sim_bug_img:", sim_bug_img2.shape)

        #print("IMG 3: *******************************************************")
        ##attImg3 = Attention()([bug, img3])
        #attImg3 = attention_distribution(bug, img3)
        #print("att img", attImg3.shape, img3.shape)
        #sim_bug_img3 = Multiply()([attImg3, img3])
        #print("att img", sim_bug_img3.shape, attImg3.shape, img3.shape)
        #print("sim_bug_code:", sim_bug_code.shape)
        #print("sim_bug_img:", sim_bug_img3.shape)

        #print("NORM AFTER: *******************************************************")
        #sim_bug_code = Activation('softmax')(sim_bug_code)
        #sim_bug_img2 = Activation('softmax')(sim_bug_img2)
        #sim_bug_img3 = Activation('softmax')(sim_bug_img3)

        sim_img_combined = Add()([sim_bug_code, sim_bug_img2])
        sim_img_combined = Activation('softmax')(sim_img_combined)

        combined = Concatenate(axis=-2)([bug, sim_bug_code, sim_img_combined, sim_bug_img2])  #, sim_combined3])  #
        combined = Flatten()(combined)
        # >>>>>>>>>>>> combined = Dropout(rate=.5)(combined)
        print("All Combined:", combined.shape)

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)  ##############, W_regularizer=l2(0.01))(combined)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz, code_input_img], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=config["loss"], metrics=['accuracy'])  #############3, tp_rate, tn_rate])
        #self.model.summary()

        #return model
        print("Done Building Model.")


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

class IMG_Model_GRU2x1_CNN2_FC2_Direct:

    def __init__(self, config_file):
        print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())


        #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2"])
        #with mirrored_strategy.scope():


        # Code Image Branch ____________________________________________________________________________________________________
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        code_input_img = Input(shape=(None, None, 3), name="CODE_IMG")

        conv1 = Conv2D(32, (3, 3))(code_input_img)
        conv1 = Activation('relu')(conv1)
        img1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3))(img1)
        conv2 = Activation('relu')(conv2)
        img2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        img2 = img_features(img2)

        #conv23 = Conv2D(32, (3, 3))(img1)
        #conv23 = Activation('relu')(conv23)
        #img23 = MaxPooling2D(pool_size=(2, 2))(conv23)

        #conv3 = Conv2D(64, (3, 3))(img23)
        #conv3 = Activation('relu')(conv3)
        #img3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        #img3 = img_features(img3)


        # Code Branch __________________________________________________________________________________________________________
        self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        code_input_f2_xyz = Input(shape=(None, ), name="CODE_TOKEN")



        code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
            input_length=None
            )(code_input_f2_xyz)

        code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        code = \
            Bidirectional(
                GRU(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
                name='CODE_GRU_1'
            )(code_input)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)


        # Combined _____________________________________________________________________________________________________________
        print("test code", code.shape)
        print("test bug", bug.shape)
        code = Reshape((1, 64))(code)
        bug = Reshape((1, 64))(bug)

        print("NORM BEFORE: *******************************************************")
        ##bug = Activation('softmax')(bug)
        code = Activation('softmax')(code)
        img2 = Activation('softmax')(img2)
        #img3 = Activation('softmax')(img3)

        print("CODE: *******************************************************")
        #attCode = Attention()([bug, code])
        attCode = attention_distribution(bug, code)
        print("att code", attCode.shape, code.shape)
        sim_bug_code = Multiply()([attCode, code])
        print("att code", sim_bug_code.shape, attCode.shape, code.shape)

        print("IMG 2: *******************************************************")
        #attImg2 = Attention()([bug, img2])
        attImg2 = attention_distribution(bug, img2)
        print("att img", attImg2.shape, img2.shape)
        sim_bug_img2 = Multiply()([attImg2, img2])
        print("att img", sim_bug_img2.shape, attImg2.shape, img2.shape)
        print("sim_bug_code:", sim_bug_code.shape)
        print("sim_bug_img:", sim_bug_img2.shape)

        #print("IMG 3: *******************************************************")
        ##attImg3 = Attention()([bug, img3])
        #attImg3 = attention_distribution(bug, img3)
        #print("att img", attImg3.shape, img3.shape)
        #sim_bug_img3 = Multiply()([attImg3, img3])
        #print("att img", sim_bug_img3.shape, attImg3.shape, img3.shape)
        #print("sim_bug_code:", sim_bug_code.shape)
        #print("sim_bug_img:", sim_bug_img3.shape)

        #print("NORM AFTER: *******************************************************")
        #sim_bug_code = Activation('softmax')(sim_bug_code)
        #sim_bug_img2 = Activation('softmax')(sim_bug_img2)
        #sim_bug_img3 = Activation('softmax')(sim_bug_img3)

        #sim_img_combined = Add()([sim_bug_code, sim_bug_img2])
        #sim_img_combined = Activation('softmax')(sim_img_combined)

        combined = Concatenate(axis=-2)([bug, sim_bug_code, sim_bug_img2])  #, sim_combined3])  #
        combined = Flatten()(combined)
        # >>>>>>>>>>>> combined = Dropout(rate=.5)(combined)
        print("All Combined:", combined.shape)

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)  ##############, W_regularizer=l2(0.01))(combined)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz, code_input_img], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=config["loss"], metrics=['accuracy'])  #############3, tp_rate, tn_rate])
        #self.model.summary()

        #return model
        print("Done Building Model.")


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

class IMG_Model_Attention_Before_Level23:

    def __init__(self, config_file):
        print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())


        #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2"])
        #with mirrored_strategy.scope():


        # Code Image Branch ____________________________________________________________________________________________________
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        code_input_img = Input(shape=(None, None, 3), name="CODE_IMG")

        conv1 = Conv2D(32, (3, 3))(code_input_img)
        conv1 = Activation('relu')(conv1)
        img1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3))(img1)
        conv2 = Activation('relu')(conv2)
        img2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        img2 = img_features(img2)

        conv23 = Conv2D(32, (3, 3))(img1)
        conv23 = Activation('relu')(conv23)
        img23 = MaxPooling2D(pool_size=(2, 2))(conv23)

        conv3 = Conv2D(64, (3, 3))(img23)
        conv3 = Activation('relu')(conv3)
        img3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        img3 = img_features(img3)


        # Code Branch __________________________________________________________________________________________________________
        self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        code_input_f2_xyz = Input(shape=(None, ), name="CODE_TOKEN")



        code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
            input_length=None
            )(code_input_f2_xyz)

        code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        code = \
            Bidirectional(
                GRU(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
                name='CODE_GRU_1'
            )(code_input)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)


        # Combined _____________________________________________________________________________________________________________
        print("test code", code.shape)
        print("test bug", bug.shape)
        code = Reshape((1, 64))(code)
        bug = Reshape((1, 64))(bug)

        print("NORM BEFORE: *******************************************************")
        ##bug = Activation('softmax')(bug)
        code = Activation('softmax')(code)
        img2 = Activation('softmax')(img2)
        #img3 = Activation('softmax')(img3)

        print("CODE: *******************************************************")
        #attCode = Attention()([bug, code])
        attCode = attention_distribution(bug, code)
        print("att code", attCode.shape, code.shape)
        sim_bug_code = Multiply()([attCode, code])
        print("att code", sim_bug_code.shape, attCode.shape, code.shape)

        print("IMG 2: *******************************************************")
        #attImg2 = Attention()([bug, img2])
        attImg2 = attention_distribution(bug, img2)
        print("att img", attImg2.shape, img2.shape)
        sim_bug_img2 = Multiply()([attImg2, img2])
        print("att img", sim_bug_img2.shape, attImg2.shape, img2.shape)
        print("sim_bug_code:", sim_bug_code.shape)
        print("sim_bug_img:", sim_bug_img2.shape)

        print("IMG 3: *******************************************************")
        #attImg3 = Attention()([bug, img3])
        attImg3 = attention_distribution(bug, img3)
        print("att img", attImg3.shape, img3.shape)
        sim_bug_img3 = Multiply()([attImg3, img3])
        print("att img", sim_bug_img3.shape, attImg3.shape, img3.shape)
        print("sim_bug_code:", sim_bug_code.shape)
        print("sim_bug_img:", sim_bug_img3.shape)

        #print("NORM AFTER: *******************************************************")
        #sim_bug_code = Activation('softmax')(sim_bug_code)
        sim_bug_img2 = Activation('softmax')(sim_bug_img2)
        sim_bug_img3 = Activation('softmax')(sim_bug_img3)

        sim_img_combined = Add()([sim_bug_code, sim_bug_img2])
        sim_img_combined = Activation('softmax')(sim_img_combined)

        combined = Concatenate(axis=-2)([bug, sim_bug_code, sim_bug_img2, sim_bug_img3])  #, sim_combined3])  #
        combined = Flatten()(combined)
        combined = Dropout(rate=.2)(combined)
        print("All Combined:", combined.shape)

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)  ##############, W_regularizer=l2(0.01))(combined)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz, code_input_img], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=config["loss"], metrics=['accuracy'])  #############3, tp_rate, tn_rate])
        #self.model.summary()

        #return model
        print("Done Building Model.")


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

class IMG_Model_Attention_Before_Level3:

    def __init__(self, config_file):
        print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())


        #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2"])
        #with mirrored_strategy.scope():


        # Code Image Branch ____________________________________________________________________________________________________
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        code_input_img = Input(shape=(None, None, 3), name="CODE_IMG")

        conv1 = Conv2D(32, (3, 3))(code_input_img)
        conv1 = Activation('relu')(conv1)
        img1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        #conv2 = Conv2D(64, (3, 3))(img1)
        #conv2 = Activation('relu')(conv2)
        #img2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        #img2 = img_features(img2)

        conv23 = Conv2D(32, (3, 3))(img1)
        conv23 = Activation('relu')(conv23)
        img23 = MaxPooling2D(pool_size=(2, 2))(conv23)

        conv3 = Conv2D(64, (3, 3))(img23)
        conv3 = Activation('relu')(conv3)
        img3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        img3 = img_features(img3)


        # Code Branch __________________________________________________________________________________________________________
        self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        code_input_f2_xyz = Input(shape=(None, ), name="CODE_TOKEN")



        code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
            input_length=None
            )(code_input_f2_xyz)

        code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        code = \
            Bidirectional(
                GRU(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
                name='CODE_GRU_1'
            )(code_input)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)


        # Combined _____________________________________________________________________________________________________________
        print("test code", code.shape)
        print("test bug", bug.shape)
        code = Reshape((1, 64))(code)
        bug = Reshape((1, 64))(bug)

        print("NORM BEFORE: *******************************************************")
        ##bug = Activation('softmax')(bug)
        code = Activation('softmax')(code)
        #img2 = Activation('softmax')(img2)
        img3 = Activation('softmax')(img3)

        print("CODE: *******************************************************")
        #attCode = Attention()([bug, code])
        attCode = attention_distribution(bug, code)
        print("att code", attCode.shape, code.shape)
        sim_bug_code = Multiply()([attCode, code])
        print("att code", sim_bug_code.shape, attCode.shape, code.shape)

        print("IMG 2: *******************************************************")
        #attImg2 = Attention()([bug, img2])
        #attImg2 = attention_distribution(bug, img2)
        #print("att img", attImg2.shape, img2.shape)
        #sim_bug_img2 = Multiply()([attImg2, img2])
        #print("att img", sim_bug_img2.shape, attImg2.shape, img2.shape)
        #print("sim_bug_code:", sim_bug_code.shape)
        #print("sim_bug_img:", sim_bug_img2.shape)

        print("IMG 3: *******************************************************")
        #attImg3 = Attention()([bug, img3])
        attImg3 = attention_distribution(bug, img3)
        print("att img", attImg3.shape, img3.shape)
        sim_bug_img3 = Multiply()([attImg3, img3])
        print("att img", sim_bug_img3.shape, attImg3.shape, img3.shape)
        print("sim_bug_code:", sim_bug_code.shape)
        print("sim_bug_img:", sim_bug_img3.shape)

        #print("NORM AFTER: *******************************************************")
        #sim_bug_code = Activation('softmax')(sim_bug_code)
        #sim_bug_img2 = Activation('softmax')(sim_bug_img2)
        #sim_bug_img3 = Activation('softmax')(sim_bug_img3)

        #sim_img_combined = Add()([sim_bug_code, sim_bug_img2])
        #sim_img_combined = Activation('softmax')(sim_img_combined)

        combined = Concatenate(axis=-2)([bug, sim_bug_code, sim_bug_img3])  #, sim_combined3])  #
        combined = Flatten()(combined)
        combined = Dropout(rate=.2)(combined)
        print("All Combined:", combined.shape)

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)  ##############, W_regularizer=l2(0.01))(combined)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz, code_input_img], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=config["loss"], metrics=['accuracy'])  #############3, tp_rate, tn_rate])
        #self.model.summary()

        #return model
        print("Done Building Model.")


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

class IMG_Model_Attention_Before_Level2:

    def __init__(self, config_file):
        print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())


        #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2"])
        #with mirrored_strategy.scope():


        # Code Image Branch ____________________________________________________________________________________________________
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        code_input_img = Input(shape=(None, None, 3), name="CODE_IMG")

        conv1 = Conv2D(32, (3, 3))(code_input_img)
        conv1 = Activation('relu')(conv1)
        img1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3))(img1)
        conv2 = Activation('relu')(conv2)
        img2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        img2 = img_features(img2)

        #conv23 = Conv2D(32, (3, 3))(img1)
        #conv23 = Activation('relu')(conv23)
        #img23 = MaxPooling2D(pool_size=(2, 2))(conv23)

        #conv3 = Conv2D(64, (3, 3))(img23)
        #conv3 = Activation('relu')(conv3)
        #img3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        #img3 = img_features(img3)


        # Code Branch __________________________________________________________________________________________________________
        self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        code_input_f2_xyz = Input(shape=(None, ), name="CODE_TOKEN")



        code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
            input_length=None
            )(code_input_f2_xyz)

        code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        code = \
            Bidirectional(
                GRU(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
                name='CODE_GRU_1'
            )(code_input)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)


        # Combined _____________________________________________________________________________________________________________
        print("test code", code.shape)
        print("test bug", bug.shape)
        code = Reshape((1, 64))(code)
        bug = Reshape((1, 64))(bug)

        print("NORM BEFORE: *******************************************************")
        ##bug = Activation('softmax')(bug)
        code = Activation('softmax')(code)
        img2 = Activation('softmax')(img2)
        #img3 = Activation('softmax')(img3)

        print("CODE: *******************************************************")
        #attCode = Attention()([bug, code])
        attCode = attention_distribution(bug, code)
        print("att code", attCode.shape, code.shape)
        sim_bug_code = Multiply()([attCode, code])
        print("att code", sim_bug_code.shape, attCode.shape, code.shape)

        print("IMG 2: *******************************************************")
        #attImg2 = Attention()([bug, img2])
        attImg2 = attention_distribution(bug, img2)
        print("att img", attImg2.shape, img2.shape)
        sim_bug_img2 = Multiply()([attImg2, img2])
        print("att img", sim_bug_img2.shape, attImg2.shape, img2.shape)
        print("sim_bug_code:", sim_bug_code.shape)
        print("sim_bug_img:", sim_bug_img2.shape)

        print("IMG 3: *******************************************************")
        #attImg3 = Attention()([bug, img3])
        #attImg3 = attention_distribution(bug, img3)
        #print("att img", attImg3.shape, img3.shape)
        #sim_bug_img3 = Multiply()([attImg3, img3])
        #print("att img", sim_bug_img3.shape, attImg3.shape, img3.shape)
        #print("sim_bug_code:", sim_bug_code.shape)
        #print("sim_bug_img:", sim_bug_img3.shape)

        #print("NORM AFTER: *******************************************************")
        #sim_bug_code = Activation('softmax')(sim_bug_code)
        #sim_bug_img2 = Activation('softmax')(sim_bug_img2)
        #sim_bug_img3 = Activation('softmax')(sim_bug_img3)

        #sim_img_combined = Add()([sim_bug_code, sim_bug_img2])
        #sim_img_combined = Activation('softmax')(sim_img_combined)

        combined = Concatenate(axis=-2)([bug, sim_bug_code, sim_bug_img2])  #, sim_combined3])  #
        combined = Flatten()(combined)
        combined = Dropout(rate=.2)(combined)
        print("All Combined:", combined.shape)

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)  ##############, W_regularizer=l2(0.01))(combined)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz, code_input_img], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=config["loss"], metrics=['accuracy'])  #############3, tp_rate, tn_rate])
        #self.model.summary()

        #return model
        print("Done Building Model.")


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

class IMG_Model_LSTM_Level2:

    def __init__(self, config_file):
        print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())


        #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2"])
        #with mirrored_strategy.scope():


        # Code Image Branch ____________________________________________________________________________________________________
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        code_input_img = Input(shape=(None, None, 3), name="CODE_IMG")

        conv1 = Conv2D(32, (3, 3))(code_input_img)
        conv1 = Activation('relu')(conv1)
        img1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3))(img1)
        conv2 = Activation('relu')(conv2)
        img2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        img2 = img_features(img2)

        #conv23 = Conv2D(32, (3, 3))(img1)
        #conv23 = Activation('relu')(conv23)
        #img23 = MaxPooling2D(pool_size=(2, 2))(conv23)

        #conv3 = Conv2D(64, (3, 3))(img23)
        #conv3 = Activation('relu')(conv3)
        #img3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        #img3 = img_features(img3)


        # Code Branch __________________________________________________________________________________________________________
        self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        code_input_f2_xyz = Input(shape=(None, ), name="CODE_TOKEN")



        code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
            input_length=None
            )(code_input_f2_xyz)

        code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        code = \
            Bidirectional(
                LSTM(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
                name='CODE_GRU_1'
            )(code_input)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = \
            Bidirectional(
                LSTM(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)


        # Combined _____________________________________________________________________________________________________________
        print("test code", code.shape)
        print("test bug", bug.shape)
        code = Reshape((1, 64))(code)
        bug = Reshape((1, 64))(bug)

        print("NORM BEFORE: *******************************************************")
        ##bug = Activation('softmax')(bug)
        code = Activation('softmax')(code)
        img2 = Activation('softmax')(img2)
        #img3 = Activation('softmax')(img3)

        print("CODE: *******************************************************")
        #attCode = Attention()([bug, code])
        attCode = attention_distribution(bug, code)
        print("att code", attCode.shape, code.shape)
        sim_bug_code = Multiply()([attCode, code])
        print("att code", sim_bug_code.shape, attCode.shape, code.shape)

        print("IMG 2: *******************************************************")
        #attImg2 = Attention()([bug, img2])
        attImg2 = attention_distribution(bug, img2)
        print("att img", attImg2.shape, img2.shape)
        sim_bug_img2 = Multiply()([attImg2, img2])
        print("att img", sim_bug_img2.shape, attImg2.shape, img2.shape)
        print("sim_bug_code:", sim_bug_code.shape)
        print("sim_bug_img:", sim_bug_img2.shape)

        print("IMG 3: *******************************************************")
        #attImg3 = Attention()([bug, img3])
        #attImg3 = attention_distribution(bug, img3)
        #print("att img", attImg3.shape, img3.shape)
        #sim_bug_img3 = Multiply()([attImg3, img3])
        #print("att img", sim_bug_img3.shape, attImg3.shape, img3.shape)
        #print("sim_bug_code:", sim_bug_code.shape)
        #print("sim_bug_img:", sim_bug_img3.shape)

        #print("NORM AFTER: *******************************************************")
        #sim_bug_code = Activation('softmax')(sim_bug_code)
        #sim_bug_img2 = Activation('softmax')(sim_bug_img2)
        #sim_bug_img3 = Activation('softmax')(sim_bug_img3)

        #sim_img_combined = Add()([sim_bug_code, sim_bug_img2])
        #sim_img_combined = Activation('softmax')(sim_img_combined)
        sim_img_combined = Add()([sim_bug_code, sim_bug_img2])
        sim_img_combined = Activation('softmax')(sim_img_combined)

        combined = Concatenate(axis=-2)([bug, sim_bug_code, sim_img_combined])  #, sim_combined3])  #
        combined = Flatten()(combined)
        #combined = Dropout(rate=.2)(combined)
        print("All Combined:", combined.shape)

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)  ##############, W_regularizer=l2(0.01))(combined)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz, code_input_img], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=config["loss"], metrics=['accuracy'])  #############3, tp_rate, tn_rate])
        #self.model.summary()

        #return model
        print("Done Building Model.")


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

class Base_Model_IMG_ONLY:

    def __init__(self, config_file):
        print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())


        #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2"])
        #with mirrored_strategy.scope():


        # Code Image Branch ____________________________________________________________________________________________________
        # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        code_input_img = Input(shape=(None, None, 3), name="CODE_IMG")

        conv1 = Conv2D(32, (3, 3))(code_input_img)
        conv1 = Activation('relu')(conv1)
        img1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3))(img1)
        conv2 = Activation('relu')(conv2)
        img2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        img2 = img_features(img2)

        #conv23 = Conv2D(32, (3, 3))(img1)
        #conv23 = Activation('relu')(conv23)
        #img23 = MaxPooling2D(pool_size=(2, 2))(conv23)

        #conv3 = Conv2D(64, (3, 3))(img23)
        #conv3 = Activation('relu')(conv3)
        #img3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        #img3 = img_features(img3)


        # Code Branch __________________________________________________________________________________________________________
        self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        #self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        #self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        #self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        #self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        code_input_f2_xyz = Input(shape=(None, ), name="CODE_TOKEN")



        #code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
        #    input_length=None
        #    )(code_input_f2_xyz)

        #code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        #code = \
        #    Bidirectional(
        #        LSTM(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
        #        name='CODE_GRU_1'
        #    )(code_input)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = \
            Bidirectional(
                LSTM(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)


        # Combined _____________________________________________________________________________________________________________
        #print("test code", code.shape)
        print("test bug", bug.shape)
        #code = Reshape((1, 64))(code)
        bug = Reshape((1, 64))(bug)

        print("NORM BEFORE: *******************************************************")
        ##bug = Activation('softmax')(bug)
        #code = Activation('softmax')(code)
        img2 = Activation('softmax')(img2)
        #img3 = Activation('softmax')(img3)

        print("CODE: *******************************************************")
        #attCode = Attention()([bug, code])
        #attCode = attention_distribution(bug, code)
        #print("att code", attCode.shape, code.shape)
        #sim_bug_code = Multiply()([attCode, code])
        #print("att code", sim_bug_code.shape, attCode.shape, code.shape)

        print("IMG 2: *******************************************************")
        #attImg2 = Attention()([bug, img2])
        attImg2 = attention_distribution(bug, img2)
        print("att img", attImg2.shape, img2.shape)
        sim_bug_img2 = Multiply()([attImg2, img2])
        print("att img", sim_bug_img2.shape, attImg2.shape, img2.shape)
        #print("sim_bug_code:", sim_bug_code.shape)
        print("sim_bug_img:", sim_bug_img2.shape)

        #print("IMG 3: *******************************************************")
        #attImg3 = Attention()([bug, img3])
        #attImg3 = attention_distribution(bug, img3)
        #print("att img", attImg3.shape, img3.shape)
        #sim_bug_img3 = Multiply()([attImg3, img3])
        #print("att img", sim_bug_img3.shape, attImg3.shape, img3.shape)
        #print("sim_bug_code:", sim_bug_code.shape)
        #print("sim_bug_img:", sim_bug_img3.shape)

        #print("NORM AFTER: *******************************************************")
        #sim_bug_code = Activation('softmax')(sim_bug_code)
        #sim_bug_img2 = Activation('softmax')(sim_bug_img2)
        #sim_bug_img3 = Activation('softmax')(sim_bug_img3)

        #sim_img_combined = Add()([sim_bug_code, sim_bug_img2])
        #sim_img_combined = Activation('softmax')(sim_img_combined)
        #sim_img_combined = Add()([sim_bug_code, sim_bug_img2])
        #sim_img_combined = Activation('softmax')(sim_img_combined)

        combined = Concatenate(axis=-2)([bug, sim_bug_img2])  #, sim_combined3])  #
        combined = Flatten()(combined)
        #combined = Dropout(rate=.2)(combined)
        print("All Combined:", combined.shape)

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)  ##############, W_regularizer=l2(0.01))(combined)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz, code_input_img], outputs=binary)
        #self.model = Model(inputs=[bug_input, code_input_img], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=config["loss"], metrics=['accuracy'])  #############3, tp_rate, tn_rate])
        #self.model.summary()

        #return model
        print("Done Building Model.")


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

class KRA_Model_B1:

    def __init__(self, config_file):

        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())

        #https: // stats.stackexchange.com / questions / 273486 / network - in -network - in -keras - implementation
        # KRA Code Branch
        kra_input = Input(shape=(2054, 9), name='kra_input')
        kra_conv = Conv1D(filters=192, kernel_size=5, activation='relu')(kra_input)
        kra_conv = Conv1D(filters=160, kernel_size=1, strides=1, activation='relu')(kra_conv)
        kra_conv = Conv1D(filters=96, kernel_size=1, strides=1, activation='relu')(kra_conv)
        kra_conv = MaxPool1D(pool_size=3, strides=2)(kra_conv)
        kra_conv = Dropout(0.7)(kra_conv)

        kra_conv = Conv1D(filters=192, kernel_size=5, activation='relu')(kra_conv)
        kra_conv = Conv1D(filters=192, kernel_size=1, strides=1, activation='relu')(kra_conv)
        kra_conv = Conv1D(filters=192, kernel_size=1, strides=1, activation='relu')(kra_conv)
        kra_conv = MaxPool1D(pool_size=3, strides=2)(kra_conv)
        kra_conv = Dropout(0.7)(kra_conv)

        kra_conv = Conv1D(filters=192, kernel_size=3, activation='relu')(kra_conv)
        kra_conv = Conv1D(filters=192, kernel_size=1, strides=1, activation='relu')(kra_conv)
        kra_conv = Conv1D(filters=64, kernel_size=1, strides=1, activation='relu')(kra_conv)

        kra = GlobalAveragePooling1D()(kra_conv)


        print("KRA Shape", K.shape(kra))

        # Code Branch __________________________________________________________________________________________________________
        #self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        #self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        #self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        #self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        #self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        #code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        #code_input_f2_xyz = Input(shape=(None,), name="CODE_TOKEN")

        #code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
        #                                  input_length=None
        #                                  )(code_input_f2_xyz)

        #code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        #code = \
        #    Bidirectional(
        #        LSTM(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
        #        name='CODE_GRU_1'
        #    )(code_input)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)



        # Combined _____________________________________________________________________________________________________________
        #print("test code", code.shape)
        print("test bug", bug.shape)
        bug = Reshape((1, 64))(bug)
        # code = Reshape((1, 64))(code)
        kra = Reshape((1, 64))(kra)

        print("NORM BEFORE: *******************************************************")
        ##bug = Activation('softmax')(bug)
        # code = Activation('softmax')(code)
        #img2 = Activation('softmax')(img2)
        # img3 = Activation('softmax')(img3)
        kra = Activation('softmax')(kra)

        print("CODE: *******************************************************")
        ## attCode = Attention()([bug, code])
        #attCode = attention_distribution(bug, code)
        #print("att code", attCode.shape, code.shape)
        #sim_bug_code = Multiply()([attCode, code])
        #print("att code", sim_bug_code.shape, attCode.shape, code.shape)

        print("KRA: *******************************************************")
        # attCode = Attention()([bug, code])
        attKRA = attention_distribution(bug, kra)
        print("att code", attKRA.shape, kra.shape)
        sim_bug_kra = Multiply()([attKRA, kra])
        print("att code", sim_bug_kra.shape, attKRA.shape, kra.shape)

        #print("IMG 2: *******************************************************")
        # attImg2 = Attention()([bug, img2])
        #attImg2 = attention_distribution(bug, img2)
        #print("att img", attImg2.shape, img2.shape)
        #sim_bug_img2 = Multiply()([attImg2, img2])
        #print("att img", sim_bug_img2.shape, attImg2.shape, img2.shape)
        ## print("sim_bug_code:", sim_bug_code.shape)
        #print("sim_bug_img:", sim_bug_img2.shape)

        print("COMBINE: *******************************************************")
        # sim_img_combined = Add()([sim_bug_code, sim_bug_img2])
        # sim_img_combined = Activation('softmax')(sim_img_combined)
        # sim_img_combined = Add()([sim_bug_code, sim_bug_img2])
        # sim_img_combined = Activation('softmax')(sim_img_combined)

        #combined = Concatenate(axis=-2)([bug, sim_bug_img2])  # , sim_combined3])  #
        combined = Concatenate(axis=-2)([bug, kra, sim_bug_kra])  # , sim_combined3])  #
        combined = Flatten()(combined)
        # combined = Dropout(rate=.2)(combined)
        print("All Combined:", combined.shape)

        print("DECISION: *******************************************************")

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)
        print("binary", binary.shape)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_input], outputs=binary)
        #self.model.summary()

        #return model
        print("Done Building Model.")

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


class KRA_Model_B2:

    def __init__(self, config_file):

        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())

        #https: // stats.stackexchange.com / questions / 273486 / network - in -network - in -keras - implementation
        # KRA Code Branch
        kra_input = Input(shape=(2054, 9), name='kra_input')
        kra_conv = Conv1D(filters=192, kernel_size=5, activation='relu')(kra_input)
        kra_conv = Conv1D(filters=160, kernel_size=1, strides=1, activation='relu')(kra_conv)
        kra_conv = Conv1D(filters=96, kernel_size=1, strides=1, activation='relu')(kra_conv)
        kra_conv = MaxPool1D(pool_size=3, strides=2)(kra_conv)
        kra_conv = Dropout(0.7)(kra_conv)

        kra_conv = Conv1D(filters=192, kernel_size=5, activation='relu')(kra_conv)
        kra_conv = Conv1D(filters=192, kernel_size=1, strides=1, activation='relu')(kra_conv)
        kra_conv = Conv1D(filters=192, kernel_size=1, strides=1, activation='relu')(kra_conv)
        kra_conv = MaxPool1D(pool_size=3, strides=2)(kra_conv)
        kra_conv = Dropout(0.7)(kra_conv)

        kra_conv = Conv1D(filters=192, kernel_size=3, activation='relu')(kra_conv)
        kra_conv = Conv1D(filters=192, kernel_size=1, strides=1, activation='relu')(kra_conv)
        kra_conv = Conv1D(filters=64, kernel_size=1, strides=1, activation='relu')(kra_conv)

        kra = GlobalAveragePooling1D()(kra_conv)
        print("KRA Shape", K.shape(kra))

        # Code Branch __________________________________________________________________________________________________________
        self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        code_input_f2_xyz = Input(shape=(None,), name="CODE_TOKEN")

        code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
                                          input_length=None
                                          )(code_input_f2_xyz)

        code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        code = \
            Bidirectional(
                LSTM(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
                name='CODE_GRU_1'
            )(code_input)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)



        # Combined _____________________________________________________________________________________________________________
        #print("test code", code.shape)
        print("test bug", bug.shape)
        bug = Reshape((1, 64))(bug)
        code = Reshape((1, 64))(code)
        kra = Reshape((1, 64))(kra)

        print("NORM BEFORE: *******************************************************")
        ##bug = Activation('softmax')(bug)
        # code = Activation('softmax')(code)
        #img2 = Activation('softmax')(img2)
        # img3 = Activation('softmax')(img3)
        kra = Activation('softmax')(kra)

        print("CODE: *******************************************************")
        ## attCode = Attention()([bug, code])
        attCode = attention_distribution(bug, code)
        print("att code", attCode.shape, code.shape)
        sim_bug_code = Multiply()([attCode, code])
        print("att code", sim_bug_code.shape, attCode.shape, code.shape)

        print("KRA: *******************************************************")
        # attCode = Attention()([bug, code])
        attKRA = attention_distribution(bug, kra)
        print("att code", attKRA.shape, kra.shape)
        sim_bug_kra = Multiply()([attKRA, kra])
        print("att code", sim_bug_kra.shape, attKRA.shape, kra.shape)

        #print("IMG 2: *******************************************************")
        # attImg2 = Attention()([bug, img2])
        #attImg2 = attention_distribution(bug, img2)
        #print("att img", attImg2.shape, img2.shape)
        #sim_bug_img2 = Multiply()([attImg2, img2])
        #print("att img", sim_bug_img2.shape, attImg2.shape, img2.shape)
        ## print("sim_bug_code:", sim_bug_code.shape)
        #print("sim_bug_img:", sim_bug_img2.shape)

        print("COMBINE: *******************************************************")
        # sim_img_combined = Add()([sim_bug_code, sim_bug_img2])
        # sim_img_combined = Activation('softmax')(sim_img_combined)
        # sim_img_combined = Add()([sim_bug_code, sim_bug_img2])
        # sim_img_combined = Activation('softmax')(sim_img_combined)

        #combined = Concatenate(axis=-2)([bug, sim_bug_img2])  # , sim_combined3])  #
        combined = Concatenate(axis=-2)([bug, kra, sim_bug_kra, code, sim_bug_code])  # , sim_combined3])  #
        combined = Flatten()(combined)
        # combined = Dropout(rate=.2)(combined)
        print("All Combined:", combined.shape)

        print("DECISION: *******************************************************")

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)
        print("binary", binary.shape)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz, kra_input], outputs=binary)
        #self.model.summary()

        #return model
        print("Done Building Model.")

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

class KRA_Model_CODE_FRAGE:

    def __init__(self, config_file):

        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())

        #https: // stats.stackexchange.com / questions / 273486 / network - in -network - in -keras - implementation
        # KRA Code Branch
        kra_input = Input(shape=(2054, 9), name='kra_input')
        kra = Conv1D(filters=32, kernel_size=7, activation='relu')(kra_input)
        kra = MaxPool1D(pool_size=4, strides=3)(kra)
        kra = Conv1D(filters=32, kernel_size=7, activation='relu')(kra)
        kra = MaxPool1D(pool_size=2, strides=2)(kra)
        kra = Conv1D(filters=64, kernel_size=1, activation='relu')(kra)
        kra = GlobalAveragePooling1D()(kra)
        print("KRA Shape", K.shape(kra))

        # Code Branch __________________________________________________________________________________________________________
        self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        code_input_f2_xyz = Input(shape=(None,), name="CODE_TOKEN")

        code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
                                          input_length=None
                                          )(code_input_f2_xyz)

        code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        code = \
            Bidirectional(
                LSTM(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
                name='CODE_GRU_1'
            )(code_input)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)



        # Combined _____________________________________________________________________________________________________________
        #print("test code", code.shape)
        print("test bug", bug.shape)
        bug = Reshape((1, 64))(bug)
        code = Reshape((1, 64))(code)
        kra = Reshape((1, 64))(kra)

        print("NORM BEFORE: *******************************************************")
        ##bug = Activation('softmax')(bug)
        # code = Activation('softmax')(code)
        #img2 = Activation('softmax')(img2)
        # img3 = Activation('softmax')(img3)
        kra = Activation('softmax')(kra)

        print("CODE: *******************************************************")
        ## attCode = Attention()([bug, code])
        attCode = attention_distribution(bug, code)
        print("att code", attCode.shape, code.shape)
        sim_bug_code = Multiply()([attCode, code])
        print("att code", sim_bug_code.shape, attCode.shape, code.shape)

        print("KRA: *******************************************************")
        # attCode = Attention()([bug, code])
        attKRA = attention_distribution(bug, kra)
        print("att code", attKRA.shape, kra.shape)
        sim_bug_kra = Multiply()([attKRA, kra])
        print("att code", sim_bug_kra.shape, attKRA.shape, kra.shape)

        #print("IMG 2: *******************************************************")
        # attImg2 = Attention()([bug, img2])
        #attImg2 = attention_distribution(bug, img2)
        #print("att img", attImg2.shape, img2.shape)
        #sim_bug_img2 = Multiply()([attImg2, img2])
        #print("att img", sim_bug_img2.shape, attImg2.shape, img2.shape)
        ## print("sim_bug_code:", sim_bug_code.shape)
        #print("sim_bug_img:", sim_bug_img2.shape)

        print("COMBINE: *******************************************************")
        # sim_img_combined = Add()([sim_bug_code, sim_bug_img2])
        # sim_img_combined = Activation('softmax')(sim_img_combined)
        # sim_img_combined = Add()([sim_bug_code, sim_bug_img2])
        # sim_img_combined = Activation('softmax')(sim_img_combined)

        #combined = Concatenate(axis=-2)([bug, sim_bug_img2])  # , sim_combined3])  #
        combined = Concatenate(axis=-2)([bug, kra, sim_bug_kra, code, sim_bug_code])  # , sim_combined3])  #
        combined = Flatten()(combined)
        # combined = Dropout(rate=.2)(combined)
        print("All Combined:", combined.shape)

        print("DECISION: *******************************************************")

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)
        print("binary", binary.shape)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz, kra_input], outputs=binary)
        #self.model.summary()

        #return model
        print("Done Building Model.")

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


class KRA_Model_B1_FRAGE:

    def __init__(self, config_file):

        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())

        #https: // stats.stackexchange.com / questions / 273486 / network - in -network - in -keras - implementation

        # KRA Code Branch
        kra_input = Input(shape=(2054, 9), name='kra_input')
        kra = Conv1D(filters=32, kernel_size=7, activation='relu')(kra_input)
        kra = MaxPool1D(pool_size=4, strides=3)(kra)
        kra = Conv1D(filters=32, kernel_size=7, activation='relu')(kra)
        kra = MaxPool1D(pool_size=2, strides=2)(kra)
        kra = Conv1D(filters=64, kernel_size=1, activation='relu')(kra)
        kra = GlobalAveragePooling1D()(kra)


        #kra_conv = Conv1D(filters=192, kernel_size=5, activation='relu')(kra_input)
        #kra_conv = Conv1D(filters=160, kernel_size=1, strides=1, activation='relu')(kra_conv)
        #kra_conv = Conv1D(filters=96, kernel_size=1, strides=1, activation='relu')(kra_conv)
        #kra_conv = MaxPool1D(pool_size=3, strides=2)(kra_conv)
        #kra_conv = Dropout(0.7)(kra_conv)

        #kra_conv = Conv1D(filters=192, kernel_size=5, activation='relu')(kra_conv)
        #kra_conv = Conv1D(filters=192, kernel_size=1, strides=1, activation='relu')(kra_conv)
        #kra_conv = Conv1D(filters=192, kernel_size=1, strides=1, activation='relu')(kra_conv)
        #kra_conv = MaxPool1D(pool_size=3, strides=2)(kra_conv)
        #kra_conv = Dropout(0.7)(kra_conv)

        #kra_conv = Conv1D(filters=192, kernel_size=3, activation='relu')(kra_conv)
        #kra_conv = Conv1D(filters=192, kernel_size=1, strides=1, activation='relu')(kra_conv)
        #kra_conv = Conv1D(filters=64, kernel_size=1, strides=1, activation='relu')(kra_conv)

        #kra = GlobalAveragePooling1D()(kra_conv)
        print("KRA Shape", K.shape(kra))

        # Code Branch __________________________________________________________________________________________________________
        #self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        #self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        #self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        #self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        #self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        #code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        #code_input_f2_xyz = Input(shape=(None,), name="CODE_TOKEN")

        #code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
        #                                  input_length=None
        #                                  )(code_input_f2_xyz)

        #code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        #code = \
        #    Bidirectional(
        #        LSTM(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
        #        name='CODE_GRU_1'
        #    )(code_input)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)



        # Combined _____________________________________________________________________________________________________________
        #print("test code", code.shape)
        print("test bug", bug.shape)
        bug = Reshape((1, 64))(bug)
        # code = Reshape((1, 64))(code)
        kra = Reshape((1, 64))(kra)

        print("NORM BEFORE: *******************************************************")
        ##bug = Activation('softmax')(bug)
        # code = Activation('softmax')(code)
        #img2 = Activation('softmax')(img2)
        # img3 = Activation('softmax')(img3)
        kra = Activation('softmax')(kra)

        print("CODE: *******************************************************")
        ## attCode = Attention()([bug, code])
        #attCode = attention_distribution(bug, code)
        #print("att code", attCode.shape, code.shape)
        #sim_bug_code = Multiply()([attCode, code])
        #print("att code", sim_bug_code.shape, attCode.shape, code.shape)

        print("KRA: *******************************************************")
        # attCode = Attention()([bug, code])
        attKRA = attention_distribution(bug, kra)
        print("att code", attKRA.shape, kra.shape)
        sim_bug_kra = Multiply()([attKRA, kra])
        print("att code", sim_bug_kra.shape, attKRA.shape, kra.shape)

        #print("IMG 2: *******************************************************")
        # attImg2 = Attention()([bug, img2])
        #attImg2 = attention_distribution(bug, img2)
        #print("att img", attImg2.shape, img2.shape)
        #sim_bug_img2 = Multiply()([attImg2, img2])
        #print("att img", sim_bug_img2.shape, attImg2.shape, img2.shape)
        ## print("sim_bug_code:", sim_bug_code.shape)
        #print("sim_bug_img:", sim_bug_img2.shape)

        print("COMBINE: *******************************************************")
        # sim_img_combined = Add()([sim_bug_code, sim_bug_img2])
        # sim_img_combined = Activation('softmax')(sim_img_combined)
        # sim_img_combined = Add()([sim_bug_code, sim_bug_img2])
        # sim_img_combined = Activation('softmax')(sim_img_combined)

        #combined = Concatenate(axis=-2)([bug, sim_bug_img2])  # , sim_combined3])  #
        combined = Concatenate(axis=-2)([bug, kra, sim_bug_kra])  # , sim_combined3])  #
        combined = Flatten()(combined)
        # combined = Dropout(rate=.2)(combined)
        print("All Combined:", combined.shape)

        print("DECISION: *******************************************************")

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)
        print("binary", binary.shape)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_input], outputs=binary)
        #self.model.summary()

        #return model
        print("Done Building Model.")

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

class KRA_Model_B1_FRAGE2D:

    def __init__(self, config_file):

        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())

        #https: // stats.stackexchange.com / questions / 273486 / network - in -network - in -keras - implementation

        # KRA Code Branch
        kra_input = Input(shape=(2054, 9), name='kra_input')
        kra = Reshape((2054, 9, 1))(kra_input)
        kra = Conv2D(filters=64, kernel_size=(4, 9), strides=3, data_format="channels_last", activation='relu')(kra)
        # out: [None, 694, 1, 64]
        kra = Permute((3, 1, 2))(kra)
        kra = TimeDistributed(Flatten())(kra)
        kra = Permute((2, 1))(kra)
        kra = MaxPool1D(pool_size=2)(kra)
        kra = Conv1D(filters=32, kernel_size=7, activation='relu')(kra)
        kra = MaxPool1D(pool_size=4, strides=3)(kra)
        kra = Conv1D(filters=32, kernel_size=7, activation='relu')(kra)
        kra = MaxPool1D(pool_size=2, strides=2)(kra)
        kra = Conv1D(filters=64, kernel_size=1, activation='relu')(kra)
        kra = GlobalAveragePooling1D()(kra)


        print("KRA Shape", K.shape(kra))

        # Code Branch __________________________________________________________________________________________________________
        #self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        #self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        #self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        #self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        #self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        #code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        #code_input_f2_xyz = Input(shape=(None,), name="CODE_TOKEN")

        #code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
        #                                  input_length=None
        #                                  )(code_input_f2_xyz)

        #code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        #code = \
        #    Bidirectional(
        #        LSTM(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
        #        name='CODE_GRU_1'
        #    )(code_input)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)



        # Combined _____________________________________________________________________________________________________________
        #print("test code", code.shape)
        print("test bug", bug.shape)
        bug = Reshape((1, 64))(bug)
        # code = Reshape((1, 64))(code)
        kra = Reshape((1, 64))(kra)

        print("NORM BEFORE: *******************************************************")
        ##bug = Activation('softmax')(bug)
        # code = Activation('softmax')(code)
        #img2 = Activation('softmax')(img2)
        # img3 = Activation('softmax')(img3)
        kra = Activation('softmax')(kra)

        print("CODE: *******************************************************")
        ## attCode = Attention()([bug, code])
        #attCode = attention_distribution(bug, code)
        #print("att code", attCode.shape, code.shape)
        #sim_bug_code = Multiply()([attCode, code])
        #print("att code", sim_bug_code.shape, attCode.shape, code.shape)

        print("KRA: *******************************************************")
        # attCode = Attention()([bug, code])
        attKRA = attention_distribution(bug, kra)
        print("att code", attKRA.shape, kra.shape)
        sim_bug_kra = Multiply()([attKRA, kra])
        print("att code", sim_bug_kra.shape, attKRA.shape, kra.shape)

        #print("IMG 2: *******************************************************")
        # attImg2 = Attention()([bug, img2])
        #attImg2 = attention_distribution(bug, img2)
        #print("att img", attImg2.shape, img2.shape)
        #sim_bug_img2 = Multiply()([attImg2, img2])
        #print("att img", sim_bug_img2.shape, attImg2.shape, img2.shape)
        ## print("sim_bug_code:", sim_bug_code.shape)
        #print("sim_bug_img:", sim_bug_img2.shape)

        print("COMBINE: *******************************************************")
        # sim_img_combined = Add()([sim_bug_code, sim_bug_img2])
        # sim_img_combined = Activation('softmax')(sim_img_combined)
        # sim_img_combined = Add()([sim_bug_code, sim_bug_img2])
        # sim_img_combined = Activation('softmax')(sim_img_combined)

        #combined = Concatenate(axis=-2)([bug, sim_bug_img2])  # , sim_combined3])  #
        combined = Concatenate(axis=-2)([bug, kra, sim_bug_kra])  # , sim_combined3])  #
        combined = Flatten()(combined)
        # combined = Dropout(rate=.2)(combined)
        print("All Combined:", combined.shape)

        print("DECISION: *******************************************************")

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)
        print("binary", binary.shape)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_input], outputs=binary)
        #self.model.summary()

        #return model
        print("Done Building Model.")

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

class KRA_Model_CODE_FRAGE2D:

    def __init__(self, config_file):

        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())

        #https: // stats.stackexchange.com / questions / 273486 / network - in -network - in -keras - implementation
        # KRA Code Branch
        kra_input = Input(shape=(2054, 9), name='kra_input')
        kra = Reshape((2054, 9, 1))(kra_input)
        kra = Conv2D(filters=64, kernel_size=(4, 9), strides=3, data_format="channels_last", activation='relu')(kra)
        # out: [None, 694, 1, 64]
        kra = Permute((3, 1, 2))(kra)
        kra = TimeDistributed(Flatten())(kra)
        kra = Permute((2, 1))(kra)
        kra = MaxPool1D(pool_size=2)(kra)
        kra = Conv1D(filters=32, kernel_size=7, activation='relu')(kra)
        kra = MaxPool1D(pool_size=4, strides=3)(kra)
        kra = Conv1D(filters=32, kernel_size=7, activation='relu')(kra)
        kra = MaxPool1D(pool_size=2, strides=2)(kra)
        kra = Conv1D(filters=64, kernel_size=1, activation='relu')(kra)
        kra = GlobalAveragePooling1D()(kra)
        print("KRA Shape", K.shape(kra))

        # Code Branch __________________________________________________________________________________________________________
        self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        code_input_f2_xyz = Input(shape=(None,), name="CODE_TOKEN")

        code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
                                          input_length=None
                                          )(code_input_f2_xyz)

        code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        code = \
            Bidirectional(
                LSTM(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
                name='CODE_GRU_1'
            )(code_input)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = \
            Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, dropout=config["dropout"], name='BUG_GRU_1')
            )(bug_input)



        # Combined _____________________________________________________________________________________________________________
        #print("test code", code.shape)
        print("test bug", bug.shape)
        bug = Reshape((1, 64))(bug)
        code = Reshape((1, 64))(code)
        kra = Reshape((1, 64))(kra)

        print("NORM BEFORE: *******************************************************")
        ##bug = Activation('softmax')(bug)
        # code = Activation('softmax')(code)
        #img2 = Activation('softmax')(img2)
        # img3 = Activation('softmax')(img3)
        kra = Activation('softmax')(kra)

        print("CODE: *******************************************************")
        ## attCode = Attention()([bug, code])
        attCode = attention_distribution(bug, code)
        print("att code", attCode.shape, code.shape)
        sim_bug_code = Multiply()([attCode, code])
        print("att code", sim_bug_code.shape, attCode.shape, code.shape)

        print("KRA: *******************************************************")
        # attCode = Attention()([bug, code])
        attKRA = attention_distribution(bug, kra)
        print("att code", attKRA.shape, kra.shape)
        sim_bug_kra = Multiply()([attKRA, kra])
        print("att code", sim_bug_kra.shape, attKRA.shape, kra.shape)

        #print("IMG 2: *******************************************************")
        # attImg2 = Attention()([bug, img2])
        #attImg2 = attention_distribution(bug, img2)
        #print("att img", attImg2.shape, img2.shape)
        #sim_bug_img2 = Multiply()([attImg2, img2])
        #print("att img", sim_bug_img2.shape, attImg2.shape, img2.shape)
        ## print("sim_bug_code:", sim_bug_code.shape)
        #print("sim_bug_img:", sim_bug_img2.shape)

        print("COMBINE: *******************************************************")
        # sim_img_combined = Add()([sim_bug_code, sim_bug_img2])
        # sim_img_combined = Activation('softmax')(sim_img_combined)
        # sim_img_combined = Add()([sim_bug_code, sim_bug_img2])
        # sim_img_combined = Activation('softmax')(sim_img_combined)

        #combined = Concatenate(axis=-2)([bug, sim_bug_img2])  # , sim_combined3])  #
        combined = Concatenate(axis=-2)([bug, kra, sim_bug_kra, code, sim_bug_code])  # , sim_combined3])  #
        combined = Flatten()(combined)
        # combined = Dropout(rate=.2)(combined)
        print("All Combined:", combined.shape)

        print("DECISION: *******************************************************")

        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)
        print("binary", binary.shape)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, code_input_f1_cat, code_input_f2_xyz, kra_input], outputs=binary)
        #self.model.summary()

        #return model
        print("Done Building Model.")

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

class Base_Model_KRA_SEQ_OL_L3:

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


        # Code Branch __________________________________________________________________________________________________________
        #self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        #self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        #self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        #self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        #self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        #code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        #code_input_f2_xyz = Input(shape=(None, ), name="CODE_TOKEN")

        #code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
        #    input_length=None
        #    )(code_input_f2_xyz)

        #code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        #code = \
        #    Bidirectional(
        #        GRU(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
        #        name='CODE_GRU_1'
        #    )(code_input)


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
        #print("test code", code.shape)
        #print("test bug", bug.shape)
        #code = Reshape((1, 64))(code)
        bug = Reshape((1, 64), name="xxx1")(bug)
        kra_objects = Reshape((1, 64), name="xxx2")(kra_objects)
        kra_levels = Reshape((1, 64), name="xxx3")(kra_levels)

        print("NORM BEFORE: *******************************************************")
        ##bug = Activation('softmax')(bug)
        #code = Activation('softmax')(code)
        #img2 = Activation('softmax')(img2)
        #img3 = Activation('softmax')(img3)
        kra_objects = Activation('softmax')(kra_objects)
        kra_levels = Activation('softmax')(kra_levels)

        print("KRA OBJECTS *******************************************************")
        attKraObjects = attention_distribution(bug, kra_objects)
        sim_bug_kra_objects = Multiply()([attKraObjects, kra_objects])

        print("KRA LEVELS *******************************************************")
        attKraLevels = attention_distribution(bug, kra_levels)
        sim_bug_kra_levels = Multiply()([attKraObjects, kra_levels])

        #print("CODE: *******************************************************")
        ##attCode = Attention()([bug, code])
        #attCode = attention_distribution(bug, code)
        #print("att code", attCode.shape, code.shape)
        #sim_bug_code = Multiply()([attCode, code])
        #print("att code", sim_bug_code.shape, attCode.shape, code.shape)

        #print("IMG 2: *******************************************************")
        ##attImg2 = Attention()([bug, img2])
        #attImg2 = attention_distribution(bug, img2)
        #print("att img", attImg2.shape, img2.shape)
        #sim_bug_img2 = Multiply()([attImg2, img2])
        #print("att img", sim_bug_img2.shape, attImg2.shape, img2.shape)
        #print("sim_bug_code:", sim_bug_code.shape)
        #print("sim_bug_img:", sim_bug_img2.shape)

        #print("IMG 3: *******************************************************")
        ##attImg3 = Attention()([bug, img3])
        #attImg3 = attention_distribution(bug, img3)
        #print("att img", attImg3.shape, img3.shape)
        #sim_bug_img3 = Multiply()([attImg3, img3])
        #print("att img", sim_bug_img3.shape, attImg3.shape, img3.shape)
        #print("sim_bug_code:", sim_bug_code.shape)
        #print("sim_bug_img:", sim_bug_img3.shape)

        #print("NORM AFTER: *******************************************************")
        #sim_bug_code = Activation('softmax')(sim_bug_code)
        #sim_bug_img2 = Activation('softmax')(sim_bug_img2)
        #sim_bug_img3 = Activation('softmax')(sim_bug_img3)

        #sim_img_combined = Add()([sim_bug_code, sim_bug_img2])

        #sim_img_combined = Activation('softmax')(sim_img_combined)

        combined = Concatenate(axis=-2)([bug, sim_bug_kra_objects, sim_bug_kra_levels])  #, sim_img_combined])  #, sim_combined3])  #
        combined = Flatten()(combined)

        # >>>>>>>>>>>> combined = Dropout(rate=.5)(combined)
        print("All Combined:", combined.shape)
        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)  ##############, W_regularizer=l2(0.01))(combined)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=self.configs["loss"], metrics=['accuracy'])  #############3, tp_rate, tn_rate])
        #self.model.summary()

        #return model
        print("Done Building Model.")

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

class Base_Model_KRA_SEQ_OL_L2:

    def __init__(self, config_file):

        self.config_file = config_file
        if not os.path.exists(config_file):
            print("Error, Experiment config file: " + config_file + " does not exist.")
            sys.exit(1)
        with open(self.config_file) as f:
            config = json.loads(f.read())

        choose = Feed_Type()
        self.feed_type = choose.feed_CAR
        # KRA SEQ OBJECTS
        gru_units = config["gru_units"]
        gru_units = 32
        kra_objects_input = Input(shape=(None, ), name="KRA_SEQ_OBJECTS_INPUT")
        kra_objects = Embedding(config["kra_objects_features"],
                            config["kra_objects_embedding"],
                            input_length=None)(kra_objects_input)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=True), name="KRA_SEQ_OBJECTS_GRU_1"
        )(kra_objects)
        kra_objects = Dropout(rate=0.2)(kra_objects)
        #kra_objects = Bidirectional(
        #    GRU(units=gru_units, return_sequences=True), name="KRA_SEQ_OBJECTS_GRU_2"
        #)(kra_objects)
        #kra_objects = Dropout(rate=0.2)(kra_objects)
        kra_objects = Bidirectional(
            GRU(units=gru_units, return_sequences=False), name="KRA_SEQ_OBJECTS_GRU_3"
        )(kra_objects)


        # KRA SEQ LEVELS
        kra_levels_input = Input(shape=(None, ), name="KRA_SEQ_LEVELS_INPUT")
        kra_levels = Embedding(config["kra_levels_features"],
                            config["kra_levels_embedding"],
                            input_length=None)(kra_levels_input)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=True), name="KRA_SEQ_LEVELS_GRU_1"
        )(kra_levels)
        kra_levels = Dropout(rate=0.2)(kra_levels)
        #kra_levels = Bidirectional(
        #    GRU(units=gru_units, return_sequences=True), name="KRA_SEQ_LEVELS_GRU_2"
        #)(kra_levels)
        #kra_levels = Dropout(rate=0.2)(kra_levels)
        kra_levels = Bidirectional(
            GRU(units=gru_units, return_sequences=False), name="KRA_SEQ_LEVELS_GRU_3"
        )(kra_levels)


        # Code Branch __________________________________________________________________________________________________________
        #self.hp_code_dim1_max_word_features = int(config['code_dim1_max_word_features'])
        #self.hp_code_dim2_max_word_features = int(config['code_dim2_max_word_features'])
        #self.hp_code_embedding_dim_1 = int(config["code_embedding_d1_cat"])
        #self.hp_code_embedding_dim_2 = int(config["code_embedding_d2_token"])
        #self.hp_code_gru_latent_dim = int(config["code_gru_units"])

        # Variable Sequence Length, shape=(None...
        # Code Dim 1 is one-hot-encoded
        # Code Dim 2 is vectorized, (NOT one-hot encoded)
        #code_input_f1_cat = Input(shape=(None, self.hp_code_dim1_max_word_features), name="CODE_TYPE")
        #code_input_f2_xyz = Input(shape=(None, ), name="CODE_TOKEN")

        #code_embedding_f2_xyz = Embedding(self.hp_code_dim2_max_word_features, self.hp_code_embedding_dim_2,
        #    input_length=None
        #    )(code_input_f2_xyz)

        #code_input = Concatenate()([code_input_f1_cat, code_embedding_f2_xyz])

        # Encode the code information (dim1+dim2)
        # GRU INPUT: (None, None, 323) == (Batch, Steps, Features) == (B, S, F)
        #code = \
        #    Bidirectional(
        #        GRU(units=self.hp_code_gru_latent_dim, return_sequences=False, dropout=config["dropout"]),
        #        name='CODE_GRU_1'
        #    )(code_input)


        # Bug Branch ___________________________________________________________________________________________________________
        self.hp_bug_embedding_dim = int(config['bug_embedding'])
        self.hp_bug_gru_latent_dim = int(config["bug_gru_units"])

        # Bug input has bp_bug_embedding_dim=300 features create by GLOVE
        bug_input = Input(shape=(None, self.hp_bug_embedding_dim), name="NL_BUG_TEXT")
        bug = Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=True, name='BUG_GRU_1')
            )(bug_input)
        bug = Dropout(rate=0.20)(bug)
        #bug = Bidirectional(
        #    GRU(self.hp_bug_gru_latent_dim, return_sequences=True, name='BUG_GRU_2')
        #)(bug)
        #bug = Dropout(rate=0.20)(bug)
        bug = Bidirectional(
                GRU(self.hp_bug_gru_latent_dim, return_sequences=False, name='BUG_GRU_3')
            )(bug)


        # Combined _____________________________________________________________________________________________________________
        #print("test code", code.shape)
        #print("test bug", bug.shape)
        #code = Reshape((1, 64))(code)
        bug = Reshape((1, 64), name="xxx1")(bug)
        kra_objects = Reshape((1, 64), name="xxx2")(kra_objects)
        kra_levels = Reshape((1, 64), name="xxx3")(kra_levels)

        print("NORM BEFORE: *******************************************************")
        ##bug = Activation('softmax')(bug)
        #code = Activation('softmax')(code)
        #img2 = Activation('softmax')(img2)
        #img3 = Activation('softmax')(img3)
        kra_objects = Activation('softmax')(kra_objects)
        kra_levels = Activation('softmax')(kra_levels)

        print("KRA OBJECTS *******************************************************")
        attKraObjects = attention_distribution(bug, kra_objects)
        sim_bug_kra_objects = Multiply()([attKraObjects, kra_objects])

        print("KRA LEVELS *******************************************************")
        attKraLevels = attention_distribution(bug, kra_levels)
        sim_bug_kra_levels = Multiply()([attKraObjects, kra_levels])

        #print("CODE: *******************************************************")
        ##attCode = Attention()([bug, code])
        #attCode = attention_distribution(bug, code)
        #print("att code", attCode.shape, code.shape)
        #sim_bug_code = Multiply()([attCode, code])
        #print("att code", sim_bug_code.shape, attCode.shape, code.shape)

        #print("IMG 2: *******************************************************")
        ##attImg2 = Attention()([bug, img2])
        #attImg2 = attention_distribution(bug, img2)
        #print("att img", attImg2.shape, img2.shape)
        #sim_bug_img2 = Multiply()([attImg2, img2])
        #print("att img", sim_bug_img2.shape, attImg2.shape, img2.shape)
        #print("sim_bug_code:", sim_bug_code.shape)
        #print("sim_bug_img:", sim_bug_img2.shape)

        #print("IMG 3: *******************************************************")
        ##attImg3 = Attention()([bug, img3])
        #attImg3 = attention_distribution(bug, img3)
        #print("att img", attImg3.shape, img3.shape)
        #sim_bug_img3 = Multiply()([attImg3, img3])
        #print("att img", sim_bug_img3.shape, attImg3.shape, img3.shape)
        #print("sim_bug_code:", sim_bug_code.shape)
        #print("sim_bug_img:", sim_bug_img3.shape)

        #print("NORM AFTER: *******************************************************")
        #sim_bug_code = Activation('softmax')(sim_bug_code)
        #sim_bug_img2 = Activation('softmax')(sim_bug_img2)
        #sim_bug_img3 = Activation('softmax')(sim_bug_img3)

        #sim_img_combined = Add()([sim_bug_code, sim_bug_img2])

        #sim_img_combined = Activation('softmax')(sim_img_combined)

        combined = Concatenate(axis=-2)([bug, sim_bug_kra_objects, sim_bug_kra_levels])  #, sim_img_combined])  #, sim_combined3])  #
        combined = Flatten()(combined)

        # >>>>>>>>>>>> combined = Dropout(rate=.5)(combined)
        print("All Combined:", combined.shape)
        binary = Dense(1, activation='sigmoid', name='BINARY_OUTPUT')(combined)  ##############, W_regularizer=l2(0.01))(combined)

        # Model: _______________________________________________________________________________________________________________
        self.model = Model(inputs=[bug_input, kra_objects_input, kra_levels_input], outputs=binary)
        self.model.compile(optimizer=RMSprop(lr=0.0005), loss=config["loss"], metrics=['accuracy'])  #############3, tp_rate, tn_rate])
        #self.model.summary()

        #return model
        print("Done Building Model.")

    #def gen_mode(self):
    #    return "kra_seqO_seqL"

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


