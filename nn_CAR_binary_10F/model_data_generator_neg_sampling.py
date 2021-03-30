import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing as pp
from tensorflow.keras.preprocessing import sequence as pps
from tensorflow.keras.models import Sequential
#from tensorflow import keras
#import keras
import json
import numpy as np
from numpy import linalg as LA
import os
import models as m
from numpy import random
import sys
import math
from numpy import load
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import array_to_img
import tensorflow.keras.preprocessing.image as keras_img
#import PIL
# References:
#https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class DeepTrace_DataGenerator(tf.keras.utils.Sequence):
    # posObs[id] = [BUG, FILE], all possible positive cases
    # negObs[epoch] = negObs_list[id] = [BUG, FILE], sampled negative cases per epoch
    # split_indices: a subset of possible indices of positive cases
    def __init__(self, configs, data_source, feed_flag, validation_fold=-1, train_mode=True, include_cat=[0, 1]):

        self.configs = configs

        ##########################################################
        # Once self.data is set the concept of folds is not needed
        self.data_source = data_source
        self.validation_fold = validation_fold
        self.train_mode = train_mode
        self.data = {}
        self.set_data()
        ##########################################################

        ##########################################################
        # create list of list for each epoch of negative samples
        # note that validation and test do not have use epochs
        # in that case the list will contain only one list
        # same as if it was one epoch
        self.bug_ids = []
        self.bug_prob = []
        self.set_bug_ids_and_bug_prob()

        # FEED CONFIGURATIONS
        self.feed_flag = feed_flag
        self.feed_flag_options = m.Feed_Type()

        self.include_cat = include_cat
        self.pos_triplets = []
        self.set_pos_triplets()
        # set on_epoch_end() as well but needs to be in case mode_train = false, validation
        self.neg_triplets = []
        self.set_neg_triplets()
        self.cases = len(self.pos_triplets) + len(self.neg_triplets)
        self.batch_size = configs['batch_size']
        self.shuffle = configs['shuffle']
        self.resample_neg_on_epoch = configs['resample_neg_on_epoch']
        # set on_epoch_end()
        self.observation_key_triplets = self.pos_triplets + self.neg_triplets
        self.on_epoch_end()

        self.bug_embedding = configs["bug_embedding"]
        self.code_dim1_max_word_features = configs["code_dim1_max_word_features"]
        self.code_dim2_max_word_features = configs["code_dim2_max_word_features"]

    def set_bug_ids_and_bug_prob(self):
        bug_fq = {}
        for key, obs in self.data.items():
            bug_id = obs['bug']
            if bug_id in bug_fq:
                bug_fq[bug_id] += 1
            else:
                bug_fq[bug_id] = 1

        self.bug_ids = []
        self.bug_ids_seq = []
        self.bug_prob = []
        obs_N = sum(bug_fq.values())
        for bug_id, bug_fq in bug_fq.items():
            self.bug_ids_seq += [str(bug_id)]
            self.bug_ids += [str(bug_id)]
            self.bug_prob += [bug_fq/obs_N]

    def set_data(self):
        self.data = {}
        new_key = 0
        # Training: Use All Folds but the validation_fold
        # For each Epoch Balance negative scenarios with positive
        # Sample possible Bug based on its frequency in the positive scenarios
        if self.validation_fold > -1 and self.train_mode and self.data_source == self.configs["train_pos_folds"]:
            with open(self.data_source) as f:
                data = json.load(f)
            for key, observation in data.items():
                if not observation['fold'] == self.validation_fold:
                    self.data[new_key] = observation
                    new_key += 1
        # Validation: Use only the validation_fold
        # Provide All possible Negative Scenarios
        elif self.validation_fold > -1 and not self.train_mode and self.data_source == self.configs["train_pos_folds"]:
            with open(self.data_source) as f:
                data = json.load(f)
            for key, observation in data.items():
                if observation['fold'] == self.validation_fold:
                    self.data[new_key] = observation
                    new_key += 1
        # Testing: Use all the training data
        # data_source should be "train_pos"
        # Provide All possible Negative Scenarios
        elif self.validation_fold < 0 and self.data_source == self.configs["test_pos"]:
            with open(self.data_source) as f:
                self.data = json.load(f)
        else:
            print("incorrect input combination")
            exit("13")

    def set_pos_triplets(self):
        self.pos_triplets = []
        for k, v in self.data.items():
            self.pos_triplets += [[v["cfm_npz"], v["bug"], v["pos"]]]

    def set_negative_triplets_testing(self):
        bug_dict = dict()
        method_list = []
        for k, v in self.data.items():
            method = v["cfm_npz"]
            if not method in method_list:
                method_list.append(method)
            method_index = method_list.index(method)
            bug = v["bug"]
            if bug in bug_dict:
                bug_dict[bug] += [method_index]
            else:
                bug_dict[bug] = [method_index]

        for bug, pos_list_index in bug_dict.items():

            neg_list = [i for i in method_list]
            for i in pos_list_index:
                neg_list.pop(i)
            assert len(pos_list_index) + len(neg_list) == len(method_list)
            for cfm_npz in neg_list:
                self.neg_triplets += [[cfm_npz, bug, 0]]

    def set_neg_triplets(self):
        self.neg_triplets = []
        if 0 in self.include_cat:
            # IF TESTING:
            if self.validation_fold < 0:  # TESTING
                self.set_negative_triplets_testing()
                return

            # ELSE (VALIDATION or TRAINING):
            self.neg_triplets = []
            for k, v in self.data.items():
                new_bug_id = -1
                while new_bug_id == -1 or new_bug_id == v['bug']:
                    #new_bug_id = random.choice(self.bug_ids, self.bug_prob)  # dim too large!
                    new_bug_id = random.choice(self.bug_ids_seq)
                self.neg_triplets += [[v["cfm_npz"], new_bug_id, 0]]

    def on_epoch_end(self):
        print("on epoch end.............", self.data_source)
        self.gen_indices = np.arange(self.cases)
        if self.shuffle:
            np.random.shuffle(self.gen_indices)
        if self.train_mode and self.resample_neg_on_epoch:
            self.set_neg_triplets()
            self.observation_key_triplets = self.pos_triplets + self.neg_triplets
            assert len(self.pos_triplets) == len(self.neg_triplets)

    def __len__(self):
        return self.cases

    # A batch at position index
    def __getitem__(self, batch_index):

        # Select Data for the Batch
        start = batch_index * self.batch_size
        end = (batch_index + 1) * self.batch_size
        batch_indices = self.gen_indices[start:end]

        batch_obs = [self.observation_key_triplets[id] for id in batch_indices]

        # Load for batch_obs
        # there should be exactly one observation
        for obs in batch_obs:
            #keys = batch_keys[0]
            method_id = obs[0]
            bug_id = obs[1]
            r = obs[2]

            bug_mat = self.load_bug(bug_id)
            label = np.array([r])

            if self.feed_flag == self.feed_flag_options.feed_CAR:
                #method_category_mat, method_token_mat = self.load_method(method_id)
                seqO = self.load_kra_seqO(method_id)
                seqL = self.load_kra_seqL(method_id)
                return [bug_mat, seqO, seqL], label.reshape(1, *label.shape)
            else:
                #method_category_mat, method_token_mat = self.load_method(method_id)
                print("Unrecognized Feed", self.feed_flag)
                exit(1)
    # Dimensionality of output; batch_size X seq_length X feature_dimensionality

    def load_kra_seqO(self, method_id):
        file_name = str(os.path.splitext(method_id)[0]).split(".")[0]
        file_path_tokens = [self.configs["kra_seqO_path"], file_name + ".npy"]
        seq = np.load(os.path.join(*file_path_tokens))
        return seq.reshape(1, *seq.shape)

    def load_kra_seqL(self, method_id):
        file_name = str(os.path.splitext(method_id)[0]).split(".")[0]
        file_path_tokens = [self.configs["kra_seqL_path"], file_name + ".npy"]
        seq = np.load(os.path.join(*file_path_tokens))
        #seq = pps.pad_sequences(seq, padding="post")
        seq = tf.keras.preprocessing.sequence.pad_sequences(seq.reshape(1, *seq.shape), padding='post', maxlen=200)
        return seq

    # https://www.tensorflow.org/guide/keras/masking_and_padding
    def load_method_img(self, method_id):
        #file_name = str(method_id).replace(".raw.npz", ".npy")
        #file_path = os.path.join(self.img_path, file_name)
        #img_mat = np.load(file_path)

        file_name = str(method_id)
        #file_path = os.path.join(self.img_path, file_name)
        file_path = os.path.join(self.imgDir, file_name)
        npzfile = np.load(file_path)
        img_mat = npzfile['arr_0']

        #img_mat = np.zeros((150, 150, 3), dtype=np.uint8)

        return img_mat.reshape(1, *img_mat.shape)

    def load_bug(self, bug_id):
        file_name = "gh-" + bug_id + ".npy"
        bug_embedded = np.load(os.path.join(self.configs["bugs_path"], "vectorized", file_name))
        dim = bug_embedded.shape
        if dim[0] > 100:
            bug_embedded = bug_embedded[0:500,:]
        bug_norm = LA.norm(bug_embedded, axis=1)
        bug_normalized = bug_embedded / bug_norm[:, None]
        return bug_normalized.reshape(1, *bug_normalized.shape)
    # /home/iwona/optane/msr_input/vectorized_trainedGloVe_300_paired/vectorized
    # /home/iwona/optane/msr_input/vectorized_trainedGloVe_300_paired/vectorized
    def load_method(self, method_id):
        file_name = os.path.splitext(method_id)[0]
        file_name = os.path.splitext(file_name)[0] + ".modified2.javaparser.varType.npy"
        code_matrix = np.load(os.path.join(self.codeDir, "vectorized", file_name))
        #code_categories = code_matrix[:, 0]
        code_categories = tf.keras.utils.to_categorical(code_matrix[:, 0], num_classes=self.code_dim1_max_word_features)
        code_tokens = code_matrix[:, 1]
        return code_categories.reshape(1, *code_categories.shape), code_tokens.reshape(1, *code_tokens.shape)

    def load_class(self, method_id, word_features):
        class_vectorized = np.load(os.path.join(self.codeDir, "context_metadata/class/", method_id))
        class_one_hot = tf.keras.utils.to_categorical(class_vectorized, word_features)
        return class_one_hot.reshape(1, *class_one_hot.shape)

    def load_package(self, method_id, word_features):
        package_vectorized = np.load(os.path.join(self.codeDir, "context_metadata/package/",  method_id))
        package_one_hot = tf.keras.utils.to_categorical(package_vectorized, word_features)
        return package_one_hot.reshape(1, *package_one_hot.shape)

    def load_extends(self, method_id, word_features):
        extends_vectorized = np.load(os.path(self.codeDir, "context_metadata/extends/", method_id))
        extends_one_hot = tf.keras.utils.to_categorical(extends_vectorized, word_features)
        return extends_one_hot.reshape(1, *extends_one_hot.shape)


    def load_implements(self, method_id, word_features):
        implements_vectorized = np.load(os.path.join(self.codeDir + "context_metadata/implements/" + method_id))
        implements_one_hot = tf.keras.utils.to_categorical(implements_vectorized, word_features)
        return implements_one_hot.reshape(1, *implements_one_hot.shape)



def load_observations_key_triplets(jsonfilePos, jsonfileNeg):

    key_triplets = []
    reverse_map_fileid = {}
    countPos = 0
    countNeg = 0

    with open(jsonfilePos, 'r') as pf:
        for pline in pf:
            body = json.loads(pline)
            key_triplets += [[body["method"], body["bug"], 1]]
            countPos += 1

            # Reverse Dict
            if body["method"] not in reverse_map_fileid:
                reverse_map_fileid[body["method"]] = body["id"]


    with open(jsonfileNeg, 'r') as nf:
        for nline in nf:
            body = json.loads(nline)
            key_triplets += [[body["method"], body["bug"], 0]]
            countNeg += 1

    return countPos, countNeg, key_triplets, reverse_map_fileid


if __name__ == "__main__":

    with open("config.json") as f:
        configs = json.load(f)
    # configs, data_source, feed_flag, validation_fold=-1, train_mode=True, shuffle=True
    gen_train = DeepTrace_DataGenerator(configs, configs["train_pos_folds"], "CAR",
                                  validation_fold=0,
                                  train_mode=True)
    for batch in gen_train:
        pass
    for batch in gen_train:
        pass
    print("Gen Train Pass ..................................................")

    gen_validate = DeepTrace_DataGenerator(configs, configs["train_pos_folds"], "CAR",
                                  validation_fold=0,
                                  train_mode=False)
    for batch in gen_validate:
        pass
    print("Gen Validation Pass .............................................")

    gen_test = DeepTrace_DataGenerator(configs, configs["test_pos"], "CAR",
                                  validation_fold=-1,
                                  train_mode=False)
    for batch in gen_test:
        pass
    print("Gen Test Pass ....................................................")

    for i, k in configs.items():
        print(i, k)

    exit(111)
    count = 0
    for batch in gen_test:
        count += 1
        print("Batch *************************************************", gen_test.cases, count)
        #print(batch)
        print("batch len", len(batch))
        dim = len(batch)
        for di in range(dim):
            print(">>>> inputs ", len(batch[di]))
            for dij in range(len(batch[di])):
               print("item:", dij, batch[di][dij].shape)

    for batch in gen_test:
        pass





