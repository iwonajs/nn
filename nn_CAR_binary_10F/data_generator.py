import tensorflow as tf
from tensorflow.keras.models import Sequential
#from tensorflow import keras
#import keras
import json
import numpy as np
from numpy import linalg as LA
import os
import models as m
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
    def __init__(self, sample_ids, observation_key_triplets, feed_flag, configs, shuffle=True):
        #self.imgs = load(img_mat_file)['arr_0']
        self.configs = configs
        #self.bugDir = configs["bugs_path"]
        #self.codeDir = configs["code_path"]
        #self.imgDir = configs["image_path"]
        self.feed_flag = feed_flag
        self.feed_flag_options = m.Feed_Type()
        #self.img = useImgFlag
        #if not self.imgDir is None and len(self.imgDir) > 0:
        #    self.img = True
        #self.imgs_map = img_map
        #self.datagen = ImageDataGenerator(rescale=1/255)
        self.shuffle = shuffle
        self.observation_key_triplets = observation_key_triplets
        self.sample_ids = sample_ids
        self.cases = len(self.sample_ids)
        self.batch_size = 1

        #self.code_vectorized_dir = code_vectorized_dir
        #self.bug_embedded_dir = bug_embedded_dir

        #with open(jsonConfigs, 'r') as f:
        #    body = json.loads(f.read())
        # self.code_time_steps = body["code_time_steps"]
        # self.bug_time_steps = body["bug_time_steps"]
        self.bug_embedding = configs["bug_embedding"]
        self.code_dim1_max_word_features = configs["code_dim1_max_word_features"]
        self.code_dim2_max_word_features = configs["code_dim2_max_word_features"]
        self.img_path = configs["image_path"]
        #self.code_max_word_features_classes = body["code_max_word_features_class"]
        #self.code_max_word_features_package = body["code_max_word_features_package"]
        #self.code_max_word_features_extends = body["code_max_word_features_extends"]
        #self.code_max_word_features_implements = body["code_max_word_features_implements"]

        self.on_epoch_end()

    def on_epoch_end(self):

        self.gen_indices = np.arange(self.cases)
        np.random.shuffle(self.gen_indices)

    #def reverse_dict_id_fileid(observations):
    #    reversed_dict = {}
    #    for obs in observations:
    #
    #    return {}

    # Number of Batches in the Epoch
    # (Sequence) / (Batch Size)
    def __len__(self):
        return self.cases

    # A batch at position index
    def __getitem__(self, batch_index):

        # Select Data for the Batch
        start = batch_index * self.batch_size
        end = (batch_index + 1) * self.batch_size
        batch_indices = self.gen_indices[start:end]
        batch_ids = [self.sample_ids[i] for i in batch_indices]
        batch_keys = [self.observation_key_triplets[id] for id in batch_ids]
        # Load for batch_obs
        # there should be exactly one observation
        keys = batch_keys[0]
        method_id = keys[0]
        bug_id = keys[1]
        r = keys[2]

        #print("method_id", method_id)
        #print("bug_id", bug_id)
        #exit(1)


        #class_mat = self.load_class(method_id, self.code_max_word_features_classes, self.code_vectorized_dir)
        #package_mat = self.load_package(method_id, self.code_max_word_features_package, self.code_vectorized_dir)
        #extends_mat = self.load_extends(method_id, self.code_max_word_features_extends, self.code_vectorized_dir)
        #implements_mat = self.load_implements(method_id, self.code_max_word_features_implements, self.code_vectorized_dir)
        bug_mat = self.load_bug(bug_id)

        label = np.array([r])
        #print("iwona test *****************")
        #print("bug_mat", bug_mat.shape)
        #print("method_category_mat", method_category_mat.shape)
        #print("method_token_mat",method_token_mat.shape)
        #return [bug_mat, method_category_mat, method_token_mat, method_img], label.reshape(1, *label.shape)
        #if self.img:
        #    method_img_mat = self.load_method_img(method_id)
        #    return [bug_mat, method_category_mat, method_token_mat, method_img_mat], label.reshape(1, *label.shape)
        if self.feed_flag == self.feed_flag_options.feed_CAR:
            #method_category_mat, method_token_mat = self.load_method(method_id)
            seqO = self.load_kra_seqO(method_id)
            seqL = self.load_kra_seqL(method_id)
            return [bug_mat, seqO, seqL], label.reshape(1, *label.shape)
        else:
            #method_category_mat, method_token_mat = self.load_method(method_id)
            print("Unrecognized Feed", self.feed_flag)
            exit(1)
        #    return [bug_mat, method_category_mat, method_token_mat], label.reshape(1, *label.shape)
        #return method_category_mat, label.reshape(1, *label.shape)

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
        return seq.reshape(1, *seq.shape)

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

    results_dir = "/media/iwona/Optane/Project_BugLoc/compute_canada/exp01/"
    exp_folder = "run00"
    config = json.load(open("config.json", "r"))
    folds_data_path = config["folds_data_path"]
    with open(folds_data_path) as json_file:
        fold_data = json.load(json_file)


    train_indices = []
    validation_indices = []
    observations_key_triplets = []
    id = 0
    for k, v in fold_data.items():
        observations_key_triplets += [[v["cfm_npz"], v["bug"], v["pos"]]]
        train_indices += [id]
        id += 1
        file_name = v["cfm_npz"]
        file_name = os.path.splitext(file_name)[0] + ".modified2.javaparser.varType.npy"
        t = os.path.isfile(config["code_path"] + "vectorized/" + file_name)
    bugs_path = config["bugs_path"]
    code_path = config["code_path"]
    img_path = config["image_path"]
    nn = m.Base_Model_KRA_SEQ_OL_L3(config)
    gen = DeepTrace_DataGenerator(train_indices,
                                  observations_key_triplets,
                                  nn.feed_flag,
                                  config,
                                  shuffle=True)

    shape_check = []
    shape_check_small = []
    count = 0
    for batch in gen:
        count += 1
        print("Batch ***************************************************************** ", count)
        print("batch len", len(batch))
        dim = len(batch)
        for di in range(dim):
            print(">>>> inputs ", len(batch[di]))
            for dij in range(len(batch[di])):
                print("item:", dij, batch[di][dij].shape)
    print("")
                #p
            #d = batch[id].shape

        #print("img shape:", batch[0][3].shape)
        #img_shape = batch[0][3].shape
        #if img_shape[3] > 3:
        #    print("error")
        #    exit(1)
        #if img_shape[1]*img_shape[2] > 150000:
        #    shape_check += [img_shape]
        #else:
        #    shape_check_small += [img_shape]

    #for i in shape_check:
    #    print(i)
    print("big: ", len(shape_check))
    print("small: ", len(shape_check_small))
        #for input_i in batch[0]:
        #    print("input: ", input_i.shape)
            #r = input("show image y/n:")
            #if r == "y":
            #    img_pil = array_to_img(input_i, dtype='uint8')
            #    img_pil.show()
        #print("SECOND...")
        #for input_i in batch[1]:
        #    print("input: ", input_i.shape)
        #    print(input_i)
        #print("END...............................................")

        #bug_batch_steps_embedding = batch[0][0]
        #batchx, stepsx, embeddingx = bug_batch_steps_embedding.shape
        #issue = np.sum(np.isinf(bug_batch_steps_embedding))
        #if issue:
        #    resp = input("Issue Found in:")
        #else:
        #    print("Bug has no +/- inf")

        #for s in range(stepsx):
        #    embedding_norm = LA.norm(bug_batch_steps_embedding[0][s][:])
        #    #print(embedding_norm.shape, embedding_norm)
        #    if not np.isclose(embedding_norm, 1, atol=0.000001):
        #        issue = input("Norm not close to 1.")
        #print("output:", batch[1])

    #print("observation count:", len(train_indices))

