import json
import numpy as np
import os
import sys
import shutil
import time
import tensorflow as tf
import generator as gen
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
import tensorflow.keras as keras

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

def fold_x(mode, patience, results_path, exp_folder, folds_data_path, dtm, x):
    if not os.path.exists(results_path):
        os.makedirs(exp_folder)
    if results_path == ".":
        exp_path = os.path.join(exp_folder)
    else:
        exp_path = os.path.join(results_path, exp_folder)
    exp_path = os.path.join(results_path, exp_folder)

    print("Experiment Path:", exp_path)
    if os.path.exists(exp_path):
        print("Experiment folder already exists.", exp_path)
    else:
        os.makedirs(exp_path, exist_ok=True)

    model = dtm.model
    config_file = dtm.config_file
    if not os.path.exists(config_file):
        print("Error, Experiment config file: " + config_file + " does not exist.")
        sys.exit(1)
    with open(config_file) as f:
        config = json.loads(f.read())

    fold_name = "fold_" + str(x)
    fold_folder = fold_name
    fold_dir = os.path.join(results_path, exp_folder, fold_folder)
    fold_dir_exists = os.path.exists(fold_dir)
    if fold_dir_exists:
        # clear previous output files
        for f in os.listdir(fold_dir):
            os.remove(fold_dir + f)
    else:
        os.mkdir(fold_dir)

    out_csv = os.path.join(exp_path, fold_name + ".csv")  # config["logger_path"]
    logger = CSVLogger(out_csv)
    checkpoint_file = os.path.join(exp_path, fold_folder, config["callback_file"])
    chkpoint = ModelCheckpoint(checkpoint_file, verbose=1)

    #print("test: ", fold_dir_exists)
    #print("checkpoint: ", fold_dir)
    #print("logger: ", out_csv)
    #print("checkpoint: ", checkpoint_file)

    # load observation pairs
    #folds_data_path = results_path + "folds_pos_neg.json"
    with open(folds_data_path) as json_file:
        fold_data = json.load(json_file)
    train_indices = []
    validation_indices = []
    observations_key_triplets = []
    id = 0
    #sampleSizeTaken = 0
    for k, v in fold_data.items():
        observations_key_triplets += [[v["cfm_npz"], v["bug"], v["pos"]]]
        #if sampleSizeTaken > 10:
        #    continue
        #sampleSizeTaken += 1
        if v["fold"] == x:
            validation_indices += [id]
        else:
            train_indices += [id]
        id += 1

    # Init the Generator
    bugs_path = config["bugs_path"]
    code_path = config["code_path"]
    img_path = "/media/iwona/Optane/Project_BugLoc/compute_canada/input/images/"
    kra_path = "/media/iwona/Optane/Project_BugLoc/compute_canada/input/kra/"
    img_mat_file = "/media/iwona/Optane/Project_BugLoc/code/img_code.npz"
    gent = gen.DeepTrace_DataGenerator(train_indices,
                                       observations_key_triplets,
                                       config_file,
                                       mode,
                                       shuffle=True)

    genv = gen.DeepTrace_DataGenerator(validation_indices,
                                       observations_key_triplets,
                                       config_file,
                                       mode,
                                       shuffle=True)

    #print("xxxxxxxxxxxxx")
    #print(gent.batch_size, gent.cases, len(train_indices))
    #print(genv.batch_size, genv.cases, len(validation_indices))
    #exit(1)
    #epochs = config["epochs"]
    #model.compile(optimizer=config["optimizer"], loss=config["loss"], metrics=['accuracy', tp_rate, tn_rate, fp_rate, fn_rate])
    #model.compile(optimizer=config["optimizer"], loss=config["loss"], metrics=['accuracy', tp_rate, tn_rate])
    model.compile(optimizer=keras.optimizers.Adam(), loss=config["loss"], metrics=['accuracy'])

    model.save(os.path.join(exp_path + "model.h5"))
    print("Model Saved*:", exp_path)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    model.fit_generator(generator=gent,
                        validation_data=genv,
                        epochs=config["epochs"],
                        callbacks=[early_stopping, logger]) # , chkpoint])




