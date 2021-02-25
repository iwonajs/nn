import json
import os
import tensorflow as tf
import data_generator as gen
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
import models as mm


class RunModel:

    def __init__(self):
        self.config_path = "config.json"
        self.config = json.load(open(self.config_path, "r"))

    def get_model(self, model_id):
        if model_id == 1:
            return mm.Base_Model_Attention_Before(self.config)
        elif model_id == 2:
            return mm.Base_Model_KRA_SEQ_OL_L3(self.config)

    def make_out_paths(self, run_i, fold_i, nn):
        # MODEL -> RUN -> FOLD

        results_model_path = os.path.join(self.config["results_dir"], nn.__class__.__name__)
        results_run_path = os.path.join(results_model_path, "run_" + str(run_i))
        results_fold_path = os.path.join(results_run_path, "fold_" + str(fold_i))

        if not os.path.exists(results_run_path):
            os.makedirs(results_run_path)

        if os.path.exists(results_fold_path):
            for f in os.listdir(results_fold_path):
                os.remove(os.path.join(results_fold_path, f))
        else:
            os.mkdir(results_fold_path)

        return results_model_path, results_run_path, results_fold_path, "fold_" + str(fold_i)

    def fold_x(self, run_i, fold_i, useImgFlag, flag_save_weights, model_id):

        # MODEL and OUTPUT PATHS: MODEL -> RUN -> FOLD
        nn = self.get_model(model_id)
        results_model_path, results_run_path, results_fold_path, fold_folder = self.make_out_paths(run_i, fold_i, nn)
        nn.model.save(os.path.join(results_run_path, fold_folder + ".h5"))
        print("Model Saved*:", results_run_path)

        # FOLD DATA
        with open(self.config["folds_data_path"]) as json_file:
            fold_data = json.load(json_file)

        train_indices = []
        validation_indices = []
        observations_key_triplets = []
        id = 0
        for k, v in fold_data.items():
            observations_key_triplets += [[v["cfm_npz"], v["bug"], v["pos"]]]
            if v["fold"] == fold_i:
                validation_indices += [id]
            else:
                train_indices += [id]
            id += 1

        gent = gen.DeepTrace_DataGenerator(train_indices,
                                           observations_key_triplets,
                                           nn.feed_flag,
                                           self.config,
                                           shuffle=True)

        genv = gen.DeepTrace_DataGenerator(validation_indices,
                                           observations_key_triplets,
                                           nn.feed_flag,
                                           self.config,
                                           shuffle=True)


        # FIT PARAMETERS
        epochs = self.config["epochs"]
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor=self.config['early_stopping_metric'],
                                                          patience=self.config['patience'])
        out_csv = os.path.join(results_run_path, fold_folder + ".csv")
        logger = CSVLogger(out_csv)
        checkpoint_file = os.path.join(results_fold_path, self.config["callback_file"])
        chkpoint = ModelCheckpoint(checkpoint_file, verbose=1)
        callbacks_list = [early_stopping, logger]
        if flag_save_weights:
            callbacks_list += [chkpoint]

        nn.model.fit_generator(generator=gent,
                               validation_data=genv,
                               epochs=epochs,
                               callbacks=callbacks_list,
                               use_multiprocessing=True)

    def fold_x_test_one(self, run_i, fold_i, useImgFlag, flag_save_weights, model_id):

        # MODEL and OUTPUT PATHS: MODEL -> RUN -> FOLD
        nn = self.get_model(model_id)
        results_model_path, results_run_path, results_fold_path, fold_folder = self.make_out_paths(run_i, fold_i, nn)
        nn.model.save(os.path.join(results_run_path, fold_folder + ".h5"))
        print("Model Saved*:", results_run_path)

        # OUTPUT PATHS: MODEL -> RUN -> FOLD
        results_model_path, results_run_path, results_fold_path, fold_folder = self.make_out_paths(run_i, fold_i, nn)

        # FOLD DATA
        with open(self.config["folds_data_path"]) as json_file:
            fold_data = json.load(json_file)

        train_indices = []
        validation_indices = []
        observations_key_triplets = []
        id = 0
        for k, v in fold_data.items():
            observations_key_triplets += [[v["cfm_npz"], v["bug"], v["pos"]]]
            if v["fold"] == fold_i:
                validation_indices += [id]
            else:
                train_indices += [id]
            id += 1

        gent = gen.DeepTrace_DataGenerator(train_indices,
                                           observations_key_triplets,
                                           nn.feed_flag,
                                           self.config,
                                           shuffle=True)

        # FIT PARAMETERS
        epochs = self.config["epochs"]
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor=self.config['early_stopping_metric'],
                                                          patience=self.config['patience'])
        out_csv = os.path.join(results_run_path, fold_folder + ".csv")
        logger = CSVLogger(out_csv)
        checkpoint_file = os.path.join(results_fold_path, self.config["callback_file"])
        chkpoint = ModelCheckpoint(checkpoint_file, verbose=1)
        callbacks_list = [early_stopping, logger]
        if flag_save_weights:
            callbacks_list += [chkpoint]

        batch0 = gent.__getitem__(0)
        #print(len(batch0[0]), len(batch0[1]))
        #print(batch0[0][0].shape)
        #print(batch0[0][1].shape)
        #print(batch0[0][2].shape)

        nn.model.fit(x=batch0[0], y=batch0[1], epochs=epochs, callbacks=callbacks_list)


