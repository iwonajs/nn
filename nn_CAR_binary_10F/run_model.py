import json
import os
import tensorflow as tf
import model_data_generator_neg_sampling as gen
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
import models as mm
import datetime
import shutil
import numpy as np
import importlib

class RunModel:

    def __init__(self, run_i, fold_i, flag_chkpt, model_id, results_flat_dir, tag):
        self.run_i = run_i
        self.fold_i = fold_i
        self.flag_chkpt = flag_chkpt
        self.model_id = model_id
        self.results_flat_dir = results_flat_dir
        self.tag = tag

        self.config_path = "config.json"
        self.config = json.load(open(self.config_path, "r"))

        self.nn = self.get_model(self.model_id)
        self.nn.model.summary()
        self.fold_folder = "fold_" + str(self.fold_i)
        if results_flat_dir:
            self.config["results_dir"] = self.config["results_dir_flat"]
            self.results_run_path = self.config["results_dir"]
            #self.results_fold_path = os.path.join()
            # create dirs
            self.create_path(self.results_run_path)
            #self.create_path(self.results_fold_path)
            # out files
            self.out_csv = os.path.join(self.results_run_path, "metrics.csv")
            self.out_h5 = os.path.join(self.results_run_path, "model.h5")
            self.out_json = os.path.join(self.results_run_path, "config.json")
            self.out_chkpt = os.path.join(self.results_run_path, "chkpt", self.config["callback_file"])
            self.out_pb = os.path.join(self.results_run_path, "model")
        else:
            # MODEL -> RUN -> FOLD
            results_model_path = os.path.join(self.config["results_dir"], self.nn.__class__.__name__)
            self.results_run_path = os.path.join(results_model_path, "run_" + str(run_i))
            results_fold_path = os.path.join(self.results_run_path, "fold_" + str(fold_i))

            # create dirs
            self.check_and_create_path(results_model_path, False)
            self.check_and_create_path(self.results_run_path, False)
            self.check_and_create_path(results_fold_path, True)

            # out files
            self.out_csv = os.path.join(self.results_run_path, self.fold_folder + ".csv")
            self.out_h5 = os.path.join(self.results_run_path, self.fold_folder + ".h5")
            self.out_json = os.path.join(self.results_run_path, self.fold_folder + ".json")
            self.out_chkpt = os.path.join(results_fold_path, self.config["callback_file"])
            self.out_pb = os.path.join(results_fold_path, "model")
            self.remove_file(self.out_csv)
            self.remove_file(self.out_h5)
            self.remove_file(self.out_json)

        # configs
        self.model_variant = self.nn.__class__.__name__
        self.run_date = datetime.date.today()
        date_dict = dict()
        date_dict["year"] = self.run_date.year
        date_dict["month"] = self.run_date.month
        date_dict["day"] = self.run_date.day
        self.config["run_date"] = date_dict
        self.config["run_model_variant"] = self.model_variant
        self.config["run_tag"] = self.tag
        self.config["run_results_flat_directory"] = self.results_flat_dir
        self.config["run_fold"] = self.fold_i
        self.config["run"] = self.run_i
        this_script_path = os.path.realpath(__file__)
        self.config["project"] = this_script_path.split(os.sep)[-2]
        with open(self.out_json, 'w+') as f:
            json.dump(self.config, f)
        self.printout(["MODEL VARIANT:", self.model_variant])


    def get_model(self, model_id):
        class_name = "CAR_variant_"+str(model_id).zfill(2)
        model_class_instance = getattr(mm, class_name)
        return model_class_instance(self.config)

    @staticmethod
    def check_and_create_path(folder_path, purge=False):
        if os.path.exists(folder_path):
            if purge:
                for subdir, dirs, files, in os.walk(folder_path):
                    for f in files:
                        os.remove(os.path.join(subdir, f))
                    for dir_ in dirs:
                        shutil.rmtree(os.path.join(subdir, dir_))
        else:
            os.makedirs(folder_path)

    @staticmethod
    def remove_file(file_path):
        if os.path.exists(file_path) and os.path.isfile(file_path):
            os.remove(file_path)

    @staticmethod
    def create_path(folder_path):
        if os.path.exists(folder_path):
            print("ERROR: OUTPUT PATH ALREADY EXISTS", folder_path)
            exit(99)
        else:
            os.makedirs(folder_path)

    @staticmethod
    def printout(note):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(os.path.realpath(__file__))
        print(*note)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    # run_i, fold_i, useImgFlag, flag_save_weights, model_id, results_flat_dir
    def fold_x(self):

        # SAVE MODEL
        # tf.saved_model.save(self.nn.model, self.out_pb)
        # https://www.tensorflow.org/guide/keras/save_and_serialize/
        # https://www.tensorflow.org/guide/keras/custom_callback
        ###################self.nn.model.save(self.out_pb)
        self.printout(["Model Saved:", str(self.out_pb)])
        self.nn.model.save(self.out_h5)
        self.printout(["Model Saved:", str(self.out_h5)])

        # FOLD DATA
        gent = gen.DeepTrace_DataGenerator(self.config, self.config["train_pos_folds"], "CAR",
                                           validation_fold=self.fold_i,
                                           train_mode=True)

        genv = gen.DeepTrace_DataGenerator(self.config, self.config["train_pos_folds"], "CAR",
                                           validation_fold=self.fold_i,
                                           train_mode=False)
        # FIT PARAMETERS
        epochs = self.config["epochs"]
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor=self.config['early_stopping_metric'],
                                                          patience=self.config['patience'])
        save_model_best_val_accuracy = Save_Model_Best_Metric("val_accuracy", self.out_pb)
        save_model_best_val_loss = Save_Model_Best_Metric("val_loss", self.out_pb)
        save_model_best_accuracy = Save_Model_Best_Metric("accuracy", self.out_pb)
        save_model_best_loss = Save_Model_Best_Metric("loss", self.out_pb)
        logger = CSVLogger(self.out_csv)
        callbacks_list = [early_stopping,
                          logger,
                          save_model_best_val_loss,
                          save_model_best_val_accuracy,
                          save_model_best_accuracy,
                          save_model_best_loss]
        #if self.flag_chkpt:
        #    chkpoint = ModelCheckpoint(self.out_chkpt, verbose=1, save_best_only=True)
        #    callbacks_list += [chkpoint]

        # TRAIN
        self.nn.model.fit_generator(generator=gent,
                                    validation_data=genv,
                                    epochs=epochs,
                                    callbacks=callbacks_list,
                                    use_multiprocessing=True)

    def fold_x_test_one(self):

        # SAVE MODEL
        #tf.saved_model.save(self.nn.model, self.out_pb)
        #self.printout(["Model Saved:", str(self.out_pb)])
        self.nn.model.save(self.out_h5)
        self.printout(["Model Saved:", str(self.out_h5)])

        # FOLD DATA
        gent = gen.DeepTrace_DataGenerator(self.config, self.config["train_pos_folds"], "CAR",
                                           validation_fold=0,
                                           train_mode=True)
        genv = gen.DeepTrace_DataGenerator(self.config, self.config["train_pos_folds"], "CAR",
                                           validation_fold=0,
                                           train_mode=True)

        # FIT PARAMETERS
        epochs = self.config["epochs"]
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor=self.config['early_stopping_metric'],
                                                          patience=self.config['patience'])
        logger = CSVLogger(self.out_csv)
        save_model_best_val_accuracy = Save_Model_Best_Metric("val_accuracy", self.out_pb)
        save_model_best_val_loss = Save_Model_Best_Metric("val_loss", self.out_pb)
        save_model_best_accuracy = Save_Model_Best_Metric("accuracy", self.out_pb)
        save_model_best_loss = Save_Model_Best_Metric("loss", self.out_pb)
        callbacks_list = [early_stopping,
                          logger,
                          save_model_best_val_loss,
                          save_model_best_val_accuracy,
                          save_model_best_accuracy,
                          save_model_best_loss]
        #if self.flag_chkpt:
        #    chkpoint = ModelCheckpoint(self.out_chkpt, verbose=1, save_best_only=True)
        #    callbacks_list += [chkpoint]

        # TRAIN
        batch0 = gent.__getitem__(0)
        valid0 = genv.__getitem__(0)
        self.nn.model.fit(x=batch0[0], y=batch0[1],
                          epochs=epochs,
                          callbacks=callbacks_list,
                          validation_data=(valid0[0], valid0[1]))

# https://www.tensorflow.org/guide/keras/custom_callback
class Save_Model_Best_Metric(tf.keras.callbacks.Callback):
    """Save the model which has the lowest Validation Accuarcy
    Arguments: metric name: val_loss, loss, val_accuracy, accuracy
        """

    def __init__(self, metric_name, results_path):
        super(Save_Model_Best_Metric, self).__init__()
        self.best_weights = None
        self.metric_name = metric_name
        self.results_path = results_path
        self.path = os.path.join(results_path, self.metric_name)
        if "loss" in str(self.metric_name):
            self.optimize_min = True
        else:
            self.optimize_min = False

    def on_train_begin(self, logs=None):
        if self.optimize_min:
            self.best_metric = np.Inf
        else:
            self.best_metric = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current_metric_value = logs.get(self.metric_name)
        if self.optimize_min and np.less(current_metric_value, self.best_metric):
            self.model.save(self.path)
            print("BEST: ", str(self.metric_name),
                  " at ", current_metric_value,
                  " from ", self.best_metric,
                  " saved to:", self.path)
            self.epoch = epoch
            self.best_metric = current_metric_value
            self.metrics_accuracy = [logs.get("accuracy"), logs.get("val_accuracy")]
            self.metrics_loss = [logs.get("loss"), logs.get("val_loss")]
        if not self.optimize_min and np.less(self.best_metric, current_metric_value):
            self.model.save(self.path)
            print("BEST: ", str(self.metric_name),
                  " at ", current_metric_value,
                  " from ", self.best_metric,
                  " saved to:", self.path)
            self.epoch = epoch
            self.best_metric = current_metric_value
            self.metrics_accuracy = [logs.get("accuracy"), logs.get("val_accuracy")]
            self.metrics_loss = [logs.get("loss"), logs.get("val_loss")]

    def on_train_end(self, logs=None):
        metadata = dict()
        metadata["metric_name"] = self.metric_name
        metadata["metric_value"] = self.best_metric
        metadata["epoch"] = self.epoch
        metadata["metrics_accuracy"] = self.metrics_accuracy
        metadata["metrics_loss"] = self.metrics_loss
        json.dump(metadata, open(os.path.join(self.results_path, self.metric_name + ".json"), "w"))





