import json
import numpy as np
import os
import sys
import shutil
import time
import tensorflow as tf
import data_generator as gen
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from varname import nameof
import models as mm
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

class Run_Model:

    def __init__(self):
        self.config_path = "config.json"
        self.config = json.load(open(self.config_path, "r"))

    def get_model(self, model_id):
        if model_id == 1:
            return mm.Base_Model_Attention_Before(self.config)
        elif model_id == 2:
            return mm.Base_Model_KRA_SEQ_OL_L3(self.config)

    def out_paths(self, configs, run_i, fold_i):
        results_dir = self.config["results_dir"]
        exp_folder = "run_" + str(run_i)
        results_path = os.path.join(results_dir, nn.__class__.__name__)

        print("--- Input ---")
        print("Model", nn.model.__class__.__name__)
        print(nameof(run_i), str(run_i))
        print(nameof(fold_i), str(fold_i))

        # exp_folder = str(run_)
        if results_path == ".":
            exp_path = exp_folder
        else:
            exp_path = os.path.join(results_path, exp_folder)

        print("Experiment Path:", exp_path)
        if os.path.exists(exp_path):
            print("Experiment folder already exists.")
        #    sys.exit(1)
        else:
            os.makedirs(exp_path)

        fold_folder = "fold_" + str(fold_i)
        # fold_folder = fold_name + "/"
        fold_path = os.path.join(exp_path, fold_folder)
        fold_path_exists = os.path.exists(fold_path)
        if fold_path_exists:
            # clear previous output files
            for f in os.listdir(fold_path):
                os.remove(fold_path + f)
        else:
            os.mkdir(fold_path)

        out_csv = os.path.join(exp_path, fold_folder + ".csv")  # config["logger_path"]


    def fold_x(self, run_i, fold_i, useImgFlag, flag_save_weights, model_id):
    #def fold_x(results_path, exp_folder, folds_data_path, dtm, x, useImgFlag):
        #r.fold_x(results_model_path, run, folds_data_path, nn, fold_i, True)

        #config = json.load(open("config.json", "r"))
        results_dir = self.config["results_dir"]
        folds_data_path = self.config["folds_data_path"]

        #run = "run" + str(run_i)
        exp_folder = "run_" + str(run_i)

        nn = self.get_model(model_id)


        #results_model_path = os.path.join(results_dir, nn.__class__.__name__)
        results_path = os.path.join(results_dir, nn.__class__.__name__)
        # fold_csv_path = os.path.join(results_model_path, "fold_" + str(fold_i) + ".csv")

        print("--- Input ---")
        print("Model", nn.model.__class__.__name__)
        print(nameof(run_i), str(run_i))
        print(nameof(fold_i), str(fold_i))
        #exit(6)

        #exp_folder = str(run_)
        if results_path == ".":
            exp_path = exp_folder
        else:
            exp_path = os.path.join(results_path, exp_folder)

        print("Experiment Path:", exp_path)
        if os.path.exists(exp_path):
            print("Experiment folder already exists.")
        #    sys.exit(1)
        else:
            os.makedirs(exp_path)


        #config_file = nn.config_file
        #if not os.path.exists(config_file):
        #    print("Error, Experiment config file: " + self.config_file + " does not exist.")
        #    sys.exit(1)
        #with open(config_file) as f:
        #   self.config = json.loads(f.read())

        fold_folder = "fold_" + str(fold_i)
        #fold_folder = fold_name + "/"
        fold_path = os.path.join(exp_path, fold_folder)
        fold_path_exists = os.path.exists(fold_path)
        if fold_path_exists:
            # clear previous output files
            for f in os.listdir(fold_path):
                os.remove(fold_path + f)
        else:
            os.mkdir(fold_path)

        out_csv = os.path.join(exp_path, fold_folder + ".csv")  # config["logger_path"]
        logger = CSVLogger(out_csv)
        checkpoint_file = os.path.join(fold_path, self.config["callback_file"])
        chkpoint = ModelCheckpoint(checkpoint_file, verbose=1)

        #print("test: ", fold_dir_exists)
        #print("checkpoint: ", fold_dir)
        #print("logger: ", out_csv)
        #print("checkpoint: ", checkpoint_file)
        #exit(1)

        # load observation pairs
        #folds_data_path = results_path + "folds_pos_neg.json"
        folds_data_path = self.config["folds_data_path"]

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
            if v["fold"] == fold_i:
                validation_indices += [id]
            else:
                train_indices += [id]
            id += 1

        # Init the Generator
        #bugs_path = self.config["bugs_path"]
        #code_path = self.config["code_path"]
        #img_path = self.config["image_path"]
        #img_mat_file = "/media/iwona/Optane/Project_BugLoc/code/img_code.npz"
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


        batch0 = gent.__getitem__(0)
        # print(batch0)
        # print(len(batch0))
        #print(len(batch0[0]), len(batch0[1]))
        #print(batch0[0][0].shape)
        #print(batch0[0][1].shape)
        #print(batch0[0][2].shape)
        #print(batch0[0][3].shape)
        #print("------------------------------------")
        epochs = self.config["epochs"]
        #model.compile(optimizer=config["optimizer"], loss=config["loss"], metrics=['accuracy', tp_rate, tn_rate, fp_rate, fn_rate])
        #model.compile(optimizer=config["optimizer"], loss=config["loss"], metrics=['accuracy', tp_rate, tn_rate])

        #tf.keras.models.save_model(model, exp_path+"model.h5", overwrite=True, include_optimizer=True, save_format='h5')
        # model.save_weights("/", save_format='tf')
        nn.model.save(os.path.join(exp_path, "model.h5"))
        print("Model Saved*:", exp_path)

        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=exp_path)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor=self.config['early_stopping_metric'],
                                                          patience=self.config['patience'])
        callbacks_list = [early_stopping, logger]
        if flag_save_weights:
            callbacks_list += [chkpoint]
        nn.model.fit_generator(generator=gent,
                            validation_data=genv,
                            epochs=epochs,
                            callbacks=callbacks_list,
                            use_multiprocessing=True)
                            #callbacks=[tensorboard_callback])
                            #callbacks=[logger, chkpoint])

    def fold_x_test_one(self, run_i, fold_i, useImgFlag, flag_save_weights, model_id):
    #def fold_x_test_one(results_model_path, run_folder, folds_data_path, dtm, x, useImgFlag):

        #if results_model_path == ".":
        #    exp_path = os.path.join(exp_folder)
        #else:
        #exp_path = os.path.join(results_model_path, run_folder)

        #print("Experiment Path:", exp_path)
        #if os.path.exists(exp_path):
        #    print("Experiment folder already exists.")
        #    sys.exit(1)
        #else:
        #    os.makedirs(exp_path)
        config = json.load(open("config.json", "r"))
        results_dir = config["results_dir"]
        folds_data_path = config["folds_data_path"]

        # run = "run" + str(run_i)
        run_folder = "run_" + str(run_i)

        nn = self.get_model(model_id)
        print("tessssssssssssss ", nameof(nn.model))

        #model = nn.model
        #feed_flag = nn.feed_flag

        results_model_path = os.path.join(results_dir, nn.__class__.__name__)
        #results_path = os.path.join(results_dir, nn.__class__.__name__)
        # fold_csv_path = os.path.join(results_model_path, "fold_" + str(fold_i) + ".csv")

        print("--- Input ---")
        print("Model x", nn.__class__.__name__)
        print(nameof(run_i), str(run_i))
        print(nameof(fold_i), str(fold_i))


        #exit(4)

        #config_file = nn.config
        #if not os.path.exists(config_file):
        #    print("Error, Experiment config file: " + config_file + " does not exist.")
        #    sys.exit(1)
        #with open(config_file) as f:
        #    config = json.loads(f.read())

        fold_folder = "fold_" + str(fold_i)
        output_path = os.path.join(results_model_path, run_folder, fold_folder)
        if os.path.exists(output_path):
            # clear previous output files
            for f in os.listdir(output_path):
                os.remove(os.path.join(output_path, f))
        else:
            #os.mkdirs(output_path)
            os.makedirs(output_path)


        model_path = os.path.join(output_path, "model.h5")
        nn.model.save(model_path)

        out_csv = output_path + ".csv"  # config["logger_path"]
        logger = CSVLogger(out_csv)

        checkpoint_file = os.path.join(output_path, config["callback_file"])
        chkpoint = ModelCheckpoint(checkpoint_file, verbose=1)

        print("--- Output Files ---")
        print("CSV*: ", out_csv)
        print("Model Saved*:", model_path)



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
            if v["fold"] == fold_i:
                validation_indices += [id]
            else:
                train_indices += [id]
            id += 1

        # Init the Generator
        #bugs_path = config["bugs_path"]
        #code_path = config["code_path"]
        #img_path = config["image_path"]
        #img_mat_file = "/media/iwona/Optane/Project_BugLoc/code/img_code.npz"
        # #bugs_path, code_path, img_path,
        #useImgFlag=useImgFlag,


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

        epochs = config["epochs"]
        #model.compile(optimizer=config["optimizer"], loss=config["loss"], metrics=['accuracy', tp_rate, tn_rate, fp_rate, fn_rate])
        #model.compile(optimizer=config["optimizer"], loss=config["loss"], metrics=['accuracy', tp_rate, tn_rate])

        #tf.keras.models.save_model(model, exp_path+"model.h5", overwrite=True, include_optimizer=True, save_format='h5')
        # model.save_weights("/", save_format='tf')


        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=exp_path)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor=config["early_stopping_metric"],
                                                          patience=config["patience"])
        #model.fit_generator(generator=gent,
        #                    validation_data=genv,
        #                    epochs=30,
        #                    callbacks=[early_stopping, logger, chkpoint],
        #                    use_multiprocessing=True)
        #                    #callbacks=[tensorboard_callback])
        #                    #callbacks=[logger, chkpoint])
        #for batch in gent:
        batch0 = gent.__getitem__(0)
        #print(batch0)
        #print(len(batch0))
        print(len(batch0[0]), len(batch0[1]))
        print(batch0[0][0].shape)
        print(batch0[0][1].shape)
        print(batch0[0][2].shape)


        #print(batch0[0][3].shape)
        # FIX IMAGE SIZE
        #batch0[0][3] = np.ones([1, 150, 150, 3], dtype=np.uint8)
        callbacks_list = [logger]
        if flag_save_weights:
            callbacks_list += [chkpoint]


        nn.model.fit(x=batch0[0], y=batch0[1], epochs=epochs, callbacks=callbacks_list)
        #nn.model.fit(x=batch0[0], y=batch0[1], epochs=epochs)


