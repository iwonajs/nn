from __future__ import absolute_import, division, print_function, unicode_literals
import models as dtm
import run_gen4 as r
import sys
import os
import tensorflow as tf
import gc

run_i = sys.argv[1]
fold_i = int(sys.argv[2])

#results_dir = "/media/iwona/Optane/Project_BugLoc/compute_canada/paper/"
#config_file = "/media/iwona/Optane/Project_BugLoc/compute_canada/paper/meta_hperparams.json"
#folds_data_path = "/media/iwona/Optane/Project_BugLoc/compute_canada/paper/folds_pos_neg.json"
#results_dir_model = "/home/iwona/bugloc/results/paper/kra/Base_Model_KRA_SEQ_OL_L3/"

config_file = "config.json"
folds_data_path = "folds_10_pos_neg.json"
results_dir_model = "msr_results_33"

run = "run_" + str(run_i)
fold_csv = os.path.join(results_dir_model, run, "/fold_" + str(fold_i) + ".csv")

if not os.path.exists(fold_csv):
    nn = dtm.Base_Model_KRA_SEQ_OL_L3(config_file)
    patience = 10
    r.fold_x(nn.gen_mode(), patience, results_dir_model, run, folds_data_path, nn, fold_i)
    tf.keras.backend.clear_session()
    #gc.collect()






