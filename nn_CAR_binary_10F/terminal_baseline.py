from __future__ import absolute_import, division, print_function, unicode_literals
import models as dtm
import run_gen4 as r
import sys
import os
import tensorflow as tf
import gc


run_i = sys.argv[1]
fold_i = int(sys.argv[2])

config_file = "config.json"
folds_data_path = "folds_10_pos_neg.json"
results_dir_model = "msr_results_baseline_x"

run = "run" + str(run_i)
fold_csv = os.path.join(results_dir_model, run, "/fold_" + str(fold_i) + ".csv")

if not os.path.exists(fold_csv):
      nn = dtm.Base_Model_Attention_Before(config_file)

      # results_path, exp_folder, folds_data_path, dtm, x, useImgFlag
      #r.fold_x(results_dir_model, run, folds_data_path, nn, fold_i)
      #tf.keras.backend.clear_session()
      patience = 10
      r.fold_x(nn.gen_mode(), patience, results_dir_model, run, folds_data_path, nn, fold_i)
      tf.keras.backend.clear_session()








