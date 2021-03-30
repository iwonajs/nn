import json
import os
config = {}



config["epochs"] = 100
config["batch_size"] = 1
config["bug_embedding"] = 300
config["code_dim1_max_word_features"] = 23
config["code_dim2_max_word_features"] = 10001
config["code_embedding_d1_cat"] = 4
config["code_embedding_d2_token"] = 300
config["code_gru_units"] = 32
config["bug_gru_units"] = 32
config["dropout"] = 0.5
config["gru_units"] = 32
config["kra_objects_features"] = 1502
config["kra_levels_features"] = 1463
config["kra_objects_embedding"] = 32
config["kra_levels_embedding"] = 32
config["optimizer"] = "rmsprop"
config["loss"] = "binary_crossentropy"
config["callback_file"] = "weights-improvement-{epoch:02d}.hdf5"

config["patience"] = 5
config["early_stopping_metric"] = "loss" #"val_accuracy"

# NOTE FOR NOW WE CAN ONLY DO batch_size = 1!
config['batch_size'] = 1
config['shuffle'] = True
config['resample_neg_on_epoch'] = True
# DATA IN
#                         /home/iwona/optane/msr_input
folds_data_path_tokens = ["/"] + "/home/iwona/optane/msr_input/folds_10_pos_neg.json".split("/")
config["folds_data_path"] = "" #os.path.join(*folds_data_path_tokens)

training_pos_10_folds_tokens = ["/"] + "/home/iwona/optane/msr_input/folds_10_pos/training_pos_folds_10.json".split("/")
config["train_pos_folds"] = os.path.join(*training_pos_10_folds_tokens)
test_pos_tokens = ["/"] + "/home/iwona/optane/msr_input/folds_10_pos/test_pos.json".split("/")
config["test_pos"] = os.path.join(*test_pos_tokens)

bugs_path_tokens = ["/"] + "/home/iwona/optane/msr_input/vectorized_trainedGloVe_300_paired".split("/")
config["bugs_path"] = os.path.join(*bugs_path_tokens)

kra_path_tokens = ["/"] + "/home/iwona/optane/msr_input/kra".split("/")
config["kra_path"] = os.path.join(*kra_path_tokens)

kra_seqO_path_tokens = ["/"] + "/home/iwona/optane/msr_input/kra_seqO".split("/")
config["kra_seqO_path"] = os.path.join(*kra_seqO_path_tokens)

kra_seqL_path_tokens = ["/"] + "/home/iwona/optane/msr_input/kra_seqL".split("/")
config["kra_seqL_path"] = os.path.join(*kra_seqL_path_tokens)

# DATA OUT
results_dir_tokens = ["/"] + "/home/iwona/results_out/scratch/run_local".split("/")
config["results_dir"] = os.path.join(*results_dir_tokens)

results_dir_tokens = ["/"] + "/home/iwona/results_out/scratch/run_test".split("/")
config["results_dir_flat"] = os.path.join(*results_dir_tokens)

# "" #"C:\\Users\\evona\\Downloads\\Data_Large\\msr_input\\vectorized_vocab10K_pairs\\"
config["code_path"] = ""
#"/media/iwona/Optane/Project_BugLoc/compute_canada/input/images/"
# img_path
config["image_path"] = ""

config["compute_canada_staging"] = "/home/iwona/optane/compute_canada_staging/nn_CAR_binary_10F/"

config["batch_size"] = 1

config["variant_model"] = ""
config["variant_batch"] = ""
config["variant_preprocess"] = ""

batch_variants = dict()
batch_variants["batch_v00"] = dict()
batch_variants["batch_v00"]["batch_size"] = 1
config["variant_batch_list"] = batch_variants



with open("config.json", 'w+') as f:
    json.dump(config, f)