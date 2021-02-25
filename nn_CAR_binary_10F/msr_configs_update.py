import json
import os
config = {}



config["epochs"] = 30
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
#config["bugs_path"] = os.path.join("C:/", "Users", "evona","Downloads", "Data_Large", "msr_input","vectorized_trainedGloVe_300_paired")
config["bugs_path"] = os.path.join("/mnt", "c", "Users", "evona","Downloads", "Data_Large", "msr_input","vectorized_trainedGloVe_300_paired")
# "" #"C:\\Users\\evona\\Downloads\\Data_Large\\msr_input\\vectorized_vocab10K_pairs\\"
config["code_path"] = os.path.join("/mnt", "c", "Users", "evona", "Downloads", "Data_Large", "msr_input", "vectorized_vocab10K_pairs")
config["image_path"] = "" #"/media/iwona/Optane/Project_BugLoc/compute_canada/input/images/"
#"C:\\Users\\evona\\Downloads\\Data_Large\\msr_input\\kra\\"
config["kra_path"] = os.path.join("/mnt", "c", "Users", "evona", "Downloads", "Data_Large", "msr_input", "kra")
# "C:\\Users\\evona\\Downloads\\Data_Large\\msr_input\\kra_seqO\\"
config["kra_seqO_path"] = os.path.join("/mnt", "c", "Users", "evona", "Downloads", "Data_Large", "msr_input", "kra_seqO")
"C:\\Users\\evona\\Downloads\\Data_Large\\msr_input\\input/kra_seqL\\"
config["kra_seqL_path"] = os.path.join("/mnt", "c", "Users", "evona", "Downloads", "Data_Large", "msr_input", "kra_seqL")

#config["epochs"] = 10
#config["patience"] = 5
#config["early_stopping_metric"] = "val_accuracy"
#config["batch_size"] = 1
#config["bug_embedding"] = 300
#config["code_dim1_max_word_features"] = 23
#config["code_dim2_max_word_features"] = 10001
#config["code_embedding_d1_cat"] = 4
#config["code_embedding_d2_token"] = 300
#config["code_gru_units"] = 32
#config["bug_gru_units"] = 32
#config["dropout"] = 0.5
#config["gru_units"] = 32
#config["kra_objects_features"] = 1501+1
#config["kra_levels_features"] = 1462+1
#config["kra_objects_embedding"] = 300
#config["kra_levels_embedding"] = 300
#config["optimizer"] = "rmsprop"
#config["loss"] = "binary_crossentropy"
#config["callback_file"] = "weights-improvement-{epoch:02d}.hdf5"

# IMAGE PREPROCESSING INPUT
#config["code_tokens_with_type"] = "/home/iwona/Data/spring_boot_extract_old/spring-boot/colored_objects"
#config["code_tokens_with_type_original_ext"] = ".txt.raw"
# IMAGE PREPROCESSING OUTPUT PATHS
#config["img_path_decimal_standard_npz"] = "/media/iwona/Optane/Data/compute_canada/input/code/code_decimal_standard_npz"
#config["img_path_decimal_standard_png"] = "/home/iwona/Data/spring_boot_extract_old/spring-boot/code_decimal_standard_png"
#config["img_path_decimal_metadata_p"] = "/home/iwona/Data/spring_boot_extract_old/spring-boot/code_decimal_metadata_p"
#config["img_path_decimal_maxval_npz"] = "/media/iwona/Optane/Data/compute_canada/input/code/code_decimal_maxval_npz"
#config["img_path_decimal_maxval_png"] = "/home/iwona/Data/spring_boot_extract_old/spring-boot/code_decimal_maxval_png"

# NN input for GENERATOR (KRA)
#config["kra_path"] = "/media/iwona/Optane/Data/compute_canada/input/kra/"
#config["kra_seqO_path"] = "/media/iwona/Optane/Data/compute_canada/input/kra_seqO/"
#config["kra_seqL_path"] = "/media/iwona/Optane/Data/compute_canada/input/kra_seqL/"

# NN input for GENERATOR
#config["folds_data_path"] = "/media/iwona/Optane/Data/compute_canada/input/observations/folds_5/folds_pos_neg.json"
#config["bugs_path"] = "/media/iwona/Optane/Data/compute_canada/input/bugs/vectorized_trainedGloVe_300_paired/"
#config["code_path"] = "/media/iwona/Optane/Data/compute_canada/input/code/vectorized_vocab10K_pairs/"
#config["img_path"] = "/media/iwona/Optane/Data/compute_canada/input/images/"
#config["img_path"] = config["img_path_decimal_standard_npz"]
# NN OUTPUT #
#config["results_dir"] = "/home/iwona/Data/results/img_paper/"

#config["compute_canada_staging"] = "/home/iwona/compute_canada_staging/rq_bugloc_img"

#config["config_file"] = "/home/iwona/bugloc/deep_trace/reserach_questions/ECIR/meta_hperparams.json"
#json_path = "/home/iwona/bugloc/deep_trace/reserach_questions/ECIR/meta_hperparams.json"

#if not os.path.exists(config["img_path_decimal_standard_npz"]):
#    print("missing path:", config["img_path_decimal_mat"])
#    json.dump(dict(), open("config.json", "w+"))
#    exit(1)
#if not os.path.exists(config["img_path_decimal_standard_png"]):
#    print("missing path:", config["img_path_decimal_png"])
#    json.dump(dict(), open("config.json", "w+"))
#    exit(1)

with open("msr_config.json", 'w+') as f:
    json.dump(config, f)