import json
import tarfile
import os
import shutil
import glob
import stat

def tar_input(configs):
    destination = os.path.join(configs["compute_canada_staging"], "input.tar.gz")
    with tarfile.open(destination, "w:gz") as tar:
        print("0", configs["train_pos_folds"])
        tar.add(configs["train_pos_folds"])
        print("1", configs["bugs_path"])
        tar.add(configs["bugs_path"])
        print("2", configs["kra_seqO_path"])
        tar.add(configs["kra_seqO_path"])
        print("3", configs["kra_seqL_path"])
        tar.add(configs["kra_seqL_path"])
        #tar.add(configs["img_path_decimal_standard_npz"])
        #print("4")
        #tar.add(configs["img_path_decimal_maxval_npz"])
        #print("5")
        tar.close()

if __name__ == "__main__":
    configs = json.load(open("config.json", "r"))
    # TAR THE OBSERVATION FILES
    #tar_input(configs)
    # COPY FILES (CONFIG / SCRIPTS)
    target_path = configs["compute_canada_staging"]
    #shutil.copy("config.json", target_path)
    shutil.copy("models.py", target_path)
    shutil.copy("model_data_generator_neg_sampling.py", target_path)
    shutil.copy("run_model.py", target_path)
    shutil.copy("run_terminal.py", target_path)
    for file in glob.glob(r'./sbatch_*.sh'):
        sbatch_file = os.path.basename(file)
        print("sbatch file:", sbatch_file)
        os.chmod(sbatch_file, stat.S_IRWXU)
        shutil.copy(sbatch_file, target_path)
    shutil.copy("clear_code.sh", target_path)
    shutil.copy("clear_results.sh", target_path)
    shutil.copy("push_cedar_rsync.sh", target_path)
    shutil.copy("push_graham_rsync.sh", target_path)
    #shutil.copy(configs["train_pos_folds"], target_path)

    cc_configs = dict()
    cc_configs = configs
    # cc_configs["epochs"] = 5
    # PATHS: NN input
    # NN input for GENERATOR
    #temp_dir = "$SLURM_TMPDIR"
    cc_configs["train_pos_folds"] = os.path.join(*configs["train_pos_folds"].split(os.sep))  #"/media/iwona/Optane/Data/compute_canada/input/observations/folds_5/folds_pos_neg.json"
    cc_configs["bugs_path"] = os.path.join(*configs["bugs_path"].split(os.sep))  #"/media/iwona/Optane/Data/compute_canada/input/bugs/vectorized_trainedGloVe_300_paired/"
    cc_configs["kra_seqO_path"] = os.path.join(*configs["kra_seqO_path"].split(os.sep))  #"/media/iwona/Optane/Data/compute_canada/input/code/vectorized_vocab10K_pairs/"
    cc_configs["kra_seqL_path"] = os.path.join(*configs["kra_seqL_path"].split(os.sep))  #"/media/iwona/Optane/Data/compute_canada/input/code/vectorized_vocab10K_pairs/"
    #cc_configs["img_path"] = os.path.join(*configs["img_path"].split(os.sep))  #config["img_path_decimal_standard_npz"]
    #print(cc_configs["img_path"])
    # PATHS: NN output
    cc_configs["results_dir"] = os.path.join("results")  #"/home/iwona/Data/results/img_paper/"
    cc_configs["results_dir_flat"] = os.path.join("results")  # "/home/iwona/Data/results/img_paper/"
    json.dump(cc_configs, open("config_cc.json", "w"))
    #print("config....................", target_path)
    shutil.copy("config_cc.json", os.path.join(target_path, "config.json"))
    #with open(os.path.join(target_path, "config.json"),'r') as f:
    #    data = json.load(f)
    #print(data)

