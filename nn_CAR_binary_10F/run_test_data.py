import tensorflow as tf
import os
import json
import model_data_generator_neg_sampling as gen
import pickle
import timeit


def run_model(results_path):
        config_path = os.path.join(results_path, "config.json")


        configs = json.load(open(config_path, "r"))

        #print(list(model.signatures.keys()))
        #infer = model.signatures["serving_default"]
        #print(infer.structured_outputs)
        configs["bugs_path"] = os.path.join("/", configs["bugs_path"])
        configs["kra_seqO_path"] = os.path.join("/", configs["kra_seqO_path"])
        configs["kra_seqL_path"] = os.path.join("/", configs["kra_seqL_path"])

        gen_test = gen.DeepTrace_DataGenerator(configs, configs["test_pos"], "CAR",
                                           validation_fold=-1,
                                           train_mode=False)
        #for batch in gen_test:
        #    pass
        #for i, k in configs.items():
        #    print(i,k)
        #gen_test = DeepTrace_DataGenerator(configs, configs["test_pos"], "CAR",
        #                                   validation_fold=-1,
        #                                   train_mode=False)

        #print("zzzzzz")
        #for batch in gen_test:
        #    print("Batch *************************************************", gen_test.cases)
        #    # print(batch)
        #    print("batch len", len(batch))
        #    dim = len(batch)
        #    for di in range(dim):
        #        print(">>>> inputs ", len(batch[di]))
        #        for dij in range(len(batch[di])):
        #            print("item:", dij, batch[di][dij].shape)
        #            if di == 1 and batch[di][0] == 1:
        #                print("positive")
        #            if di == 1 and batch[di][0] == 0:
        #                print("negative")

        #batch0 = gent.__getitem__(0)
        #valid0 = genv.__getitem__(0)
        #self.nn.model.fit(x=batch0[0], y=batch0[1],
        #                  epochs=epochs,
        #                  callbacks=callbacks_list,
        #                  validation_data=(valid0[0], valid0[1]))

        model_path = os.path.join(results_path, "model")
        batch = gen_test.__getitem__(0)
        loaded = tf.saved_model.load(model_path)
        inferx = loaded.signatures["serving_default"]
        print(inferx.structured_outputs)
        #loaded = tf.keras.models.load_model(model_path)
        #loaded([batch[0], batch[0][1], batch[0][2]])
        print(len(batch[0]))
        print(batch[0][0].shape, batch[0][1].shape, batch[0][2].shape)
        inferx(tf.constant(1))
        loaded.__call__()
        print("zzzzzzzzzzzzzzzzzzzzzzzzzzz")



def use_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert len(gpus) > 0
    if gpus:
        print(gpus)
        for gpu in gpus:
            print("_______________________________________Physical GPU:")
            print(gpu.device_type)
            tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == '__main__':
    use_gpu()
    with tf.device('/GPU:0'):

        mpath = "/home/iwona/results_out/car_models/CAR_variant_32_epochs100_r2/metrics_val_loss_min/fold_0/results/model/val_loss/"
        config_path = "/home/iwona/results_out/car_models/CAR_variant_32_epochs100_r2/metrics_val_loss_min/fold_0/results/config.json"
        configs = json.load(open(config_path, "r"))
        configs = json.load(open(config_path, "r"))
        configs["bugs_path"] = os.path.join("/", configs["bugs_path"])
        configs["kra_seqO_path"] = os.path.join("/", configs["kra_seqO_path"])
        configs["kra_seqL_path"] = os.path.join("/", configs["kra_seqL_path"])
        include_cat_test = [0]
        gen_test = gen.DeepTrace_DataGenerator(configs, configs["test_pos"], "CAR",
                                               validation_fold=-1,
                                               train_mode=False,
                                               include_cat=include_cat_test)
        # ,
        #                                                include_cat=[1]
        predictions = []
        counter = 0
        for batch in gen_test:
            model = tf.keras.models.load_model(mpath)
            #batch = gen_test.__getitem__(0)
            #print("X values ----- ", len(batch[0]))
            #print(batch[0])
            #print("Y values ----- ", len(batch[1]))
            #print(batch[1])
            #print("Cases: ", gen_test.cases)
            #res = model.evaluate(batch[0], batch[1], batch_size=1)
            #print("res")
            #print(res)
            #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", len(batch[0]))
            #print(batch[0])
            #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", len(batch[1]))
            #print(batch[1])
            p = model.predict(batch[0])
            predictions += [p[0][0]]
            counter += 1
            print("progress: ", counter/float(gen_test.cases), counter, gen_test.cases, p[0][0])
            if 1 in include_cat_test:
                binary = [1 if i >= .5 else 0 for i in predictions]
            else:
                binary = [1 if i > .5 else 0 for i in predictions]
            obs = len(binary)
            TP = sum(binary)
            FN = obs - TP
            print("")
            #print(p[0], batch[1][0])

        out_path = os.path.join(mpath, "predictions_negative_-_metrics_val_loss_min_-_val_loss.p")
        pickle.dump(predictions, open(out_path, "wb"))
        binary = [ 1 if i > .5 else 0 for i in predictions]
        obs = len(binary)
        TP = sum(binary)
        FN = obs - TP
        print("OBS:", obs)
        print("TP:", TP, TP/float(obs))
        print("FN:", FN, FN/float(obs))