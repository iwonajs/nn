from __future__ import absolute_import, division, print_function, unicode_literals
import run_model as r
import tensorflow as tf
import gc
import argparse




def run(run_i, fold_i, flag_use_img, flag_one_obs, flag_save_weights, model_id):
    run_model = r.RunModel()
    if flag_one_obs:
        run_model.fold_x_test_one(run_i, fold_i, flag_use_img, flag_save_weights, model_id)
    else:
        run_model.fold_x(run_i, fold_i, flag_use_img, flag_save_weights, model_id)
    tf.keras.backend.clear_session()
    gc.collect()


def use_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert len(gpus) > 0
    if gpus:
        print(gpus)
        for gpu in gpus:
            print("_______________________________________Physical GPU:")
            print(gpu.device_type)
            tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', '-n', type=int, default=-1)
    parser.add_argument('--modulo', '-u', type=int, default=-1)
    parser.add_argument('--run', '-r', type=int, default=-1)
    parser.add_argument('--fold', '-f', type=int, default=-1)
    parser.add_argument('--gpu', '-g', help="select gpu id", type=int)
    parser.add_argument("--img", '-i', help="use the images in nn", type=bool, default=True)
    parser.add_argument("--one", "-o", help="just one observation", type=bool, default=False)
    parser.add_argument("--weights", "-w", help="save weights between epochs", type=bool, default=False)
    parser.add_argument("--model", "-m", help="", type=int, default=1)
    args = parser.parse_args()
    print("args:", args)

    if args.fold > -1 and args.run > -1:
        run_i = args.run
        fold_i = args.fold
    elif args.number > -1 and args.modulo > -1:
        number = args.number
        fold_count = args.modulo
        run_i = number // fold_count
        fold_i = number % fold_count

    use_gpu()
    with tf.device('/GPU:0'):
        run(run_i, fold_i, args.img, args.one, args.weights, args.model)
