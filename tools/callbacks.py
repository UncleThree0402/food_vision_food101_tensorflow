import datetime
import tensorflow as tf


def create_tensorboard_callback(dirname, experiment_name):
    log_dir = dirname + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TB log file to {log_dir}")
    return tensorboard_callback


def create_model_checkpoint(path):
    target_path = path + "/" + "checkpoint.ckpt"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=target_path,
                                                             monitor="val_accuracy",
                                                             save_weights_only=True,
                                                             save_best_only=True,
                                                             save_freq="epoch",
                                                             verbose=1)
    return checkpoint_callback


def create_early_stopping(monitor, patience):
    return tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience)
