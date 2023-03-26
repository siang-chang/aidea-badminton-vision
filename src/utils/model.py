import numpy as np
import tensorflow as tf
print("Tensorflow version " + tf.__version__)

def get_reducelr(monitor="val_loss"):
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=np.sqrt(0.1),
        patience=10,
        verbose=1,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=0.5e-7,
    )


def get_checkpoint(model_name, monitor="val_loss"):
    return tf.keras.callbacks.ModelCheckpoint(
        model_name,
        monitor=monitor,
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='max',
        save_freq='epoch'
    )

def get_early_stop(patience, monitor="val_loss"):
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        mode="min",
        min_delta=0,
        patience=patience,
        restore_best_weights=True
    )