import argparse
import json
import matplotlib
matplotlib.use("Agg")
import os
from config import config
from core.callbacks import TrainingMonitor
from core.io import HDF5Reader
from core.nn import ResNet
from core.preprocessing import MeanSubtraction
from core.preprocessing import Resize
from core.preprocessing import ToArray
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

EPOCHS = 75

def poly_decay(epoch):
    epochs = EPOCHS
    init_lr = 1e-1
    power = 1.0
    lr = init_lr * (1 - (epoch / float(epochs))) ** power
    return lr

def main():

    # Define preprocessors
    means = json.loads(open(config.MEAN_PATH).read())
    mean_subtraction = MeanSubtraction(means["R"], means["G"], means["B"])
    preprocessors = [Resize(64, 64), mean_subtraction, ToArray()]

    # Define image augmentation
    augmentation = ImageDataGenerator(
        rotation_range=18,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # Data generator from HDF5 files
    train_gen = HDF5Reader(config.TRAIN_PATH,
        batch_size=64,
        preprocessors=preprocessors,
        augmentation=augmentation,
        classes=config.NUM_CLASSES
    )

    valid_gen = HDF5Reader(config.VALID_PATH,
        batch_size=64,
        preprocessors=preprocessors,
        classes=config.NUM_CLASSES
    )


    print("[INFO] Compiling model ...")
    optimizer = SGD(lr=1e-1, momentum=0.9)
    model = ResNet.build(64, 64, 3, classes=config.NUM_CLASSES, stages=(3, 4, 6), filters=(64, 128, 256, 512), reg=0.005)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Training monitor
    training_monitor = TrainingMonitor(config.FIG_PATH, json_path=config.JSON_PATH)
    callbacks = [training_monitor, LearningRateScheduler(poly_decay)]

    model.fit(
        train_gen.generator(),
        epochs=EPOCHS,
        steps_per_epoch=train_gen.num_images // 64,
        validation_data=valid_gen.generator(),
        validation_steps=valid_gen.num_images // 64,
        max_queue_size=10,
        callbacks=callbacks,
        verbose=1
    )

    model.save(config.MODEL_PATH)

    train_gen.close()
    valid_gen.close()


if __name__ == '__main__':
    main()
