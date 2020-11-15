import matplotlib
matplotlib.use("Agg")
import numpy as np
import os
from core.callbacks import TrainingMonitor
from core.nn import ResNet
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def poly_decay(epoch):
    epochs = 100
    init_lr = 1e-1
    power = 1.0
    lr = init_lr * (1 - (epoch / float(epochs))) ** power
    return lr

def main():

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Apply mean subtraction
    X_train = X_train.astype("float")
    X_test = X_test.astype("float")
    mean = np.mean(X_train, axis=0)
    X_train -= mean
    X_test -= mean

    # One hot encoding
    label_encoder = LabelBinarizer()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    os.makedirs("output", exist_ok=True)

    # Training monitor
    fig_path = os.path.sep.join(["output", f"{os.getpid()}.png"])
    json_path = os.path.sep.join(["output", f"{os.getpid()}.json"])
    training_monitor = TrainingMonitor(fig_path, json_path=json_path)

    lr_scheduler = LearningRateScheduler(poly_decay)

    print("[INFO] Compiling model ...")
    optimizer = SGD(lr=1e-1)
    model = ResNet.build(32, 32, 3, 10, stages=(9, 9, 9), filters=(64, 64, 128, 256), reg=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Image augmentation
    augmentation = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    model.fit(
        augmentation.flow(X_train, y_train, batch_size=128),
        epochs=100,
        steps_per_epoch=len(X_train) // 128,
        validation_data=(X_test, y_test),
        callbacks=[training_monitor, lr_scheduler],
        verbose=1
    )

    # Checkpoint weights
    os.makedirs("output/checkpoints", exist_ok=True)
    model.save("output/checkpoints/cifar.hdf5")

if __name__ == '__main__':
    main()
