from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
import tables
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

PATH = "tmnf_1os.hdf5"
CHECKPOINTS_PATH = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"

def main():
    images, actions = get_data()
    X_train, X_test, y_train, y_test = train_test_split(
        images, actions, test_size=0.15, random_state=42
    )
    model = build_model()

    model.compile(optimizer="adam", loss=BinaryCrossentropy(), metrics=["accuracy"])

    checkpoint = ModelCheckpoint(CHECKPOINTS_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    history = model.fit(
        X_train,
        y_train,
        epochs=10_000,
        batch_size=128,
        validation_data=(X_test, y_test),
        verbose=True,
        callbacks=callbacks_list,
        verbose=1,
    )


def build_model():
    model = Sequential()

    # Conv Layers
    model.add(Conv2D(32, (8, 8), strides=4, padding="same", input_shape=(53, 150, 4)))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (4, 4), strides=2, padding="same"))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (3, 3), strides=1, padding="same"))
    model.add(Activation("relu"))
    model.add(Flatten())

    # FC Layers
    model.add(Dense(64, activation="relu"))
    model.add(Dense(4, activation="sigmoid"))

    model.compile(loss=BinaryCrossentropy, optimizer=Adam())
    return model


def get_data():
    hdf5_file = tables.open_file(PATH, mode="r")
    images = np.array(hdf5_file.root.images[:], dtype=np.float32) / 255
    actions = np.array(hdf5_file.root.actions[:], np.float32)
    hdf5_file.close()
    return images, actions


if __name__ == '__main__':
    main()