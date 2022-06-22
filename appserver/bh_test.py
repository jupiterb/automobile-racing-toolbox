from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam

import threading
import time
import cv2
import numpy as np

from pynput.keyboard import Listener, Key

from enviroments.real.interface.local import LocalInterface
from schemas import GameSystemConfiguration, GameGlobalConfiguration
from schemas.enviroment.steering import SteeringAction


class BhScreen(GridLayout):
    def __init__(self, **kwargs):
        super(BhScreen, self).__init__(**kwargs)
        self.cols = 2

        self.add_widget(Label(text="Path to model"))
        self._path_to_model = TextInput(multiline=False, text="weights.hdf5")
        self.add_widget(self._path_to_model)

        self.add_widget(Label(text="Game name"))
        self._game_name = TextInput(multiline=False, text="TrackMania Nations Forever")
        self.add_widget(self._game_name)

        self._running = False
        self._driving = None
        self._keyboard_listener: Listener = Listener(on_press=self._keyboard_callback)
        self._keyboard_listener.start()
        self.add_widget(Label(text="Click [Tab] to run/stop model"))

        self._model = None
        self._load_model_button = Button(text="Load model")
        self._load_model_button.bind(on_press=self._load_model_keras)
        self.add_widget(self._load_model_button)

    def _load_model_keras(self, instance):
        self._model = build_model()
        self._model.load_weights(self._path_to_model.text)

    def _keyboard_callback(self, key):
        if self._model and key == Key.tab:
            if self._running:
                print("stopping model")
                self._running = False
                if self._driving:
                    self._driving.join()
            else:
                print("running model")
                self._running = True
                self._driving = threading.Thread(target=self._run, args=())
                self._driving.start()
        else:
            pass

    def _run(self):
        enviroment_interface = LocalInterface(
            GameGlobalConfiguration(
                process_name=self._game_name.text, apply_grayscale=False
            ),
            GameSystemConfiguration(),
        )
        _ = enviroment_interface.reset()
        input_list = []
        all_actions = [
            SteeringAction.LEFT,
            SteeringAction.RIGHT,
            SteeringAction.FORWARD,
            SteeringAction.BREAK,
        ]
        while self._driving and self._model:
            time.sleep(1 / 10)
            screenshot = enviroment_interface.get_image_input()
            grayscaled = to_grayscale(screenshot[380:730, 10:-10])
            sample = rescale(grayscaled, 150)
            input_list.append(sample)
            if len(input_list) > 4:
                input_list.pop()
                input_nn = np.array([input_list])
                input_nn = np.swapaxes(input_nn, 1, 3)
                input_nn = np.swapaxes(input_nn, 1, 2)
                output = self._model(input_nn)
                enviroment_interface.apply_keyboard_action(
                    [action for i, action in enumerate(all_actions) if output[0][i] > 0]
                )


class BhApp(App):
    def build(self):
        return BhScreen()


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


def rescale(frame, max_side_len):
    scale = max_side_len / np.max(frame.shape)
    target_shape = list((scale * np.array(frame.shape)).astype(np.uint8))
    return cv2.resize(frame, dsize=target_shape[::-1], interpolation=cv2.INTER_CUBIC)


def to_grayscale(frame):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    grayscale_image = np.dot(frame, rgb_weights)
    return grayscale_image


if __name__ == "__main__":
    BhApp().run()
