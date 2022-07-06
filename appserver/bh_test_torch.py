from pyexpat import model
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput

import torch
from torch import nn


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
        self._path_to_model = TextInput(multiline=False, text="47-98-acc68")
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

        data = torch.load(self._path_to_model.text, map_location=torch.device("cpu"))

        layers = data["activations"]
        layers_out = data["out_channels"]
        kernel_sizes = data["kernel_sizes"]
        strides = [4, 2, 1]

        print(data.keys())

        print(
            f"{self._path_to_model.text}\n{layers}\n{layers_out}\n{kernel_sizes}\n\n#######################\n"
        )

        self._model = NeuralNetwork(layers, layers_out, kernel_sizes, strides)
        self._model.load_state_dict(data["model_state_dict"])

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
        actions_mapping = {
            0: (0, 0, 0, 0),
            1: (1, 0, 0, 0),
            2: (0, 1, 0, 0),
            3: (0, 0, 1, 0),
            4: (0, 0, 0, 1),
            5: (1, 0, 1, 0),
            6: (0, 1, 1, 0),
            7: (1, 0, 0, 1),
            8: (0, 1, 0, 1),
        }
        while self._running and self._model:
            time.sleep(1 / 10)
            screenshot = enviroment_interface.get_image_input()
            grayscaled = to_grayscale(screenshot[380:730, 10:-10])
            sample = rescale(grayscaled, 150)
            input_list.append(sample)

            if len(input_list) > 4:
                input_list.pop(0)
                input_nn = torch.tensor(np.array([input_list]), dtype=torch.float32)
                output = self._model(input_nn)[0].detach().numpy()
                action_ixs = actions_mapping[np.argmax(output)]
                actions = [
                    action for i, action in enumerate(all_actions) if action_ixs[i] == 1
                ]
                enviroment_interface.apply_keyboard_action(actions)


class BhApp(App):
    def build(self):
        return BhScreen()


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        conv_layers_types: list[str],
        conv_layers_outputs: list[int],
        kernel_sizes: list[int],
        strides: list[int],
    ):
        super(NeuralNetwork, self).__init__()

        inputs = 4
        outputs = 9
        convw, convh = 53, 150
        layers = []

        for layer_type, layer_outputs, kernel_size, stride in zip(
            conv_layers_types, conv_layers_outputs, kernel_sizes, strides
        ):
            layers.append(
                nn.Conv2d(
                    inputs,
                    layer_outputs,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=0,
                )
            )
            if layer_type == "relu":
                layers.append(nn.ReLU())
            elif layer_type == "tanh":
                layers.append(nn.Tanh())
            elif layer_type == "sigmoid":
                layers.append(nn.Sigmoid())
            inputs = layer_outputs
            convw = (convw - kernel_size) // stride + 1
            convh = (convh - kernel_size) // stride + 1

        layers.append(nn.Flatten())
        layers.append(nn.Linear(inputs * convw * convh, outputs))
        layers.append(nn.Sigmoid())

        self.layers_sequence = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.layers_sequence(x)
        return logits


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
