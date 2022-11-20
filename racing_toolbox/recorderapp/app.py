from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button

import argparse
import json
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from racing_toolbox.recorderapp import EpisodeRecordingManager
from racing_toolbox.interface import from_config
from racing_toolbox.interface.config import GameConfiguration
from racing_toolbox.interface.controllers import KeyboardController
from racing_toolbox.interface.capturing import KeyboardCapturing


class RecorderScreen(GridLayout):
    def __init__(self, **kwargs):
        super(RecorderScreen, self).__init__(**kwargs)
        self._game_config: GameConfiguration
        args = get_cli_args()
        with open(args.game_config) as gp:
            self._game_config = GameConfiguration(**json.load(gp))

        self.cols = 2
        self._recording_manager = EpisodeRecordingManager()

        self.add_widget(Label(text="Usser name"))
        self._user_name = TextInput(multiline=False)
        self.add_widget(self._user_name)

        self.add_widget(Label(text="Recording name"))
        self._recording_name = TextInput(text="recording_0")
        self.add_widget(self._recording_name)

        self._start_save_button = Button(text="Start recording")
        self._start_save_button.bind(on_press=self.start_or_save)
        self.add_widget(self._start_save_button)

        self._stop_resume_button = Button(text="Stop recording")
        self._stop_resume_button.bind(on_press=self.stop_or_resume)
        self.add_widget(self._stop_resume_button)

    def start_or_save(self, instance):
        if self._recording_manager.running:
            self._recording_manager.release()
            self._start_save_button.text = "Start recording"
        else:
            interface = from_config(
                self._game_config, KeyboardController, KeyboardCapturing
            )
            self._recording_manager.start(
                interface,
                self._user_name.text,
                self._recording_name.text,
                self._game_config.frequency_per_second,
            )
            self._start_save_button.text = "Save recording"

    def stop_or_resume(self, instance):
        if self._recording_manager.caturing():
            self._recording_manager.stop()
            self._stop_resume_button.text = "Stop recording"
        else:
            self._recording_manager.resume()
            self._stop_resume_button.text = "Resume recording"


class RecorderApp(App):
    def build(self):
        return RecorderScreen()


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game_config",
        type=str,
        default="./config/trackmania/game_config.json",
        help="Path to json with game configuartion",
    )
    return parser.parse_args()


if __name__ == "__main__":
    RecorderApp().run()
