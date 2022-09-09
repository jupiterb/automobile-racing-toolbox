from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from racing_toolbox.interface import FullLocalGameInterface
from racing_toolbox.recorderapp import EpisodeRecordingManager
from racing_toolbox.conf import get_game_config


class RecorderScreen(GridLayout):
    def __init__(self, **kwargs):
        super(RecorderScreen, self).__init__(**kwargs)
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
        if self._recording_manager.running():
            self._recording_manager.release()
            self._start_save_button.text = "Start recording"
        else:
            self._recording_manager.start(
                FullLocalGameInterface(get_game_config()),
                self._user_name.text,
                self._recording_name.text,
                get_game_config().frequency_per_second,
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


if __name__ == "__main__":
    RecorderApp().run()
