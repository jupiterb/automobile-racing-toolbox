from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput

from pynput.keyboard import Listener, Key

from datetime import datetime

from schemas import (
    GameGlobalConfiguration,
    GameSystemConfiguration,
    ScreenFrame,
    Episode,
)
from episode import EpisodeRecordingManager, InMemoryRecordingsDataService


class RecorderScreen(GridLayout):
    def __init__(self, **kwargs):
        super(RecorderScreen, self).__init__(**kwargs)
        self.cols = 2

        self._block_listener = True
        self._keyboard_listener: Listener = Listener(on_press=self._keyboard_callback)
        self._keyboard_listener.start()

        self._recording_manager = EpisodeRecordingManager()
        self._started = False
        self._is_recorded = False

        self._data_service = InMemoryRecordingsDataService()

        self.add_widget(
            Label(text="Recordoer nickname\nYou need to provide it and click [Enter]")
        )
        self.recorder_name = TextInput(multiline=False)
        self.recorder_name.bind(on_text_validate=self._unblock_listener)
        self.add_widget(self.recorder_name)

        self.add_widget(Label(text="Game name"))
        self.game_name = TextInput(multiline=False, text="TrackMania Nations Forever")
        self.add_widget(self.game_name)

        self.add_widget(Label(text="Click [R] to start/resume or stop recording"))
        self.add_widget(
            Label(
                text="Click [S] to save recording\nClick [D] to delete not saved recording"
            )
        )

    def _keyboard_callback(self, key):
        if not self._block_listener:
            if str(key) == "'r'":
                if self._started:
                    if self._is_recorded:
                        print("Stopped")
                        self._is_recorded = False
                        self._recording_manager.stop()
                    else:
                        print("Resumed")
                        self._is_recorded = True
                        self._recording_manager.resume()
                else:
                    print("Starting recording")
                    self._started = True
                    self._is_recorded = True
                    self._start_recording()
            elif str(key) == "'s'":
                self._save()
            elif str(key) == "'d'":
                print("Recording buffer is cleared")
                self._recording_manager.reset_recording()

    def _start_recording(self):
        global_config = GameGlobalConfiguration(
            process_name=self.game_name.text, apply_grayscale=False
        )
        system_config = GameSystemConfiguration(
            velocity_screen_frame=ScreenFrame(
                top=0.945, bottom=0.995, left=0.92, right=0.985
            )
        )
        self._recording_manager.start(system_config, global_config, 10)

    def _save(self):
        episode = Episode(
            id=f'{self.recorder_name.text}_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}',
            recording=self._recording_manager.release(),
        )
        self._data_service.save("trackmania", episode)
        print("Saved")
        self._started = False
        self._is_recorded = False

    def _unblock_listener(self, instance):
        self._block_listener = False


class RecorderApp(App):
    def build(self):
        return RecorderScreen()


if __name__ == "__main__":
    RecorderApp().run()
