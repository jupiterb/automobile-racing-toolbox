from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button

from pynput.keyboard import Listener, Key

from schemas import (
    GameGlobalConfiguration,
    GameSystemConfiguration,
    ScreenFrame,
    Episode,
)
from episode import EpisodeRecordingManager, InMemoryEpisodesRecordingsDataService


class RecorderScreen(GridLayout):
    def __init__(self, **kwargs):
        super(RecorderScreen, self).__init__(**kwargs)
        self.cols = 2

        self._keyboard_listener: Listener = Listener(on_press=self.keyboard_callback)
        self._keyboard_listener.start()

        self._recording_manager = EpisodeRecordingManager()
        self._started = False
        self._is_recorded = False

        self._data_service = InMemoryEpisodesRecordingsDataService()

        self.add_widget(Label(text="Recording name"))
        self.recording_name = TextInput(multiline=False)
        self.add_widget(self.recording_name)

        self.add_widget(Label(text="Game name"))
        self.game_name = TextInput(multiline=False, text="TrackMania Nations Forever")
        self.add_widget(self.game_name)

        self.add_widget(Label(text="Press [Tab] to start/resume or stop recording"))
        self.save_button = Button(text="Save recording")
        self.save_button.bind(on_press=self.save)
        self.add_widget(self.save_button)

    def keyboard_callback(self, key):
        if key == Key.tab:
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
                self.start_recording()

    def start_recording(self):
        global_config = GameGlobalConfiguration(
            process_name=self.game_name.text, apply_grayscale=False
        )
        system_config = GameSystemConfiguration(
            velocity_screen_frame=ScreenFrame(
                top=0.945, bottom=0.995, left=0.92, right=0.985
            )
        )
        self._recording_manager.start(system_config, global_config, 10)

    def save(self, instance):
        episode = Episode(
            id=self.recording_name.text, recording=self._recording_manager.release()
        )
        self._data_service.save("trackmania", episode)
        print("Saved")
        r = self._data_service.get_episode("trackmania", episode.id)
        print(r.recording.recording[100][2])
        print(r.recording.recording[100][0].shape)


class RecorderApp(App):
    def build(self):
        return RecorderScreen()


if __name__ == "__main__":
    RecorderApp().run()
