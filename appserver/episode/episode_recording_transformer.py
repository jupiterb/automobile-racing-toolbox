import numpy as np

from schemas import Episode


class EpisodeRecordingTransformer():

    @staticmethod
    def pythonize_screenshots(episode: Episode) -> Episode:
        result_episode = episode.copy()
        if result_episode.recording:
            for state, _ in result_episode.recording.recording:
                if state.screenshot_numpy_array is not None:
                    state.screenshot_python_array = state.screenshot_numpy_array.tolist()
                    state.screenshot_numpy_array = None
        return result_episode

    @staticmethod
    def only_screenshot_shapes(episode: Episode) -> Episode:
        result_episode = episode.copy()
        if result_episode.recording:
            for state, _ in result_episode.recording.recording:
                if state.screenshot_numpy_array is not None:
                    state.screenshot_shape = list(state.screenshot_numpy_array.shape)
                    state.screenshot_numpy_array = None
        return result_episode

    @staticmethod
    def numpyize_screenshots(episode: Episode) -> Episode:
        result_episode = episode.copy()
        if result_episode.recording:
            for state, _ in result_episode.recording.recording:
                if state.screenshot_python_array is not None:
                    state.screenshot_numpy_array = np.array(state.screenshot_python_array)
                    state.screenshot_numpy_array = None
        return result_episode
