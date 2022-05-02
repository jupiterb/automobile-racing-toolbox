import gym, time
from timeit import default_timer as timer
from pyglet.window import key
import numpy as np


class Runner:
    def __init__(self):
        self._env = gym.make("CarRacing-v0").unwrapped
        self.__action_input = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._last_tick = timer()

    @property
    def action_input(self):
        return self.__action_input.copy()

    @action_input.setter
    def action_input(self, action: np.array):
        if not self._env.action_space.contains(action):
            raise ValueError(
                f"invalid action input for {type(self).__name__}: {action}"
            )
        self.__action_input = action

    def run(self, max_fps: int) -> None:
        self._env.render()
        self._env.viewer.window.on_key_press = self.__key_press
        self._env.viewer.window.on_key_release = self.__key_release
        self._env.reset()

        is_open = True
        while is_open:
            self._tick(max_fps)
            s, r, done, info = self._env.step(self.action_input)
            is_open = self._env.render()
        self._env.close()

    def _tick(self, fps: int):
        elapsed = timer() - self._last_tick
        if 1 / elapsed > fps:
            time.sleep(1 / fps - elapsed)
        self._last_tick = timer()

    def __key_press(self, k, mod):
        a = self.action_input
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = 1.0
        if k == key.UP:
            a[1] = 1.0
        if k == key.DOWN:
            a[2] = 0.8
        self.action_input = a

    def __key_release(self, k, mod):
        a = self.action_input
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == 1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0
        self.action_input = a


if __name__ == "__main__":
    r = Runner()
    r.run(30)
