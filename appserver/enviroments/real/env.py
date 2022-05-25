import gym.spaces
from enviroments.real.interface.abstract import RealGameInterface
from enviroments.real.state.real_state_builder import RealStateBuilder
import numpy as np
from schemas.enviroment.steering import SteeringAction
from schemas import State


Frame = np.ndarray


class RealTimeEnv(gym.Env):
    def __init__(self, game_interface: RealGameInterface):
        self._interface = game_interface

        self.obs_shape = (100, 100, 1)
        self._state_builder = RealStateBuilder(self.obs_shape)
        self._reward_system = ...
        self.__last_frame = None

        self.available_actions = [
            [SteeringAction.FORWARD],
            [SteeringAction.FORWARD, SteeringAction.LEFT],
            [SteeringAction.FORWARD, SteeringAction.RIGHT],
            [SteeringAction.LEFT],
            [SteeringAction.RIGHT],
            [SteeringAction.BREAK]
        ] # None means no action
        self.action_space = gym.spaces.Discrete(len(self.available_actions))
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.obs_shape, dtype=np.uint8
        )

    def reset(self) -> Frame:
        self._interface.reset()
        state = self._fetch_state()
        return state.screenshot_numpy_array

    def step(self, action: int) -> tuple[Frame, float, bool, dict]:
        assert self.action_space.contains(action)

        self._aplly_action(action)
        state = self._fetch_state()
        reward = self._get_reward(state)
        is_done = self._is_final_state(state)
        return state.screenshot_numpy_array, reward, is_done, {}

    def render(self, mode="tgb_array") -> Frame:
        return self.__last_frame

    def _fetch_state(self) -> State:
        frame = self._interface.read_frame()
        return self._build_state_from_frame(frame)

    def _aplly_action(self, action: int) -> None:
        steering_input = self.available_actions[action]
        if steering_input:  # action == None -> do nothing
            self._interface.apply_keyboard_action(steering_input)

    def _build_state_from_frame(self, frame: Frame) -> State:
        self._state_builder.add_features_from_screenshot(frame)
        # self._state_builder.add_velocity_with_ocr(velocity_screenshot)
        result = self._state_builder.get_result()
        return result

    def _get_reward(self, state: State) -> float:
        # TODO: add reward system
        return -1

    def _is_final_state(self, state: State) -> bool:
        # TODO: add end of episode evaluation
        return False
