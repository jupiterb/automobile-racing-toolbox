import gym
from enviroments.real.interface.abstract import RealGameInterface
from enviroments.real.state import RealStateBuilder
import numpy as np
from schemas.enviroment.steering import SteeringAction
from schemas import State, GameGlobalConfiguration
from typing import Optional
from collections import deque

Frame = Optional[np.ndarray]


class RealTimeEnv(gym.Env):
    def __init__(
        self,
        game_interface: RealGameInterface,
        global_configuration: GameGlobalConfiguration,
    ):
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
            low=0,
            high=255,
            shape=global_configuration.observation_shape,
            dtype=np.uint8,
        )
        self._interface = game_interface
        self._state_builder = RealStateBuilder(global_configuration)
        self._reward_system = ...
        self.__last_rewards = deque(maxlen=30)
        self.__last_frame = None


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
        self._state_builder.add_features_from_screenshot(
            self._interface.get_image_input()
        )
        self._state_builder.add_velocity(self._interface.get_velocity_input())
        return self._state_builder.build()

    def _aplly_action(self, action: int) -> None:
        steering_input = self.available_actions[action]
        if steering_input:  # action == None -> do nothing
            self._interface.apply_keyboard_action(steering_input)

    def _get_reward(self, state: State) -> float:
        # TODO: add reward system
        print(state.velocity)
        self.__last_rewards.append(state.velocity)
        return float(state.velocity)

    def _is_final_state(self, state: State) -> bool:
        if len(self.__last_rewards) == self.__last_rewards.maxlen:
            m = np.mean(self.__last_rewards)
            std = np.std(self.__last_rewards)
            if m < 15 and std < 10:
                print("restart")
                return True
        return False
