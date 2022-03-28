import gym
import numpy as np 
import cv2


class CarRacingWrapper(gym.Wrapper):
    def __init__(self):
        self.env = gym.make('CarRacing-v0')
        self.env.reset()
        self.action_space_mapping = {
            0: [-1, 0, 0],
            1: [0, 0, 0],
            2: [1, 0, 0],
            3: [0, 1, 0],
            4: [0, 0, 1],
        }
        self.action_space_mapping = {
            key: np.array(val, dtype=np.uint8) 
            for key, val in self.action_space_mapping.items()
        }
    
    @property
    def action_space(self):
        return gym.spaces.discrete.Discrete(5)

    def step(self, action):
        action = int(action)
        action_box = self.action_space_mapping[action]
        obs, *data = self.env.step(action_box)
        return obs, *data

    def process_state(self, state: np.ndarray):
        clipped = np.array(state, dtype=np.uint8)[:-15, :]
        gray = cv2.cvtColor(clipped, cv2.COLOR_RGB2GRAY)
        return gray

    
