import gym
import numpy as np 
import cv2


class CarRacingWrapper:
    def __init__(self):
        self.env = gym.make('CarRacing-v1')
        self.env.reset()

    @property 
    def action_space(self):
        return self.env.action_space 

    def step(self, action):
        self.env.step(action)

    def render(self) -> np.ndarray:   
        state = self.env.render(mode="state_pixels")
        clipped = np.array(state, dtype=np.uint8)[:-15, :]
        gray = cv2.cvtColor(clipped, cv2.COLOR_RGB2GRAY)
        return gray

    def close(self):
        self.env.close()

    
