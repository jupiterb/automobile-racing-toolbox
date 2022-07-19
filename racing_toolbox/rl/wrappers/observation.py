import gym 
import matplotlib.pyplot as plt
import numpy as np 


class SqueezingWrapper(gym.ObservationWrapper):
    """This wrapper applies np.squeeze to make shape of observation compatible with stb3"""

    def observation(self, observation):
        observation = np.squeeze(observation)
        return observation 


class RescaleWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return observation / 255.0
        