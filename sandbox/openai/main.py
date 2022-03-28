import gym
import matplotlib.pyplot as plt
from wrappers.car_racing import CarRacingWrapper
from wrappers.save_best import SaveOnBestTrainingRewardCallback
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import CnnPolicy
from gym.spaces.discrete import Discrete
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
import os 


LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)


def main():
    env = Monitor(CarRacingWrapper(), filename=LOG_DIR)
    now = datetime.now().strftime("%H_%M_$S")
    save_path = LOG_DIR + "/cnn_carracing_v1" 
    load_path = LOG_DIR + "/cnn_carracing_v0" 

    # train(env, save_path, load=load_path)

    # print("TRAINING FINISHED")

    watch(env, save_path)


def train(env, model_path, load=None):
    if load:
        model = DQN.load(load, env=env)
        print(model)
    else:
        model = DQN(CnnPolicy, env, verbose=1, optimize_memory_usage=False, buffer_size=100_000)

    callback = SaveOnBestTrainingRewardCallback(2000, log_dir=LOG_DIR)

    model.learn(total_timesteps=1_000_000, callback=callback)
    model.save(model_path)
    del model # remove to demonstrate saving and loading


def watch(env, model_path):
    model = DQN.load(model_path)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs.copy())
        print(action)
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__ == '__main__':
    main()
