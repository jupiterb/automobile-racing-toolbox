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
from stable_baselines3.common.env_checker import check_env

LOG_DIR = "./logs/policy_grayscale"
TB_DIR = LOG_DIR + "/tensorboard"
TB_LOG_NAME = "v0"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)



def main():
    env = Monitor(CarRacingWrapper(), filename=LOG_DIR)
    print(check_env(env))
    now = datetime.now().strftime("%H_%M_$S")
    save_path = LOG_DIR + "/cnn_carracing_gray_v1" 
    load_path = LOG_DIR + "/best_model" 

    train(env, save_path)

    print("TRAINING FINISHED")

    # watch(env, save_path)


def train(env, model_path, load=None):
    if load:
        model = DQN.load(load, env=env)
        model.tensorboard_log = model.tensorboard_log or TB_DIR
        print(model)
    else:
        model = DQN(CnnPolicy, env, verbose=1, optimize_memory_usage=False, buffer_size=100_000, tensorboard_log=TB_DIR)

    callback = SaveOnBestTrainingRewardCallback(2000, log_dir=LOG_DIR)

    model.learn(total_timesteps=500_000, callback=callback, tb_log_name=TB_LOG_NAME, reset_num_timesteps=False)
    model.save(model_path)
    del model 


def watch(env, model_path):
    model = DQN.load(model_path)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs.copy())
        print(action)
        obs, rewards, dones, info = env.step(action)
        env.render()

def plot():
    from stable_baselines3.common import results_plotter

    # Helper from the library
    results_plotter.plot_results([LOG_DIR], 1e3, results_plotter.X_TIMESTEPS, "TD3 LunarLander", figsize=(10, 10))
    plt.show()

if __name__ == '__main__':
    main()
