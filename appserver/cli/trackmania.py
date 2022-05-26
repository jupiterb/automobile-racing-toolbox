from schemas.training.training import Training
from enviroments.real.env import RealTimeEnv
from enviroments.real.interface.local import LocalInterface
from schemas.game.game_global_configuration import GameGlobalConfiguration
from schemas.game.game_system_configuration import GameSystemConfiguration
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from training.training_manager import _run_new_training

def main():
    gconfig = GameGlobalConfiguration(process_name="TrackMania Nations Forever")
    lconfig = GameSystemConfiguration()

    interface = LocalInterface(gconfig, lconfig)
    env = RealTimeEnv(interface, gconfig)
    training = Training(id="demo", description="foo")
    _run_new_training(gconfig, lconfig, training)
    # check_env(env)

    # model = A2C("CnnPolicy", env, verbose=1)
    # model.learn(total_timesteps=100)

    # obs = env.reset()
    # for i in range(1000):
    #     action, _state = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     # env.render()
    #     if done:
    #         obs = env.reset()
