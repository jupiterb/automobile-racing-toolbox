from enviroments.real.env import RealTimeEnv
from enviroments.real.interface.local import LocalInterface
from schemas.game.game_global_configuration import GameGlobalConfiguration
from schemas.game.game_system_configuration import GameSystemConfiguration
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env


def main():
    gconfig = GameGlobalConfiguration(process_name="Trackmania Nations Forever")
    lconfig = GameSystemConfiguration()

    interface = LocalInterface(gconfig, lconfig)
    env = RealTimeEnv(interface, gconfig)
    check_env(env)

    model = A2C("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=100)

    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # env.render()
        if done:
            obs = env.reset()
