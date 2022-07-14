from conf.example_configuration import get_game_config
from rl.enviroment import RealTimeEnviroment
from interface.local import LocalGameInterface
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from rl.final_state.detector import FinalStateDetector
from rl.models.final_feature_value_detecion_params import (
    FinalValueDetectionParameters,
)


def main():
    config = get_game_config()

    interface = LocalGameInterface(config)
    final_st_det = FinalStateDetector(
        [
            FinalValueDetectionParameters(
                feature_name="speed",
                min_value=2.0,
                max_value=None,
                required_repetitions_in_row=10,
                not_final_value_required=True,
            )
        ]
    )
    env = RealTimeEnviroment(interface, final_st_det)
    check_env(env)

    model = A2C("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=100)

    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()


if __name__ == "__main__":
    main()
