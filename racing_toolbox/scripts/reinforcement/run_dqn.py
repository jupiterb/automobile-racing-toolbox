from stable_baselines3 import DQN
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from conf.example_configuration import get_game_config
from interface.training_local import TrainingLocalGameInterface
from rl.enviroment import RealTimeEnviroment
from rl.final_state.detector import FinalStateDetector
from rl.models.final_value_detecion_params import FinalValueDetectionParameters
from rl.wrappers.observation_squeezing import SqueezingWrapper

def main():
    config = get_game_config()
    interface = TrainingLocalGameInterface(config)
    final_st_det = FinalStateDetector(
        [
            FinalValueDetectionParameters(
                feature_name="speed",
                min_value=2,
                max_value=float("inf"),
                required_repetitions_in_row=20,
                not_final_value_required=True,
            )
        ]
    )
    env = RealTimeEnviroment(interface, final_st_det)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, (50, 100))
    env = FrameStack(env, 4)
    env = SqueezingWrapper(env)
    # check_env(env)

    model = DQN("CnnPolicy", env, verbose=1, buffer_size=10_000, learning_starts=100)
    model.learn(total_timesteps=100_000)

    print("EVAL")
    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()


if __name__ == "__main__":
    main()
