

import argparse
import gym

from ray.rllib.env.policy_client import PolicyClient
from scripts.reinforcement.run_dqn import get_configuration, setup_env

parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-train", action="store_true", help="Whether to disable training."
)
parser.add_argument(
    "--inference-mode", type=str, default="local", choices=["local", "remote"]
)
parser.add_argument(
    "--off-policy",
    action="store_true",
    help="Whether to compute random actions instead of on-policy "
    "(Policy-computed) ones.",
)
parser.add_argument(
    "--stop-reward",
    type=float,
    default=9999,
    help="Stop once the specified reward is reached.",
)
parser.add_argument(
    "--port", type=int, default=9900, help="The port to use (on localhost)."
)

if __name__ == "__main__":

    game_conf, obs_conf, rew_conf, train_conf = get_configuration()

    env = setup_env(game_conf, rew_conf, obs_conf)

    args = parser.parse_args()

    # The following line is the only instance, where an actual env will
    # be created in this entire example (including the server side!).
    # This is to demonstrate that RLlib does not require you to create
    # unnecessary env objects within the PolicyClient/Server objects, but
    # that only this following env and the loop below runs the entire
    # training process.
    # env = gym.make("CartPole-v0")

    # If server has n workers, all ports between 9900 and 990[n-1] should
    # be listened on. E.g. if server has num_workers=2, try 9900 or 9901.
    # Note that no config is needed in this script as it will be defined
    # on and sent from the server.
    client = PolicyClient(
        f"http://127.0.0.1:{args.port}", inference_mode=args.inference_mode
    )

    # In the following, we will use our external environment (the CartPole
    # env we created above) in connection with the PolicyClient to query
    # actions (from the server if "remote"; if "local" we'll compute them
    # on this client side), and send back observations and rewards.

    # Start a new episode.
    obs = env.reset()
    eid = client.start_episode(training_enabled=not args.no_train)

    rewards = 0.0
    while True:
        # Compute an action randomly (off-policy) and log it.
        if args.off_policy:
            action = env.action_space.sample()
            client.log_action(eid, obs, action)
        # Compute an action locally or remotely (on server).
        # No need to log it here as the action
        else:
            action = client.get_action(eid, obs)

        # Perform a step in the external simulator (env).
        obs, reward, done, info = env.step(action)
        rewards += reward

        # Log next-obs, rewards, and infos.
        client.log_returns(eid, reward, info=info)

        # Reset the episode if done.
        if done:
            print("Total reward:", rewards)
            if rewards >= args.stop_reward:
                print("Target reward achieved, exiting")
                exit(0)

            rewards = 0.0

            # End the old episode.
            client.end_episode(eid, obs)

            # Start a new episode.
            obs = env.reset()
            eid = client.start_episode(training_enabled=not args.no_train)
