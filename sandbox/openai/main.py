import gym


def main():
    env = gym.make('CarRacing-v1')
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample()) 
    env.close()


if __name__ == '__main__':
    main()