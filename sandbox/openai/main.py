import gym
import matplotlib.pyplot as plt
from wrappers.car_racing import CarRacingWrapper


def main():
    env = CarRacingWrapper()
    for _ in range(50):
        state = env.render()
        env.step(env.action_space.sample()) 
    env.close()

    print(state.shape)
    plt.imshow(state, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()