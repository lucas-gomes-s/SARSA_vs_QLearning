import gymnasium as gym
from sklearn.preprocessing import KBinsDiscretizer


class DiscreteObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, steps):
        gym.ObservationWrapper.__init__(self, env)
        self.steps = steps

    def observation(self, observation):

        discretizer = KBinsDiscretizer(n_bins=self.steps, encode='ordinal')
        discretizer.fit([[-2.4, -3.8, -0.3, -4], [2.4, 3.8, 0.3, 4]])
        observation = discretizer.transform([observation])
        observation = list(map(int, observation[0]))
        return observation
