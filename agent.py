import gymnasium as gym
from wrappers import DiscreteObservationWrapper
import numpy as np


class Agent:
    def __init__(
        self,
        space_dimensions=[5, 5, 5, 5],
        discount_factor=0.9,
        learning_rate=0.1,
        exploration_rate=0.3,
        number_of_episodes=10000,
        number_of_steps=500,
    ):
        self.env = DiscreteObservationWrapper(
            gym.make('CartPole-v1', render_mode='human'), space_dimensions
        )
        self.Q = np.zeros(
            (
                space_dimensions[0],
                space_dimensions[1],
                space_dimensions[2],
                space_dimensions[3],
                2,
            )
        )
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.exploration_rate = self.alpha = exploration_rate
        self.number_of_episodes = number_of_episodes
        self.number_of_steps = number_of_steps

    def greedy_policy(self, current_state):
        return np.argmax(
            self.Q[current_state[0]][current_state[1]][current_state[2]][
                current_state[3]
            ]
        )

    def eps_greedy_policy(self, current_state):
        if np.random.random() < self.alpha:
            return self.env.action_space.sample()
        return self.greedy_policy(current_state)

    def decrease_alpha(self):
        self.alpha -= self.exploration_rate / self.number_of_episodes

    def update_Q(
        self,
        current_state,
        current_action,
        current_reward,
        new_state,
        new_action,
    ):
        current_Q = self.Q[current_state[0]][current_state[1]][
            current_state[2]
        ][current_state[3]][current_action]
        new_Q = self.Q[new_state[0]][new_state[1]][new_state[2]][new_state[3]][
            new_action
        ]
        self.Q[current_state[0]][current_state[1]][current_state[2]][
            current_state[3]
        ][current_action] = current_Q + self.learning_rate * (
            current_reward + self.discount_factor * new_Q - current_Q
        )

    def QLearning(self):
        for episode in range(self.number_of_episodes):
            print("Episódio: ", episode, " Exploração: ", self.alpha )
            state, info = self.env.reset()
            self.decrease_alpha()
            for step in range(self.number_of_steps):
                action = self.eps_greedy_policy(state)
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = self.env.step(action)
                new_state = observation
                new_action = self.greedy_policy(new_state)
                self.update_Q(state, action, reward, new_state, new_action)
                state = new_state
                if terminated or truncated:
                    print('Episode ended at step: ', step)
                    observation, info = self.env.reset()
                    break

    def SARSA(self):
        for episode in range(self.number_of_episodes):
            state, info = self.env.reset()
            action = self.eps_greedy_policy(state)
            self.decrease_alpha()
            for step in range(self.number_of_steps):
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = self.env.step(action)
                new_state = observation
                new_action = self.eps_greedy_policy(new_state)
                self.update_Q(state, action, reward, new_state, new_action)
                state = new_state
                action = new_action
                if terminated or truncated:
                    print('Episode ended at step: ', step)
                    observation, info = self.env.reset()
                    break
