import gymnasium as gym
from wrappers import DiscreteObservationWrapper
import numpy as np

# Agent that uses only 2 observation variables instead of 4
class Agent:
    def __init__(
        self,
        space_dimensions=[5, 5, 5, 5],
        discount_factor=0.9,
        learning_rate=0.1,
        exploration_rate=0.05,
        number_of_episodes=25000,
        number_of_steps=1000,
        exploration='constant',
    ):
        self.env = DiscreteObservationWrapper(
            gym.make('CartPole-v1'),
            space_dimensions,
        )
        self.Q = np.zeros(
            (
                space_dimensions[2],
                space_dimensions[3],
                2,
            )
        )
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.exploration_rate = self.eps = exploration_rate
        self.number_of_episodes = number_of_episodes
        self.number_of_steps = number_of_steps
        self.results = []
        self.explored = 0
        # self.success = 0
        self.space_dimensions = space_dimensions
        self.exploration = exploration

    def greedy_policy(self, current_state):
        return np.argmax(self.Q[current_state[2]][current_state[3]])

    def eps_greedy_policy(self, current_state):
        if np.random.random() < self.eps:
            self.explored += 1
            return self.env.action_space.sample()
        return self.greedy_policy(current_state)

    def decrease_eps(self, episode):
        if self.exploration == 'constant_decrease':
            self.eps -= self.exploration_rate / self.number_of_episodes
        if self.exploration == 'episode':
            self.eps = 1 / (episode + 1)

    def update_Q(
        self,
        current_state,
        current_action,
        current_reward,
        new_state,
        new_action,
    ):
        current_Q = self.Q[current_state[2]][current_state[3]][current_action]
        new_Q = self.Q[new_state[2]][new_state[3]][new_action]
        self.Q[current_state[2]][current_state[3]][
            current_action
        ] = current_Q + self.learning_rate * (
            current_reward + self.discount_factor * new_Q - current_Q
        )

    def QLearning(self):
        for episode in range(self.number_of_episodes):
            print('Episódio: ', episode)
            state, info = self.env.reset()
            self.explored = 0
            self.decrease_eps(episode)
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
                    print(
                        'Episode ended at step: ',
                        step,
                        ' Explored times: ',
                        self.explored,
                    )
                    self.results.append(int(step))
                    observation, info = self.env.reset()
                    break
            if episode == self.number_of_episodes - 1:
                results = np.array(self.results)
                np.savetxt(
                    f'./[Q][{self.learning_rate}][{self.exploration_rate}]{str(self.space_dimensions)}[{self.discount_factor}].csv',
                    results,
                    delimiter=',',
                    fmt='%1g',
                )

    def SARSA(self):
        for episode in range(self.number_of_episodes):
            print('Episódio: ', episode)
            state, info = self.env.reset()
            action = self.eps_greedy_policy(state)
            self.explored = 0
            self.decrease_eps(episode)
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
                    print(
                        'Episode ended at step: ',
                        step,
                        ' Explored times: ',
                        self.explored,
                    )
                    self.results.append(step)
                    observation, info = self.env.reset()
                    break
            if episode == self.number_of_episodes - 1:
                results = np.array(self.results)
                np.savetxt(
                    f'./[SARSA][{self.learning_rate}][{self.exploration_rate}]{str(self.space_dimensions)}[{self.discount_factor}].csv',
                    results,
                    delimiter=',',
                    fmt='%1g',
                )
