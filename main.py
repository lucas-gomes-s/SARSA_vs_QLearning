from agent import Agent
from agent_v2 import Agent as AgentV2


agent = AgentV2(
    [2, 2, 40, 40], exploration_rate=0.2, exploration='constant_decrease'
)

agent.QLearning()

agent = AgentV2(
    [2, 2, 40, 40], exploration_rate=0.1, exploration='constant_decrease'
)

agent.QLearning()

agent = AgentV2(
    [2, 2, 40, 40], exploration_rate=0.05, exploration='constant_decrease'
)

agent.QLearning()
