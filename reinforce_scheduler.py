import numpy as np
import gym
from frap import FRAP

# Create a gym environment for the traffic signal control
class TrafficSignalEnv(gym.Env):
    def __init__(self):
        self.state_dim = 16
        self.action_dim = 4
        self.state = np.zeros(self.state_dim)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.action_dim,))

    def reset(self):
        self.state = np.zeros(self.state_dim)
        return self.state

    def step(self, action):
        # Update the state based on the action
        self.state += action
        reward = -np.sum(self.state)
        done = False
        return self.state, reward, done, {}

# Create a FRAP agent
agent = FRAP(TrafficSignalEnv())

# Train the FRAP agent
for episode in range(1000):
    state = agent.env.reset()
    done = False
    rewards = 0
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = agent.env.step(action)
        rewards += reward
        agent.memory.push(state, action, reward, next_state, done)
        state = next_state
    agent.update()
    print(f'Episode {episode+1}, Reward: {rewards}')
