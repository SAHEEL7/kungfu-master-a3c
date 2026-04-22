from agent import Agent
from environment import make_env
import numpy as np
import tqdm
import torch
import os
import matplotlib.pyplot as plt

NUM_ENVS = 10
STEPS = 3000

class EnvBatch:
    def __init__(self, n):
        self.envs = [make_env() for _ in range(n)]

    def reset(self):
        return np.array([env.reset()[0] for env in self.envs])

    def step(self, actions):
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        next_states, rewards, dones, infos, _ = zip(*results)
        return np.array(next_states), np.array(rewards), np.array(dones), infos

env = make_env()
agent = Agent(env.action_space.n)

batch = EnvBatch(NUM_ENVS)
states = batch.reset()
episode_rewards = []

for i in tqdm.trange(STEPS):
    actions = agent.act(states)
    next_states, rewards, dones, _ = batch.step(actions)

    rewards *= 0.01
    agent.step(states, actions, rewards, next_states, dones)
    states = next_states

os.makedirs("checkpoints", exist_ok=True)
torch.save(agent.network.state_dict(), "checkpoints/model.pth")
print("Model saved!")
