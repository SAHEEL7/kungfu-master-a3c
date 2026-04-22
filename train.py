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

total_reward = 0

for i in tqdm.trange(STEPS):
    actions = agent.act(states)
    next_states, rewards, dones, _ = batch.step(actions)

    rewards *= 0.01
    total_reward += rewards.mean()

    agent.step(states, actions, rewards, next_states, dones)
    states = next_states

    if i % 100 == 0:
        episode_rewards.append(total_reward)
        total_reward = 0

os.makedirs("results/plots", exist_ok=True)

plt.plot(episode_rewards)
plt.title("Training Reward Over Time")
plt.xlabel("Iterations (x100)")
plt.ylabel("Reward")
plt.savefig("results/plots/reward_plot.png")

print("📈 Reward graph saved!")
