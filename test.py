import torch
from agent import Agent
from environment import make_env

env = make_env()
agent = Agent(env.action_space.n)

agent.network.load_state_dict(torch.load("checkpoints/model.pth"))

state, _ = env.reset()
done = False

while not done:
    action = agent.act([state])[0]
    state, reward, done, _, _ = env.step(action)
