import torch
import numpy as np
import torch.nn.functional as F
from model import Network

class Agent:
    def __init__(self, action_size, lr=1e-4, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = Network(action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        logits, _ = self.network(state)
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        return np.array([np.random.choice(len(p), p=p) for p in probs])

    def step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward).float().to(self.device)
        done = torch.tensor(done).float().to(self.device)

        logits, value = self.network(state)
        _, next_value = self.network(next_state)

        target = reward + self.gamma * next_value * (1 - done)
        advantage = target - value

        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        logp = log_probs[np.arange(len(action)), action]

        actor_loss = -(logp * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + critic_loss - 0.001 * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
