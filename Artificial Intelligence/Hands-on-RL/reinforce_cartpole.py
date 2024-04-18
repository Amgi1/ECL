import gym
import torch
from torch import nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(128, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the environment
env = gym.make("CartPole-v1")#, render_mode="human")

action_size = env.action_space.n
state, info = env.reset()
state_size = len(state)

# Set up policy network and optimizer
policy_net = PolicyNetwork(state_size, action_size).to(device)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=5e-3)

NUMBER_OF_EPISODES = 500
GAMMA = 0.99

total_rewards = []

for episode in range(NUMBER_OF_EPISODES):
    observation = env.reset()
    buffer = []

    while True:
        # Compute action probabilities
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs = policy_net(state_tensor)

        # Sample action based on probabilities
        action_dist = Categorical(action_probs)
        action = action_dist.sample().item()

        # Store action probability in the buffer
        buffer.append((action, torch.log(action_probs[0, action])))

        # Step the environment with the selected action
        next_state, reward, done, truncated, _ = env.step(action)

        # Store the return in the buffer
        buffer[-1] += (reward,)
        
        state = next_state

        if done or truncated:
            # Compute discounted returns
            actions, log_probs, rewards = zip(*buffer)

            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + GAMMA * R
                returns.insert(0, R)
            total_rewards.append(R)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Update the policy
            optimizer.zero_grad()
            loss = -sum([log_probs[i] * returns[i] for i in range(len(returns))])
            loss.backward()
            optimizer.step()

            print(f"Episode: {episode} \t Loss : {loss.detach()} \t Reward : {R}")
            break

env.close()

plt.plot(total_rewards)
plt.show()
