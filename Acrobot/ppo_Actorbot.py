import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt

# Hyperparameters
STATE_DIM = 6
HIDDEN_DIM = 64
ACTION_DIM = 3
ACTOR_LR = 1e-4
CRITIC_LR = 5e-3
LAMBDA = 0.95
EPOCHS = 10
EPS = 0.2
GAMMA = 0.98
EPISODES = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    def __init__(
        self,
        state_dim=STATE_DIM,
        hidden_dim=HIDDEN_DIM,
        action_dim=ACTION_DIM,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        lmbda=LAMBDA,
        epochs=EPOCHS,
        eps=EPS,
        gamma=GAMMA,
        device=device,
    ):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action_probs = self.actor(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()

    def cal_advantages(self, td_errors):
        td_errors = td_errors.cpu().detach().numpy().tolist()
        advantages = []
        adv = 0.0
        for td_error in td_errors[::-1]:
            adv = self.gamma * self.lmbda * adv + td_error
            advantages.append(adv)
        advantages.reverse()
        return torch.tensor(advantages, dtype=torch.float).to(self.device)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(
            self.device
        )
        actions = (
            torch.tensor(transition_dict["actions"], dtype=torch.long)
            .view(-1, 1)
            .to(self.device)
        )
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float
        ).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        # 计算优势
        td_targets = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_errors = (td_targets - self.critic(states)).detach().squeeze()
        advantages = self.cal_advantages(td_errors)

        # 更新策略网络
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            new_log_probs = torch.log(self.actor(states).gather(1, actions))
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * advantages
            ).detach()
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_targets.detach())
            )

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def save(self):
        torch.save(self.actor.state_dict(), "Actorbot_actor.pth")
        torch.save(self.critic.state_dict(), "Actorbot_critic.pth")
        print(f"Model saved")

    def load(self, actor_path, critic_path):
        self.actor.load_state_dict(
            torch.load(actor_path, map_location=torch.device("cpu"))
        )
        self.critic.load_state_dict(
            torch.load(critic_path, map_location=torch.device("cpu"))
        )
        self.actor.eval()
        self.critic.eval()
        print(f"Model loaded")


def Train(ppo):
    env = gym.make("Acrobot-v1")
    max_steps = env.spec.max_episode_steps
    scores = []
    ave_scores = []
    scores_window = deque(maxlen=10)

    for i in range(EPISODES):
        state = env.reset()[0]
        score = 0
        transition_dict = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
        }
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = ppo.take_action(state)
            next_state, reward, done, _, _ = env.step(action)
            transition_dict["states"].append(state)
            transition_dict["actions"].append(action)
            transition_dict["rewards"].append(reward)
            transition_dict["next_states"].append(next_state)
            transition_dict["dones"].append(done)
            state = next_state
            score += reward
            steps += 1

        if steps < max_steps:
            print(f"Episode {i} finished after {steps} timesteps")
        else:
            print(f"Episode {i} did not finish")

        scores.append(score)
        scores_window.append(score)
        ppo.update(transition_dict)

        if (i + 1) % 10 == 0:
            print(f"Episode {i}, Average Score: {np.mean(scores_window)}")
            ave_scores.append(np.mean(scores_window))
            ppo.save()
    env.close()
    return scores, ave_scores


def draw(scores, ave_scores):
    plt.figure(figsize=(10, 6))
    plt.plot([i for i in range(len(scores))], scores)
    plt.xlabel("Episodes")
    plt.ylabel("Scores")
    plt.title("Scores of Each Episode")
    # plt.show()
    plt.savefig("scores.png")
    plt.figure(figsize=(10, 6))
    plt.plot([i for i in range(len(ave_scores))], ave_scores)
    plt.xlabel("Episodes")
    plt.ylabel("Average Scores")
    plt.title("Average Scores of 10 Episodes")
    # plt.show()
    plt.savefig("ave_scores.png")


if __name__ == "__main__":
    ppo = PPO()
    scores, ave_scores = Train(ppo)
    draw(scores, ave_scores)
