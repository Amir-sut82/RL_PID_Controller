import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.negative_slope = 0.01

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc_action = nn.Linear(128, action_dim)

        self._initialize_weights()

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=self.negative_slope)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.negative_slope)
        x = F.leaky_relu(self.fc3(x), negative_slope=self.negative_slope)
        x = F.leaky_relu(self.fc4(x), negative_slope=self.negative_slope)
        actions = torch.tanh(self.fc_action(x))
        return actions

    def _initialize_weights(self):
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Linear):
                if 'fc_action' in name:
                    nn.init.uniform_(layer.weight, -3e-3, 3e-3)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                else:
                    nn.init.kaiming_normal_(layer.weight, a=self.negative_slope, nonlinearity="leaky_relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.negative_slope = 0.01

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256 + action_dim, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc_value = nn.Linear(128, 1)

        self._initialize_weights()

    def forward(self, state, action):
        x = F.leaky_relu(self.fc1(state), negative_slope=self.negative_slope)
        x = torch.cat([x, action], dim=1)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.negative_slope)
        x = F.leaky_relu(self.fc3(x), negative_slope=self.negative_slope)
        x = F.leaky_relu(self.fc4(x), negative_slope=self.negative_slope)
        q_value = self.fc_value(x)
        return q_value

    def _initialize_weights(self):
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Linear):
                if 'fc_value' in name:
                    nn.init.uniform_(layer.weight, -3e-3, 3e-3)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                else:
                    nn.init.kaiming_normal_(layer.weight, a=self.negative_slope, nonlinearity="leaky_relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, seed=None):
        self.capacity = capacity
        self.size = 0
        self.pos = 0
        self.rng = np.random.default_rng(seed=seed)

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        return self.size

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2, seed=None):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.rng = np.random.default_rng(seed=seed)
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * self.rng.standard_normal(self.action_dim)
        self.state += dx
        return self.state

class DDPG:
    def __init__(self, state_dim, action_dim,
                 lr_actor=1e-4, lr_critic=1e-3,
                 gamma=0.99, tau=0.005,
                 buffer_size=100000, batch_size=64,
                 noise_sigma=0.2, seed=None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim, seed)

        self.noise = OUNoise(action_dim, sigma=noise_sigma, seed=seed)

        self.total_steps = 0

    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        self.actor.train()

        if add_noise:
            noise = self.noise.sample()
            action = action + noise
            action = np.clip(action, -1.0, 1.0)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def optimize(self):
        if len(self.buffer) < self.batch_size:
            return None, None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return actor_loss.item(), critic_loss.item()

    def _soft_update(self, source, target):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, filepath):
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        print(f"Model loaded from {filepath}")

def train(env, agent, num_episodes=1000, max_steps=500, save_interval=100, save_path='ddpg_model.pth'):
    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        agent.noise.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state, add_noise=True)

            next_state, reward, terminated, truncated, info = env.step(
                torch.tensor([action], dtype=torch.float32)
            )
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, float(done))

            agent.optimize()

            state = next_state
            episode_reward += reward

            if done:
                break

        episode_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")

        if (episode + 1) % save_interval == 0:
            agent.save(save_path)

    return episode_rewards

if __name__ == '__main__':
    from rl_environment import PathFollowingEnv

    env = PathFollowingEnv()
    path = [(1, 0), (2, 0), (3, 1), (3, 2)]
    env.set_path(path)
    env.set_robot_pose(0, 0, 0)

    agent = DDPG(
        state_dim=env.observation_shape,
        action_dim=env.num_actions,
        lr_actor=1e-4,
        lr_critic=1e-3
    )

    rewards = train(env, agent, num_episodes=100, save_path='test_model.pth')
    print(f"Training complete. Final avg reward: {np.mean(rewards[-10:]):.2f}")
