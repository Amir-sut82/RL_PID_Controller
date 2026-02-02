import numpy as np
import matplotlib.pyplot as plt
from ddpg import DDPG
from rl_environment import PathFollowingEnv
import os
import torch

def generate_random_path(num_waypoints=4, area_size=3.0):
    
    path = [(0.0, 0.0)]


    for _ in range(num_waypoints - 1):
        last_x, last_y = path[-1]
        dx = np.random.uniform(0.5, 1.0)
        dy = np.random.uniform(-0.3, 0.3)
        new_x = np.clip(last_x + dx, 0, area_size)
        new_y = np.clip(last_y + dy, -area_size/2, area_size/2)
        path.append((new_x, new_y))

    return path

def generate_training_paths(num_paths=100):
    paths = []
    for _ in range(num_paths):
        num_waypoints = np.random.randint(3, 6)
        path = generate_random_path(num_waypoints)
        paths.append(path)
    return paths

def train_agent(num_episodes=1000, save_dir='models'):

    os.makedirs(save_dir, exist_ok=True)

    env = PathFollowingEnv()

    agent = DDPG(
        state_dim=env.observation_shape,
        action_dim=env.num_actions,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_size=100000,
        batch_size=64,
        noise_sigma=0.15
    )

    training_paths = generate_training_paths(num_paths=50)

    all_rewards = []
    recent_successes = []

    for episode in range(num_episodes):
        path = training_paths[episode % len(training_paths)]
        env.set_path(path)

        start_x = path[0][0]
        start_y = path[0][1]
        start_yaw = np.random.uniform(-0.2, 0.2)
        env.set_robot_pose(start_x, start_y, start_yaw)

        state, _ = env.reset()
        agent.noise.reset()
        episode_reward = 0
        success = False

        for step in range(env.max_steps):
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
                if env.current_waypoint_idx >= len(path):
                    success = True
                break

        all_rewards.append(episode_reward)
        recent_successes.append(1 if success else 0)

        if len(recent_successes) > 100:
            recent_successes.pop(0)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(all_rewards[-50:])
            success_rate = np.mean(recent_successes) * 100
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward (last 50): {avg_reward:.2f}")
            print(f"  Success Rate (last 100): {success_rate:.1f}%")
            print()

        if (episode + 1) % 200 == 0:
            model_path = os.path.join(save_dir, f'ddpg_episode_{episode+1}.pth')
            agent.save(model_path)

    final_path = os.path.join(save_dir, 'ddpg_final.pth')
    agent.save(final_path)
    print(f"Model saved to {final_path}")

    plot_training_curve(all_rewards, save_dir)

    return agent, all_rewards

def plot_training_curve(rewards, save_dir):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, color='blue')

    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, color='red', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(rewards, bins=50, edgecolor='black')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curve.png'))
    plt.close()
    print(f"Training curve saved to {save_dir}/training_curve.png")

def evaluate_agent(agent, num_episodes=10):
    env = PathFollowingEnv()

    total_rewards = []
    successes = 0

    for episode in range(num_episodes):
        path = generate_random_path(num_waypoints=4)
        env.set_path(path)
        env.set_robot_pose(0, 0, 0)

        state, _ = env.reset()
        episode_reward = 0

        for step in range(env.max_steps):
            action = agent.select_action(state, add_noise=False)

            next_state, reward, terminated, truncated, info = env.step(
                torch.tensor([action], dtype=torch.float32)
            )

            state = next_state
            episode_reward += reward

            if terminated or truncated:
                if env.current_waypoint_idx >= len(path):
                    successes += 1
                break

        total_rewards.append(episode_reward)
        wp_str = f"{env.current_waypoint_idx}/{len(path)}"
        status = "SUCCESS" if env.current_waypoint_idx >= len(path) else "FAIL"
        print(f"Eval Episode {episode + 1}: Reward = {episode_reward:.2f}, Waypoints: {wp_str} [{status}]")

    print(f"\n=== Evaluation Results ===")
    print(f"  Average Reward: {np.mean(total_rewards):.2f}")
    print(f"  Success Rate: {successes/num_episodes*100:.1f}%")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train DDPG for path following')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate trained model')
    parser.add_argument('--model_path', type=str, default='models/ddpg_final.pth', help='Path to model for evaluation')

    args = parser.parse_args()

    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    if args.evaluate:
        env = PathFollowingEnv()
        agent = DDPG(state_dim=env.observation_shape, action_dim=env.num_actions)
        agent.load(args.model_path)
        evaluate_agent(agent, num_episodes=10)
    else:
        agent, rewards = train_agent(num_episodes=args.episodes, save_dir=args.save_dir)
        print("\n" + "="*50)
        print("Training Complete! Running Evaluation...")
        print("="*50 + "\n")
        evaluate_agent(agent, num_episodes=10)