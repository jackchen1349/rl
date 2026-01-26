import numpy as np
import random
import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=True)

parser.add_argument(
    '--save_path', default=None, help='folder to save if mode == train else model path,'
    'qnet will be saved once target net update'
)
parser.add_argument('--seed', help='random seed', type=int, default=0)
parser.add_argument('--env_id', default='FrozenLake-v1', help='environment id')
parser.add_argument('--device', type=str, default='cpu' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--render', type=str, default='human', help='render the environment')
parser.add_argument('--episodes', type=int, default=1000, help='number of episodes to train/test')
args = parser.parse_args()


# 设置随机种子，确保实验可复现
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class SarsaAgent:
    def __init__(self, state_dim, action_dim, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99, device='cpu'):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_table = np.zeros([state_dim, action_dim])
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def choose_action(self, state):
        """ε-贪婪策略选择动作"""
        if random.uniform(0, 1) < self.exploration_rate:
            return np.random.randint(self.action_dim)
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_value(self, state, action, reward, next_state, next_action):
        """更新Q值"""
        td_target = reward + self.discount_factor * self.q_table[next_state, next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

    def decay_exploration(self):
        """衰减探索率"""
        self.exploration_rate *= self.exploration_decay


class QLearningAgent:
    def __init__(self, env, state_dim, action_dim, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99, device='cpu'):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_table = np.zeros([state_dim, action_dim])
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def choose_action(self, state):
        """ε-贪婪策略选择动作"""
        if random.uniform(0, 1) < self.exploration_rate:
            return np.random.randint(self.action_dim)
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_value(self, state, action, reward, next_state):
        """更新Q值"""
        self.q_table[state, action] += self.learning_rate * (
            reward + self.discount_factor * np.max(self.q_table[next_state, :]) - self.q_table[state, action]
        )

    def decay_exploration(self):
        """衰减探索率"""
        self.exploration_rate *= self.exploration_decay


def train_agent(episodes=1000):
    env = gym.make(args.env_id, desc=None, map_name="4x4", is_slippery=False, render_mode=args.render)
    state_dim = env.observation_space.n  # 状态维度
    action_dim = env.action_space.n            # 动作维度
    print(f"状态空间维度: {state_dim}, 动作空间大小: {action_dim}")
    device = torch.device(args.device)
    agent = QLearningAgent(env, state_dim, action_dim, device=device)
    writer = SummaryWriter(log_dir='./logs/Q_learning_Sarsa')
    frame_idx = 0

    for episode in range(episodes):
        state, _ = env.reset(seed=args.seed)
        done = False
        episode_return = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            if args.env_id == 'FrozenLake-v1':
                if state == next_state:
                    reward = -1  # 防止智能体停留在原地
                if reward == 0 and done:
                    reward = -1  # 未到达目标，给予惩罚
            episode_return += reward
            writer.add_scalar('Reward per Step', reward, frame_idx)
            frame_idx += 1
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            if done or truncated:
                break
        print(f"Episode {episode + 1}/{episodes}, Return: {episode_return}, Exploration Rate: {agent.exploration_rate:.4f}")
        print(f"当前Q表:\n{agent.q_table}")
        agent.decay_exploration()
        writer.add_scalar('Return per Episode', episode_return, episode)

    return agent

if __name__ == "__main__":
    trained_agent = train_agent(episodes=args.episodes)
    print("训练完成后的Q表：")
    print(trained_agent.q_table)
