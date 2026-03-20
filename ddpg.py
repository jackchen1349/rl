import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from datetime import datetime
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
import collections


parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--seed', help='random seed', type=int, default=0)
parser.add_argument('--env_id', default='Pendulum-v1', help='CartPole-v1 or Pendulum-v1')
parser.add_argument('--gamma', type=float, default=0.98)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_episodes', type=int, default=300)
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--render', type=str, default='rgb_array', help='rgb_array or human')
args = parser.parse_args()


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 双端队列，自动淘汰旧经验

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # 将数据整理成按列堆叠的张量，便于神经网络批量处理
        state, action, reward, next_state, done = zip(*transitions)
        return (np.array(state), np.array(action), reward, np.array(next_state), done)

    def size(self):
        return len(self.buffer)


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1) # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device,  theta=0.15, dt=1e-2):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device
        self.theta = theta      # 均值回归速度
        self.dt = dt            # 时间步长
        self.ou_noise = np.zeros(action_dim)  # 初始化OU噪声状态

    def take_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = self.actor(state).item()
        # 更新OU噪声
        self.ou_noise += self.theta * (0 - self.ou_noise) * self.dt + \
                         self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
        # 给动作添加噪声，增加探索
        # action = action + self.sigma * np.random.randn(self.action_dim)
        action = action + self.ou_noise
        action = np.clip(action, -self.actor.action_bound, self.actor.action_bound)  # 确保动作在环境允许的范围内
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络

    def reset_noise(self):
        """重置噪声状态(通常在episode开始时调用)"""
        self.ou_noise = np.zeros(self.action_dim)


def train():
    writer = SummaryWriter(log_dir='logs/ddpg')
    # 设置随机种子，确保实验可复现
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env = gym.make(args.env_id, render_mode=args.render)
    actor_lr = 3e-4
    critic_lr = 3e-3
    hidden_dim = 64
    gamma = 0.98
    tau = 0.005  # 软更新参数
    buffer_size = 10000
    minimal_size = 1000
    batch_size = 64
    sigma = 0.01  # 高斯噪声标准差
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]  # 动作最大值
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")

    agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)
    return_list = []

    for i in range(10):
        with tqdm(total=int(args.num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(args.num_episodes/10)):
                state, _ = env.reset(seed=args.seed)
                episode_return = 0
                done = False
                agent.reset_noise()
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)

                    if done or truncated:
                        break
                return_list.append(episode_return)
                writer.add_scalar('episode_reward', episode_return, i_episode + i*int(args.num_episodes/10))
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (args.num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    print("训练完成！")
    path = os.path.join('model', '_'.join(["DDPG", args.env_id]))
    if not os.path.exists(path):
        os.makedirs(path)
    date_str = datetime.now().strftime("%Y%m%d")
    torch.save(agent.actor.state_dict(), f"{path}/actor_{date_str}.pth")
    torch.save(agent.critic.state_dict(), f"{path}/critic_{date_str}.pth")

    return return_list

if __name__ == "__main__":
    if args.train:
        train()
    if args.test:
        env = gym.make(args.env_id, render_mode=args.render)
        date_str = datetime.now().strftime("%Y%m%d")

        actor_lr = 3e-4
        critic_lr = 3e-3
        hidden_dim = 64
        gamma = 0.98
        tau = 0.005  # 软更新参数
        buffer_size = 10000
        minimal_size = 1000
        batch_size = 64
        sigma = 0.00  # 高斯噪声标准差
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high[0]  # 动作最大值
        device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")

        agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device, theta=0.0, dt=0)

        path = os.path.join('model', '_'.join(["DDPG", args.env_id]))
        agent.actor.load_state_dict(torch.load(f"{path}/actor_{date_str}.pth"))
        agent.critic.load_state_dict(torch.load(f"{path}/critic_{date_str}.pth"))
      
        for i_episode in range(100):
            episode_return = 0
            state, _ = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                state = next_state
                episode_return += reward
                if done or truncated:
                    break
            print('Episode: {}/{}  Return: {:.3f}'.format(i_episode + 1, 100, episode_return))