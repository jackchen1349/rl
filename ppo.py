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


parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--seed', help='random seed', type=int, default=0)
parser.add_argument('--env_id', default='CartPole-v1', help='CartPole-v1 or Pendulum-v1')
parser.add_argument('--gamma', type=float, default=0.98)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_episodes', type=int, default=2000)
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--render', type=str, default='rgb_array', help='rgb_array or human')
args = parser.parse_args()



def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states_array = np.array(transition_dict['states'])
        states = torch.tensor(states_array, dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states_array = np.array(transition_dict['next_states'])
        next_states = torch.tensor(next_states_array,
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std


class PPOContinuous:
    ''' 处理连续动作的PPO算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim,
                                         action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return [action.item()]

    def update(self, transition_dict):
        states_array = np.array(transition_dict['states'])
        states = torch.tensor(states_array,
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states_array = np.array(transition_dict['next_states'])
        next_states = torch.tensor(next_states_array,
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        rewards = (rewards + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


def train_discrete():
    writer = SummaryWriter(log_dir='logs/ppo/discrete')
    # 设置随机种子，确保实验可复现
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env = gym.make(args.env_id, render_mode=args.render)
    actor_lr = 1e-3
    critic_lr = 1e-2
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)
    return_list = []

    for i in range(10):
        with tqdm(total=int(args.num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(args.num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, _ = env.reset(seed=args.seed)
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                    if done or truncated:
                        break
                return_list.append(episode_return)
                writer.add_scalar('episode_reward', episode_return, i_episode + i*int(args.num_episodes/10))
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (args.num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    print("训练完成！")
    path = os.path.join('model', '_'.join(["PPO", args.env_id]))
    if not os.path.exists(path):
        os.makedirs(path)
    date_str = datetime.now().strftime("%Y%m%d")
    torch.save(agent.actor.state_dict(), f"{path}/actor_{date_str}.pth")
    torch.save(agent.critic.state_dict(), f"{path}/critic_{date_str}.pth")

    return return_list


def train_continuous():
    # 训练处理连续动作的TRPO算法
    writer = SummaryWriter(log_dir='logs/ppo/continuous')
    # 设置随机种子，确保实验可复现
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env = gym.make(args.env_id, render_mode=args.render)
    actor_lr = 1e-4
    critic_lr = 5e-3
    hidden_dim = 128
    gamma = 0.9
    lmbda = 0.9
    epochs = 10
    eps = 0.2
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]  # 连续动作空间

    agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                      lmbda, epochs, eps, gamma, device)
    return_list = []

    for i in range(10):
        with tqdm(total=int(args.num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(args.num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, _ = env.reset(seed=args.seed)
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                    if done or truncated:
                        break
                return_list.append(episode_return)
                writer.add_scalar('episode_reward', episode_return, i_episode + i*int(args.num_episodes/10))
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (args.num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    print("训练完成！")
    path = os.path.join('model', '_'.join(["PPO", args.env_id]))
    if not os.path.exists(path):
        os.makedirs(path)
    date_str = datetime.now().strftime("%Y%m%d")
    torch.save(agent.actor.state_dict(), f"{path}/actor_{date_str}.pth")
    torch.save(agent.critic.state_dict(), f"{path}/critic_{date_str}.pth")

    return return_list


if __name__ == "__main__":
    if args.train:
        if args.env_id == 'Pendulum-v1':
            returns = train_continuous()
        else:
            returns = train_discrete()
    if args.test:
        env = gym.make(args.env_id, render_mode=args.render)
        date_str = datetime.now().strftime("%Y%m%d")
        if args.env_id == 'Pendulum-v1':
            actor_lr = 1e-4
            critic_lr = 5e-3
            hidden_dim = 128
            gamma = 0.9
            lmbda = 0.9
            epochs = 10
            eps = 0.2
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]  # 连续动作空间
            device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")

            agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                              lmbda, epochs, eps, gamma, device)
            path = os.path.join('model', '_'.join(["PPO", args.env_id]))
            agent.actor.load_state_dict(torch.load(f"{path}/actor_{date_str}.pth"))
            agent.critic.load_state_dict(torch.load(f"{path}/critic_{date_str}.pth"))
        else:
            actor_lr = 1e-3
            critic_lr = 1e-2
            hidden_dim = 128
            gamma = 0.98
            lmbda = 0.95
            epochs = 10
            eps = 0.2
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
            agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                        epochs, eps, gamma, device)
            
            path = os.path.join('model', '_'.join(["PPO", args.env_id]))
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