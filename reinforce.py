import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
import random
import os
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--seed', help='random seed', type=int, default=0)
parser.add_argument('--env_id', default='CartPole-v1', help='CartPole-v1 or Pendulum-v1')
parser.add_argument('--gamma', type=float, default=0.98)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_episodes', type=int, default=1000)
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--render', type=str, default='rgb_array', help='rgb_array or human')
args = parser.parse_args()


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    


class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 device):
        self.policy_net = PolicyNet(state_dim, hidden_dim,
                                    action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state):  # 根据动作概率分布随机采样动作
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.from_numpy(state_list[i]).float().unsqueeze(0).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降
    

def train():
    writer = SummaryWriter(log_dir='logs/reinforce/' + args.env_id)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env_name = args.env_id
    env = gym.make(env_name, render_mode=args.render)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    learning_rate = args.lr
    num_episodes = args.num_episodes
    hidden_dim = 128
    gamma = args.gamma
    agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma,
                    device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset(seed=args.seed)
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
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
                agent.update(transition_dict)
                writer.add_scalar('episode_reward', episode_return, i_episode + i*int(args.num_episodes/10))
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    print("训练完成！")
    path = os.path.join('model', '_'.join(["reinforce", args.env_id]))
    if not os.path.exists(path):
        os.makedirs(path)
    date_str = datetime.now().strftime("%Y%m%d")
    torch.save(agent.policy_net.state_dict(), f"{path}/REINFORCE_{args.env_id}_{date_str}.pth")
    return return_list


def test():
    env_name = args.env_id
    env = gym.make(env_name, render_mode=args.render)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    hidden_dim = 128
    agent = REINFORCE(state_dim, hidden_dim, action_dim, args.lr, args.gamma,
                    device)
    path = os.path.join('model', '_'.join(["reinforce", args.env_id]))
    date_str = datetime.now().strftime("%Y%m%d")
    model_path = f"{path}/REINFORCE_{args.env_id}_{date_str}.pth"
    agent.policy_net.load_state_dict(torch.load(model_path))
    for i_episode in range(100):
        episode_return = 0
        state, _ = env.reset(seed=args.seed)
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            episode_return += reward
            if done or truncated:
                break
        print('Episode: {}/{}  Return: {:.3f}'.format(i_episode + 1, 100, episode_return))


if __name__ == '__main__':
    if args.train:
        train()
    if args.test:
        test()