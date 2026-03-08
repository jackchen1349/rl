import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import random
import os
from torch.utils.tensorboard import SummaryWriter
import collections


parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--seed', help='random seed', type=int, default=0)
parser.add_argument('--env_id', default='CartPole-v1', help='CartPole-v1 or Pendulum-v1')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_episodes', type=int, default=1000)
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--render', type=str, default='human', help='rgb_array or human')
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
        return (state, action, reward, next_state, done)

    def size(self):
        return len(self.buffer)


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
    
class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma=0.99, device='cpu'):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def update(self, transition_dict):
        states_array = np.array(transition_dict['states'])
        states = torch.tensor(states_array,
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states_array = np.array(transition_dict['next_states'])
        next_states = torch.tensor(next_states_array,
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # Update critic
        value = self.critic(states)
        next_value = self.critic(next_states)
        target_value = rewards + (1 - dones) * self.gamma * next_value
        critic_loss = F.mse_loss(value, target_value.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        advantage = (target_value - value).detach()
        probs = self.actor(states)
        action_dist = torch.distributions.Categorical(probs)
        log_prob = action_dist.log_prob(actions)
        actor_loss = torch.mean(-log_prob * advantage)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


def train_on_policy_agent():
    writer = SummaryWriter(log_dir='logs/AC/on_policy')
    # 设置随机种子，确保实验可复现
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env = gym.make(args.env_id, render_mode=args.render)
    actor_lr = 1e-3
    critic_lr = 1e-2
    hidden_dim = 128
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, args.gamma, device)
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
    path = os.path.join('model', '_'.join(["AC", args.env_id]))
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(agent.actor.state_dict(), f"{path}/AC_{args.env_id}_actor.pth")
    torch.save(agent.critic.state_dict(), f"{path}/AC_{args.env_id}_critic.pth")

    return return_list


def train_off_policy_agent(minimal_size = 100, batch_size = 64):
    writer = SummaryWriter(log_dir='logs/AC/off_policy')
    # 设置随机种子，确保实验可复现
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env = gym.make(args.env_id, render_mode=args.render)
    actor_lr = 1e-3
    critic_lr = 1e-2
    hidden_dim = 128
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, args.gamma, device)
    return_list = []
    replay_buffer = ReplayBuffer(capacity=10000)

    for i in range(10):
        with tqdm(total=int(args.num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(args.num_episodes/10)):
                episode_return = 0
                state, _ = env.reset(seed=args.seed)
                done = False
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
    path = os.path.join('model', '_'.join(["AC", args.env_id]))
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(agent.actor.state_dict(), f"{path}/AC_{args.env_id}_actor.pth")
    torch.save(agent.critic.state_dict(), f"{path}/AC_{args.env_id}_critic.pth")

    return return_list


if __name__ == "__main__":
    if args.train:
        returns = train_on_policy_agent()

    if args.test:
        env = gym.make(args.env_id, render_mode=args.render)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        hidden_dim = 128
        device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
        agent = ActorCritic(state_dim, hidden_dim, action_dim, 1e-3, 1e-2, args.gamma, device)
        path = os.path.join('model', '_'.join(["AC", args.env_id]))
        agent.actor.load_state_dict(torch.load(f"{path}/AC_{args.env_id}_actor.pth"))
        agent.critic.load_state_dict(torch.load(f"{path}/AC_{args.env_id}_critic.pth"))
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