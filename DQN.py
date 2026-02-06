# import gym
import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import argparse
from torch.utils.tensorboard import SummaryWriter
import os


parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--seed', help='random seed', type=int, default=0)
parser.add_argument('--env_id', default='CartPole-v1', help='CartPole-v1 or Pendulum-v1')
parser.add_argument('--enable_double', action='store_true', default=False)
parser.add_argument('--enable_dueling', action='store_true', default=False)
parser.add_argument('--buffer_size', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--target_update_freq', type=int, default=10)
parser.add_argument('--initial_epsilon', type=float, default=1.0)
parser.add_argument('--epsilon_decay', type=int, default=500)
parser.add_argument('--num_episodes', type=int, default=1000)
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--render', type=str, default='human', help='rgb_array or human')
args = parser.parse_args()

# 设置随机种子，确保实验可复现
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# 创建环境
# env = gym.make('CartPole-v1', render_mode="human", sutton_barto_reward=True)
env = gym.make(args.env_id, render_mode=args.render)
if args.env_id == 'CartPole-v1':
    state_dim = env.observation_space.shape[0]  # 状态维度：4 (小车位置，速度，杆角度，角速度)
    action_dim = env.action_space.n            # 动作维度：2 (向左，向右)
    print(f"状态空间维度: {state_dim}, 动作空间大小: {action_dim}")
else:  # Pendulum-v1
    state_dim = env.observation_space.shape[0]  # 状态维度：3 (cos(θ), sin(θ), 角速度)
    action_dim = 11                            # 动作维度：离散化为11个动作
    print(f"状态空间维度: {state_dim}, 动作空间大小: {action_dim}")
device = torch.device(args.device)
writer = SummaryWriter(log_dir='logs/DQN')


def dis_to_con(discrete_action, env, action_dim):  # 离散动作转回连续的函数
    action_lowbound = env.action_space.low[0]  # 连续动作的最小值
    action_upbound = env.action_space.high[0]  # 连续动作的最大值
    return action_lowbound + (discrete_action /
                              (action_dim - 1)) * (action_upbound -
                                                   action_lowbound)


class VAnet(nn.Module):
    ''' 只有一层隐藏层的A网络和V网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 共享网络部分
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(nn.functional.relu(self.fc1(x)))
        V = self.fc_V(nn.functional.relu(self.fc1(x)))
        Q = V + A - A.mean(1).view(-1, 1)  # Q值由V值和A值计算得到
        return Q


# 定义Q网络  
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)  # 第一层全连接层
        self.fc2 = nn.Linear(128, 128)        # 第二层全连接层
        self.fc3 = nn.Linear(128, action_dim) # 输出层，每个动作一个Q值

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))  # 使用ReLU激活函数引入非线性
        # x = nn.functional.relu(self.fc2(x))
        return self.fc3(x)           # 输出Q值，不经过激活函数


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
        return (np.array(state), action, reward, np.array(next_state), done)

    def size(self):
        return len(self.buffer)
    

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.98, epsilon=0.01,
                 target_update_freq=10, buffer_size=10000, batch_size=64, device=device):
        self.action_dim = action_dim
        if args.enable_dueling:
            self.q_net = VAnet(state_dim, 128, action_dim).to(device)          # 在线网络
            self.target_q_net = VAnet(state_dim, 128, action_dim).to(device)   # 目标网络
        else:
            self.q_net = DQN(state_dim, action_dim).to(device)          # 在线网络
            self.target_q_net = DQN(state_dim, action_dim).to(device)   # 目标网络
        self.target_q_net.load_state_dict(self.q_net.state_dict()) # 初始参数一致
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr) # 优化器

        self.gamma = gamma               # 折扣因子
        self.epsilon = epsilon           # 探索率（最终）
        self.target_update_freq = target_update_freq # 目标网络更新频率
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)
        self.count = 0                   # 记录更新步数
        self.device = device

    def max_q_value(self, state):
        state = torch.tensor(state, dtype=torch.float).view(1, -1).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state).max().item()
        return q_values

    def take_action(self, state, epsilon=None):
        """根据epsilon-greedy策略选择动作"""
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)  # 探索：随机选择
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device) # 增加批次维度
            with torch.no_grad():
                q_values = self.q_net(state)
            return q_values.argmax().item()            # 利用：选择Q值最大的动作

    def update(self):
        """从经验回放池采样并更新网络"""
        if self.buffer.size() < self.batch_size:
            return

        # 1. 采样
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        # 转换为PyTorch张量
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)      # 形状变为[batch_size, 1]，便于gather操作
        rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).unsqueeze(1).to(self.device)
        # print("Sampled states shape:", states.shape)
        # print("Sampled actions shape:", actions.shape)
        # print("Sampled rewards shape:", rewards.shape)
        # print("Sampled next_states shape:", next_states.shape)
        # print("Sampled dones shape:", dones.shape)
        # print("Sampled actions:", actions.squeeze().numpy())
        # print("Sampled rewards:", rewards.squeeze().numpy())
        # print("Sampled dones:", dones.squeeze().numpy())
        # print("Sampled next_states:", next_states.numpy())
        # print("Sampled states:", states.numpy())

        # 2. 计算当前Q值 (Q(s, a))
        current_q_values = self.q_net(states).gather(1, actions)  # 只取出执行动作a对应的Q值
        # print("Current Q-values shape:", current_q_values.shape)
        # print("Current Q-values:", current_q_values.detach().numpy())

        # 3. 计算目标Q值 (r + γ * max_a' Q_target(s', a'))
        with torch.no_grad():
            if args.enable_double:
                # Double DQN
                next_actions = self.q_net(next_states).argmax(1).unsqueeze(1)  # 在线网络选择动作
                next_q_values = self.target_q_net(next_states).gather(1, next_actions) # 目标网络评估动作价值
            else:
                 # 普通DQN
                next_q_values = self.target_q_net(next_states).max(1)[0].unsqueeze(1) # 取下一状态的最大Q值
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones) # 如果回合结束(done=1)，则没有未来奖励
        # print("Target Q-values shape:", target_q_values.shape)
        # print("Target Q-values:", target_q_values.numpy())

        # 4. 计算损失 (均方误差)
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # 5. 梯度下降更新在线网络
        self.optimizer.zero_grad()
        loss.backward()
        # 可选：梯度裁剪，防止梯度爆炸
        # torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        self.optimizer.step()

        self.count += 1
        # 6. 定期更新目标网络
        if self.count % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())


def train_agent(env, agent:DQNAgent, num_episodes=500, minimal_size=500, initial_epsilon=1.0, epsilon_decay=500):
    """训练智能体"""
    return_list = []  # 记录每个回合的总奖励
    epsilon = initial_epsilon  # 初始化探索率
    frame_idx = 0
    max_q_value = 0.0

    for i_episode in range(num_episodes):
        state, _ = env.reset(seed=args.seed)
        episode_return = 0
        done = False

        while not done:
            # 1. 选择并执行动作
            action = agent.take_action(state, epsilon)  # 使用衰减的探索率
            max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995  # 平滑处理
            if args.env_id == 'CartPole-v1':
                next_state, reward, done, truncated, _ = env.step(action)
            else:  # Pendulum-v1
                action_continuous = dis_to_con(action, env, agent.action_dim)  # 离散动作转回连续动作
                next_state, reward, done, truncated, _ = env.step([action_continuous])
            # 2. 存储经验
            agent.buffer.add(state, action, reward, next_state, done)
            writer.add_scalar('max_q_value', max_q_value, frame_idx)
            frame_idx += 1

            state = next_state
            episode_return += reward
            
            # 3. 更新网络
            if agent.buffer.size() >= minimal_size:
                agent.update()

            if done or truncated:
                break

        # 探索率衰减
        # epsilon = max(agent.epsilon, epsilon * epsilon_decay)
        epsilon = initial_epsilon  - 0.99 * min(initial_epsilon, i_episode / epsilon_decay)
        writer.add_scalar('episode_reward', episode_return, i_episode)
        return_list.append(episode_return)
        if (i_episode + 1) % 50 == 0:
            print(f"回合: {i_episode+1}, 平均奖励 (最近50回合): {np.mean(return_list[-50:]):.1f}, 探索率: {epsilon:.3f}")

    print("训练完成！")
    path = os.path.join('model', '_'.join(["DQN", args.env_id]))
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(agent.q_net.state_dict(), f"{path}/DQN_{args.env_id}.pth")
    return return_list

# 创建智能体并开始训练
if args.train:
    agent = DQNAgent(state_dim, action_dim, lr=args.lr, gamma=args.gamma, epsilon=0.01)
    returns = train_agent(env, agent, num_episodes=args.num_episodes, minimal_size=500, initial_epsilon=args.initial_epsilon, epsilon_decay=args.epsilon_decay)

# 测试智能体
if args.test:
    agent = DQNAgent(state_dim, action_dim, lr=args.lr, gamma=args.gamma, epsilon=0.01)
    path = os.path.join('model', '_'.join(["DQN", args.env_id]))
    agent.q_net.load_state_dict(torch.load(f"{path}/DQN_{args.env_id}.pth", map_location=device))
    agent.q_net.eval()
    for i_episode in range(100):
        state, _ = env.reset()
        episode_return = 0
        done = False
        while not done:
            action = agent.take_action(state, epsilon=0.0)  # 测试时不探索
            if args.env_id == 'CartPole-v1':
                next_state, reward, done, truncated, _ = env.step(action)
            else:
                action_continuous = dis_to_con(action, env, agent.action_dim)  # 离散动作转回连续动作
                next_state, reward, done, truncated, _ = env.step([action_continuous])
            state = next_state
            episode_return += reward
            if done or truncated:
                break
        print(f"测试回合: {i_episode+1}, 总奖励: {episode_return}")