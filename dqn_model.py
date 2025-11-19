"""
DQN模型实现 - 基于文档RL环境核心要素总结.md
包含：Q网络、经验回放、Double DQN、训练逻辑
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import os

# 超参数（根据文档建议）
LEARNING_RATE = 5e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.997
BATCH_SIZE = 128
BUFFER_SIZE = 200000
UPDATE_EVERY = 4  # 每隔多少步更新一次网络
TAU = 1e-3  # 软更新系数（目标网络）

# 设备选择
def get_device():
    """获取计算设备（优先使用GPU）"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ 检测到CUDA，使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        print("⚠ 未检测到CUDA，使用CPU")
    return device

device = get_device()

# 经验元组
Experience = namedtuple('Experience', 
                        field_names=['state', 'action', 'reward', 'next_state', 'done'])


class QNetwork(nn.Module):
    """
    Q网络：输入状态，输出每个动作的Q值
    输入：扩展后的状态向量（27维，使用周期性编码处理环形边界）
    输出：3个动作的Q值 [Q(left), Q(noop), Q(right)]
    网络容量扩展：128-128，添加Batch Normalization
    """
    def __init__(self, state_size=27, action_size=3, seed=42, hidden_sizes=[128, 128]):
        """
        Args:
            state_size: 状态空间维度（默认27，使用周期性编码后）
            action_size: 动作空间大小（默认3）
            seed: 随机种子
            hidden_sizes: 隐藏层大小列表（默认[128, 128]）
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # 构建全连接层（添加Batch Normalization）
        layers = []
        input_size = state_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # 全连接层
            layers.append(nn.Linear(input_size, hidden_size))
            # Batch Normalization（提高训练稳定性）
            layers.append(nn.BatchNorm1d(hidden_size))
            # 激活函数
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        # 输出层（不使用BatchNorm和激活函数）
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        """前向传播"""
        # 确保state有batch维度（BatchNorm需要）
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # 如果batch_size=1且处于训练模式，BatchNorm可能不稳定
        # 但在eval模式下可以正常工作（act方法中会设置eval模式）
        return self.network(state)


class ReplayBuffer:
    """
    经验回放缓冲区
    用于存储和采样经验，打破数据相关性
    """
    def __init__(self, action_size, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=42):
        """
        Args:
            action_size: 动作空间大小
            buffer_size: 缓冲区最大容量
            batch_size: 批次大小
            seed: 随机种子
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = Experience
        random.seed(seed)
        
    def add(self, state, action, reward, next_state, done):
        """添加经验到缓冲区"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        """从缓冲区随机采样一批经验"""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """返回当前缓冲区大小"""
        return len(self.memory)


class DQNAgent:
    """
    DQN智能体
    实现Double DQN算法，包含经验回放和目标网络
    """
    def __init__(self, state_size=27, action_size=3, seed=42):
        """
        Args:
            state_size: 状态空间维度（默认27，使用周期性编码后）
            action_size: 动作空间大小
            seed: 随机种子
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # Q网络（主网络）- 使用扩展后的网络容量（128-128 + BatchNorm）
        self.qnetwork_local = QNetwork(state_size, action_size, seed, hidden_sizes=[128, 128]).to(device)
        # Q网络（目标网络）
        self.qnetwork_target = QNetwork(state_size, action_size, seed, hidden_sizes=[128, 128]).to(device)
        
        # 优化器
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LEARNING_RATE)
        
        # 经验回放
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        # 探索参数
        self.epsilon = EPSILON_START
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_END
        
        # 更新计数器
        self.t_step = 0
        
        # 损失记录
        self.losses = []
        
        # 当前训练回合数（用于分阶段梯度裁剪）
        self.current_episode = 0
        
    def step(self, state, action, reward, next_state, done):
        """
        保存经验并学习
        """
        # 保存经验
        self.memory.add(state, action, reward, next_state, done)
        
        # 每隔UPDATE_EVERY步学习一次
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # 如果缓冲区有足够的经验，进行学习
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                loss = self.learn(experiences, GAMMA)
                self.losses.append(loss)
    
    def act(self, state, training=True):
        """
        根据当前策略选择动作（ε-greedy）
        
        Args:
            state: 当前状态
            training: 是否在训练模式（训练时使用ε-greedy，测试时使用贪婪策略）
        
        Returns:
            action: 选择的动作
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # ε-greedy策略
        if training and random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences, gamma):
        """
        使用经验批次更新Q网络参数（Double DQN）
        
        Args:
            experiences: 经验元组 (states, actions, rewards, next_states, dones)
            gamma: 折扣因子
        
        Returns:
            loss.item(): 损失值（标量）
        """
        states, actions, rewards, next_states, dones = experiences
        
        # 获取当前Q值
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Double DQN: 使用主网络选择动作，目标网络评估Q值
        with torch.no_grad():
            # 主网络选择下一个状态的最佳动作
            next_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
            # 目标网络评估这些动作的Q值
            Q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions)
            # 计算目标Q值
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # 计算损失
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        
        # 根据训练阶段调整梯度裁剪值
        if self.current_episode < 300:
            max_norm = 1.0  # 早期：允许较大梯度，快速学习
        elif self.current_episode < 700:
            max_norm = 0.7  # 中期：适度约束
        else:
            max_norm = 0.5  # 后期：强约束，稳定训练
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), max_norm=max_norm)
        self.optimizer.step()
        
        # 软更新目标网络
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        
        # 返回损失值
        return loss.item()
    
    def soft_update(self, local_model, target_model, tau):
        """
        软更新目标网络参数
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def update_epsilon(self):
        """衰减探索率"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'qnetwork_local': self.qnetwork_local.state_dict(),
            'qnetwork_target': self.qnetwork_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load(self, filepath):
        """加载模型"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=device)
            self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local'])
            self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            print(f"模型已从 {filepath} 加载")
            return True
        else:
            print(f"模型文件不存在: {filepath}")
            return False

