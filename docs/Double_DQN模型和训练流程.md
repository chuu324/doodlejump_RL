# Double DQN 模型和训练流程说明

本文档介绍 Doodle Jump 游戏使用的 Double DQN 模型架构和训练流程。

## 目录

1. [模型概述](#模型概述)
2. [Double DQN 算法](#double-dqn-算法)
3. [网络架构](#网络架构)
4. [核心组件](#核心组件)
5. [训练流程](#训练流程)
6. [超参数配置](#超参数配置)
7. [模型保存与加载](#模型保存与加载)
8. [训练监控](#训练监控)

---

## 模型概述

本项目使用 **Double DQN** 算法训练智能体玩 Doodle Jump 游戏。

### 为什么选择 Double DQN？

1. **减少Q值高估**：传统DQN容易高估Q值，Double DQN通过分离动作选择和Q值评估，减少高估问题
2. **训练稳定**：目标网络软更新机制，提高训练稳定性
3. **经验回放**：打破数据相关性，提高样本效率
4. **适合离散动作**：Doodle Jump 是离散动作空间（3个动作），DQN非常适合

### 文件结构

```
dqn_model.py      # DQN模型核心实现
train_dqn.py      # 训练脚本
model_test.py     # 模型测试脚本
models/           # 模型保存目录
  - dqn_best.pth           # 最佳模型
  - dqn_checkpoint_*.pth   # 定期检查点
```

---

## Double DQN 算法

### 算法原理

Double DQN 是对标准 DQN 的改进，核心思想是**分离动作选择和Q值评估**。

#### 标准 DQN 的问题

标准 DQN 使用目标网络计算目标Q值：
```
Q_target = r + γ * max_a' Q_target(s', a')
```

问题：**动作选择和Q值评估都使用目标网络**，容易导致Q值高估。

#### Double DQN 的解决方案

Double DQN 使用主网络选择动作，目标网络评估Q值：
```
a* = argmax_a' Q_local(s', a')      # 主网络选择动作
Q_target = r + γ * Q_target(s', a*)  # 目标网络评估Q值
```

**优势**：
- 减少Q值高估
- 提高训练稳定性
- 通常能获得更好的性能

### 算法流程

```
1. 初始化主网络 Q_local 和目标网络 Q_target
2. 初始化经验回放缓冲区
3. For each episode:
   a. 重置环境，获取初始状态 s
   b. For each step:
      - 使用 ε-greedy 策略选择动作 a
      - 执行动作，获得 (s, a, r, s', done)
      - 将经验存入缓冲区
      - 每隔 UPDATE_EVERY 步：
        * 从缓冲区采样一批经验
        * 使用 Double DQN 更新主网络
        * 软更新目标网络
      - 更新探索率 ε
```

---

## 网络架构

### Q网络结构

```python
QNetwork(
    input_size=27,        # 状态空间维度
    hidden_sizes=[128, 128],  # 隐藏层大小
    output_size=3        # 动作空间大小
)
```

#### 网络层次

```
输入层: 27维状态向量
  ↓
全连接层1: 128个神经元
  ↓
Batch Normalization
  ↓
ReLU 激活
  ↓
全连接层2: 128个神经元
  ↓
Batch Normalization
  ↓
ReLU 激活
  ↓
输出层: 3个Q值 [Q(left), Q(noop), Q(right)]
```

### 设计特点

#### 1. 网络容量

- **隐藏层大小**：128-128（相比64-64扩展，提高表达能力）
- **层数**：2层隐藏层（平衡表达能力和训练效率）

#### 2. Batch Normalization

- **作用**：提高训练稳定性，加速收敛
- **位置**：每个隐藏层后
- **注意**：输出层不使用 BatchNorm

#### 3. 激活函数

- **隐藏层**：ReLU（`f(x) = max(0, x)`）
- **输出层**：无激活函数（直接输出Q值）

### 状态空间（27维）

输入状态包含：

1. **玩家基础状态**（5维）：
   - `x_sin, x_cos`: X坐标周期性编码
   - `y`: Y坐标（归一化）
   - `vel_y`: 垂直速度（归一化）
   - `vel_x`: 水平速度（归一化）⭐

2. **最近障碍物信息**（4维）：
   - `dx, dy, dist, type`

3. **最近平台信息**（3维）：
   - `dx, dy, type`

4. **上方5个平台信息**（15维）：
   - 每个平台：`dx, dy, type`

### 动作空间（3个离散动作）

- **动作0**：向左移动
- **动作1**：静止/不移动
- **动作2**：向右移动

---

## 核心组件

### 1. QNetwork（Q网络）

```python
class QNetwork(nn.Module):
    def __init__(self, state_size=27, action_size=3, 
                 seed=42, hidden_sizes=[128, 128]):
        # 构建全连接层（带BatchNorm）
        # 输入: state_size
        # 输出: action_size
```

**功能**：
- 输入状态，输出每个动作的Q值
- 使用 Batch Normalization 提高稳定性

### 2. ReplayBuffer（经验回放缓冲区）

```python
class ReplayBuffer:
    def __init__(self, buffer_size=200000, batch_size=128):
        self.memory = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        # 添加经验到缓冲区
    
    def sample(self):
        # 随机采样一批经验
```

**功能**：
- 存储经验 `(s, a, r, s', done)`
- 随机采样打破数据相关性
- 提高样本效率

**参数**：
- `buffer_size=200000`：缓冲区最大容量
- `batch_size=128`：批次大小

### 3. DQNAgent（DQN智能体）

```python
class DQNAgent:
    def __init__(self, state_size=27, action_size=3):
        # 主网络和目标网络
        self.qnetwork_local = QNetwork(...)
        self.qnetwork_target = QNetwork(...)
        
        # 经验回放
        self.memory = ReplayBuffer(...)
        
        # 探索参数
        self.epsilon = 1.0  # 初始探索率
```

**核心方法**：

#### `act(state, training=True)`

```python
def act(self, state, training=True):
    """使用ε-greedy策略选择动作"""
    if training and random.random() > self.epsilon:
        # 贪婪动作（选择Q值最大的动作）
        return np.argmax(Q_values)
    else:
        # 随机探索
        return random.choice([0, 1, 2])
```

#### `step(state, action, reward, next_state, done)`

```python
def step(self, state, action, reward, next_state, done):
    """保存经验并学习"""
    # 1. 保存经验到缓冲区
    self.memory.add(state, action, reward, next_state, done)
    
    # 2. 每隔UPDATE_EVERY步学习一次
    if len(self.memory) > BATCH_SIZE:
        experiences = self.memory.sample()
        loss = self.learn(experiences, GAMMA)
```

#### `learn(experiences, gamma)`

```python
def learn(self, experiences, gamma):
    """Double DQN更新"""
    # 1. 计算当前Q值
    Q_expected = Q_local(s).gather(1, a)
    
    # 2. Double DQN: 主网络选择动作，目标网络评估Q值
    next_actions = Q_local(s').max(1)[1]  # 主网络选择
    Q_targets_next = Q_target(s').gather(1, next_actions)  # 目标网络评估
    
    # 3. 计算目标Q值
    Q_targets = r + γ * Q_targets_next * (1 - done)
    
    # 4. 计算损失并更新
    loss = MSE(Q_expected, Q_targets)
    loss.backward()
    
    # 5. 梯度裁剪（分阶段）
    clip_grad_norm_(max_norm)
    
    # 6. 软更新目标网络
    soft_update(Q_local, Q_target, TAU)
```

### 4. 分阶段梯度裁剪

根据训练阶段动态调整梯度裁剪阈值：

```python
if current_episode < 300:
    max_norm = 1.0   # 早期：允许较大梯度，快速学习
elif current_episode < 700:
    max_norm = 0.7   # 中期：适度约束
else:
    max_norm = 0.5   # 后期：强约束，稳定训练
```

**目的**：
- 早期允许较大梯度，快速学习
- 后期强约束，稳定训练，避免震荡

---

## 训练流程

### 训练脚本

使用 `train_dqn.py` 进行训练：

```bash
python train_dqn.py --mode train
```

### 训练参数

```python
N_EPISODES = 900          # 训练回合数
MAX_STEPS = 20000         # 每个回合最大步数
SCORE_WINDOW = 100        # 计算平均分数的窗口大小
SAVE_INTERVAL = 100       # 每隔多少回合保存一次模型
BEST_CHECK_INTERVAL = 50  # 每隔多少回合检查一次最佳平均分
PRINT_INTERVAL = 10       # 每隔多少回合打印一次信息
```

### 训练循环

```python
for episode in range(1, N_EPISODES + 1):
    # 1. 重置环境
    state, info = env.reset()
    score = 0
    
    # 2. 更新智能体的当前回合数（用于分阶段梯度裁剪）
    agent.current_episode = episode
    
    # 3. 执行一个回合
    for step in range(MAX_STEPS):
        # 选择动作
        action = agent.act(state, training=True)
        
        # 执行动作
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # 保存经验并学习
        agent.step(state, action, reward, next_state, terminated)
        
        # 更新状态
        state = next_state
        score = info['score']
        
        # 游戏结束
        if terminated or truncated:
            break
    
    # 4. 更新探索率
    agent.update_epsilon()
    
    # 5. 记录统计信息
    scores.append(score)
    scores_window.append(score)
    
    # 6. 定期保存模型和打印信息
    if episode % SAVE_INTERVAL == 0:
        agent.save(checkpoint_path)
    
    if episode % BEST_CHECK_INTERVAL == 0:
        if mean_score > best_mean_score:
            agent.save(BEST_MODEL_PATH)
```

### 模型保存策略

1. **定期检查点**：每100回合保存一次 `dqn_checkpoint_{episode}.pth`
2. **最佳模型**：每50回合检查一次，如果平均分数超过历史最佳，保存 `dqn_best.pth`
3. **最终模型**：训练结束后保存 `dqn_checkpoint.pth`

### 训练环境

- **无头模式**：使用 `render_mode=None`，提升训练速度
- **GPU加速**：自动检测并使用GPU（如果可用）
- **随机种子**：支持设置随机种子，保证可复现性

---

## 超参数配置

### DQN超参数

```python
# 学习相关
LEARNING_RATE = 5e-4      # 学习率
GAMMA = 0.99              # 折扣因子

# 探索相关
EPSILON_START = 1.0       # 初始探索率
EPSILON_END = 0.01        # 最终探索率
EPSILON_DECAY = 0.997     # 探索率衰减（每回合）

# 经验回放
BATCH_SIZE = 128          # 批次大小
BUFFER_SIZE = 200000      # 缓冲区大小
UPDATE_EVERY = 4          # 每隔多少步更新一次网络

# 目标网络
TAU = 1e-3                # 软更新系数
```

### 超参数说明

#### 学习率（LEARNING_RATE）

- **值**：`5e-4`（0.0005）
- **说明**：较小的学习率，保证训练稳定
- **调整建议**：如果训练不稳定，可以降低到 `1e-4`

#### 折扣因子（GAMMA）

- **值**：`0.99`
- **说明**：长期奖励的重要性，接近1表示重视长期奖励
- **调整建议**：通常不需要调整

#### 探索率衰减（EPSILON_DECAY）

- **值**：`0.997`
- **说明**：每回合衰减，`ε = ε * 0.997`
- **效果**：900回合后，`ε ≈ 0.01`
- **调整建议**：如果探索不足，可以降低到 `0.995`

#### 批次大小（BATCH_SIZE）

- **值**：`128`
- **说明**：每次更新使用的经验数量
- **调整建议**：GPU内存充足时可以增大到 `256`

#### 缓冲区大小（BUFFER_SIZE）

- **值**：`200000`
- **说明**：经验回放缓冲区最大容量
- **调整建议**：通常不需要调整

#### 更新频率（UPDATE_EVERY）

- **值**：`4`
- **说明**：每隔4步更新一次网络
- **目的**：平衡学习效率和计算成本
- **调整建议**：可以尝试 `2` 或 `8`

#### 软更新系数（TAU）

- **值**：`1e-3`（0.001）
- **说明**：目标网络更新速度，`θ_target = τ*θ_local + (1-τ)*θ_target`
- **效果**：目标网络缓慢更新，提高稳定性
- **调整建议**：通常不需要调整

---

## 模型保存与加载

### 保存模型

```python
agent.save("models/dqn_best.pth")
```

**保存内容**：
- `qnetwork_local.state_dict()`：主网络参数
- `qnetwork_target.state_dict()`：目标网络参数
- `optimizer.state_dict()`：优化器状态
- `epsilon`：当前探索率

### 加载模型

```python
agent = DQNAgent(state_size=27, action_size=3)
agent.load("models/dqn_best.pth")
agent.epsilon = 0.0  # 测试时使用贪婪策略
```

### 模型文件

训练过程中会生成以下文件：

- `models/dqn_best.pth`：最佳模型（平均分数最高时保存）
- `models/dqn_checkpoint_{episode}.pth`：定期检查点
- `models/dqn_checkpoint.pth`：最终模型

---

## 训练监控

### 训练信息输出

每10回合打印一次训练信息：

```
回合  100 | 平均分数:   245.3 | 当前分数:   280.5 | 最佳分数:   350.2 | 
最佳平均:   245.3 | ε: 0.740 | 步数:  456 | 时间: 2.34s | Avg Loss: 0.0234
```

**信息说明**：
- **平均分数**：最近100回合的平均分数
- **当前分数**：当前回合的分数
- **最佳分数**：历史最高单回合分数
- **最佳平均**：历史最佳平均分数（用于保存模型）
- **ε**：当前探索率
- **步数**：当前回合的步数
- **时间**：平均每回合耗时
- **Avg Loss**：最近1000次更新的平均损失

### 训练曲线

训练结束后会自动生成训练曲线图 `training_curve.png`，包含：

1. **所有回合的分数曲线**：
   - 蓝色：每回合分数（透明度0.3）
   - 红色：最近100回合的平均分数

2. **最近100回合的分数趋势**：
   - 蓝色：最近100回合分数
   - 红色：10回合移动平均

### 性能指标

训练过程中关注以下指标：

1. **平均分数**：最近100回合的平均分数（主要指标）
2. **最佳分数**：历史最高单回合分数
3. **损失值**：Q网络训练损失（应该逐渐下降）
4. **探索率**：应该逐渐衰减到接近0.01

### 训练建议

1. **观察平均分数趋势**：
   - 应该逐渐上升
   - 如果长期不上升，可能需要调整超参数

2. **观察损失值**：
   - 应该逐渐下降并趋于稳定
   - 如果损失值震荡剧烈，可能需要降低学习率

3. **观察探索率**：
   - 应该逐渐衰减
   - 如果探索率衰减太快，可能导致探索不足

4. **定期测试模型**：
   - 使用 `model_test.py` 测试模型性能
   - 观察实际游戏表现

---

## 测试模型

### 使用测试脚本

```bash
python model_test.py
```

**功能**：
- 交互式选择模型文件
- 自定义测试次数
- 可视化游戏窗口
- 显示详细测试结果

### 使用训练脚本测试

```bash
# 测试最佳模型（默认）
python train_dqn.py --mode test --episodes 5

# 测试指定模型
python train_dqn.py --mode test --model models/dqn_best.pth --episodes 10

# 无渲染测试（更快）
python train_dqn.py --mode test --episodes 10 --no-render
```

---

## 总结

Double DQN 模型具备以下特点：

1. **算法优势**：Double DQN 减少Q值高估，提高训练稳定性
2. **网络架构**：128-128隐藏层，带Batch Normalization，表达能力强
3. **训练优化**：分阶段梯度裁剪、经验回放、目标网络软更新
4. **监控完善**：详细的训练信息输出和训练曲线图
5. **模型管理**：自动保存最佳模型和定期检查点

通过精心设计的超参数和训练流程，模型能够有效学习 Doodle Jump 游戏的策略，实现高分表现。

