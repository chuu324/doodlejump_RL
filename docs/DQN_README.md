# DQN模型使用说明

根据 `RL环境核心要素总结.md` 文档实现的DQN（Deep Q-Network）模型。

## 文件说明

- **dqn_model.py**: DQN模型核心实现
  - `QNetwork`: Q值网络（输入状态，输出动作Q值）
  - `ReplayBuffer`: 经验回放缓冲区
  - `DQNAgent`: DQN智能体（包含Double DQN算法）

- **train_dqn.py**: 训练脚本
  - 训练DQN智能体
  - 保存模型和训练曲线
  - 测试训练好的模型

- **main.py**: 主程序（已更新支持DQN模型）
  - 选项3: 使用训练好的DQN模型录制视频
  - 选项5: 使用训练好的DQN模型玩游戏

## 安装依赖

```bash
pip install -r requirements.txt
```

## 训练模型

### 基本训练

```bash
python train_dqn.py --mode train
```

### 训练参数说明

训练脚本默认参数（可在 `train_dqn.py` 中修改）：
- `N_EPISODES = 2000`: 训练回合数
- `MAX_STEPS = 10000`: 每个回合最大步数
- `SCORE_WINDOW = 100`: 计算平均分数的窗口大小
- `SAVE_INTERVAL = 100`: 每隔多少回合保存一次模型

### DQN超参数（在 `dqn_model.py` 中）

根据文档建议设置：
- `LEARNING_RATE = 5e-4`: 学习率
- `GAMMA = 0.99`: 折扣因子
- `EPSILON_START = 1.0`: 初始探索率
- `EPSILON_END = 0.01`: 最终探索率
- `EPSILON_DECAY = 0.995`: 探索率衰减
- `BATCH_SIZE = 64`: 批次大小
- `BUFFER_SIZE = 100000`: 经验回放缓冲区大小
- `UPDATE_EVERY = 4`: 每隔多少步更新一次网络
- `TAU = 1e-3`: 目标网络软更新系数

## 测试模型

### 测试训练好的模型

```bash
python train_dqn.py --mode test --episodes 5
```

### 测试时不渲染（更快）

```bash
python train_dqn.py --mode test --episodes 10 --no-render
```

### 使用指定模型

```bash
python train_dqn.py --mode test --model models/dqn_best.pth --episodes 5
```

## 使用训练好的模型

### 方式1: 通过main.py

```bash
python main.py
# 选择选项 3: 录制训练后的视频
# 或选择选项 5: 使用训练好的模型玩游戏
```

### 方式2: 在代码中使用

```python
from dqn_model import DQNAgent
from rl_env import DoodleJumpEnv

# 创建环境和智能体
env = DoodleJumpEnv(render_mode='human')
state_size = env.observation_space.shape[0]  # 5
action_size = env.action_space.n  # 3
agent = DQNAgent(state_size=state_size, action_size=action_size)

# 加载模型
agent.load("models/dqn_best.pth")
agent.epsilon = 0.0  # 测试时使用贪婪策略

# 运行
obs, info = env.reset()
while True:
    action = agent.act(obs, training=False)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

## 模型文件

训练过程中会生成以下文件：

- `models/dqn_checkpoint.pth`: 定期保存的检查点
- `models/dqn_best.pth`: 最佳模型（分数最高时保存）
- `training_curve.png`: 训练曲线图

## 模型架构

### Q网络结构

```
输入层: 5维状态向量
  ↓
全连接层1: 64个神经元 + ReLU
  ↓
全连接层2: 64个神经元 + ReLU
  ↓
输出层: 3个Q值（对应3个动作）
```

### 状态空间（5维）

1. 玩家X坐标（归一化到[0,1]）
2. 玩家Y坐标（归一化到[0,1]）
3. 玩家垂直速度（归一化）
4. 最近平台水平相对位置
5. 最近平台垂直相对位置

### 动作空间（3个离散动作）

- 动作0: 向左移动
- 动作1: 静止/不移动
- 动作2: 向右移动

## 训练监控

训练过程中会显示：
- 当前回合数
- 最近100回合平均分数
- 当前回合分数
- 最佳分数
- 当前探索率（ε）
- 回合步数
- 每回合耗时

训练结束后会生成训练曲线图，包含：
- 所有回合的分数曲线
- 最近100回合的平均分数趋势

## 性能优化建议

1. **GPU加速**: 如果有NVIDIA GPU，PyTorch会自动使用CUDA加速
2. **训练速度**: 训练时使用 `render_mode=None`（默认），测试时再使用渲染
3. **批次大小**: 如果内存充足，可以增大 `BATCH_SIZE` 到 128
4. **网络大小**: 可以尝试更大的网络（如128或256个神经元）

## 常见问题

### Q: 训练很慢怎么办？
A: 
- 确保使用 `render_mode=None` 进行训练
- 如果有GPU，确保PyTorch正确识别CUDA
- 可以减少 `N_EPISODES` 先测试

### Q: 模型性能不好？
A:
- 增加训练回合数
- 调整超参数（学习率、网络大小等）
- 检查奖励函数是否合理
- 确保状态归一化正确

### Q: 如何继续训练？
A: 目前需要修改代码加载已有模型并继续训练，或者使用保存的检查点重新开始训练。

## 参考文档

详细的环境说明请参考：`RL环境核心要素总结.md`

