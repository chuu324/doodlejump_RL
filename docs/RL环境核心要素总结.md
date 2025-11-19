# Doodle Jump 游戏环境 - RL核心要素总结

## 1. 状态空间 (State Space)

### 当前实现（简化版）
**维度：5维连续向量**
```python
obs = [
    player.pos.x / WIDTH,           # 玩家X坐标（归一化到[0,1]）
    player.pos.y / HEIGHT,          # 玩家Y坐标（归一化到[0,1]）
    player.vel.y / 20.0,            # 玩家垂直速度（归一化）
    closest_platform_dx / WIDTH,    # 最近平台水平相对位置
    closest_platform_dy / HEIGHT    # 最近平台垂直相对位置
]
```

### 可扩展状态信息
- **玩家状态**：
  - 位置：`pos.x`, `pos.y` (像素坐标)
  - 速度：`vel.x`, `vel.y` (像素/帧)
  - 加速度：`acc.x`, `acc.y`
  - 当前高度：`max_height` (历史最高Y坐标，越小越高)
  
- **平台状态**（建议扩展）：
  - 最近3-5个平台的相对位置 (dx, dy)
  - 平台类型编码 (normal=0, boost=1, obstacle=2)
  - 平台在屏幕内的可见性
  
- **环境状态**：
  - 当前分数：`score`
  - 难度等级：基于`score`计算的稀疏因子和障碍物概率

### 状态空间定义
```python
observation_space = spaces.Box(
    low=np.array([-np.inf] * 5, dtype=np.float32),
    high=np.array([np.inf] * 5, dtype=np.float32),
    dtype=np.float32
)
```

---

## 2. 动作空间 (Action Space)

### 动作定义
**类型：离散动作空间**
```python
action_space = spaces.Discrete(3)

动作0: 向左移动 (LEFT)
动作1: 静止/不移动 (NOOP)
动作2: 向右移动 (RIGHT)
```

### 动作执行机制
- **水平移动**：
  - 动作0：`acc.x = -PLAYER_ACC` (向左加速度)
  - 动作1：`acc.x = 0` (受摩擦力影响，逐渐减速)
  - 动作2：`acc.x = PLAYER_ACC` (向右加速度)
  
- **垂直移动**：
  - 自动跳跃：踩到平台时自动触发
  - 跳跃速度：`vel.y = -PLAYER_JUMP` (向上)
  - 奖励平台：`vel.y = -PLAYER_JUMP * 1.5` (强力跳跃)

### 动作约束
- 水平方向：屏幕环绕（`pos.x`超出边界时自动环绕）
- 垂直方向：受重力影响，只能向上跳跃，不能主动向下
- 跳跃条件：必须踩在平台上才能跳跃

### 物理参数
```python
PLAYER_ACC = 0.5        # 水平加速度
PLAYER_FRICTION = -0.12 # 摩擦力系数
PLAYER_GRAVITY = 0.8    # 重力加速度
PLAYER_JUMP = 20        # 跳跃初速度
```

---

## 3. 奖励机制 (Reward Function)

### 当前实现
```python
# 基础奖励
if terminated:
    reward = -100  # 死亡惩罚
else:
    reward = 0.1   # 存活奖励
    score_diff = score - last_score
    reward += score_diff  # 高度增加奖励（1:1映射）
```

### 奖励组成
1. **存活奖励**：`+0.1` (每步)
2. **高度奖励**：`+score_diff` (分数增加量，1像素=1分)
3. **死亡惩罚**：`-100` (游戏结束)

### 奖励特性
- **奖励尺度**：`[-100, +∞]`
- **奖励密度**：每步都有奖励（存活奖励）
- **奖励塑形**：高度奖励提供渐进式反馈
- **长期目标**：最大化累计高度（分数）

### 建议优化方向
- 接近平台奖励：`+0.05 * (1 - normalized_distance)`
- 成功跳跃奖励：`+1.0` (踩到平台时)
- 奖励平台奖励：`+2.0` (踩到boost平台)
- 危险惩罚：接近障碍物 `-0.1`，接近屏幕底部 `-0.5`
- 时间惩罚：`-0.01` (可选，鼓励快速决策)

---

## 4. 终止条件 (Terminal States)

### 游戏结束触发条件

1. **掉落死亡**：
   ```python
   if player.rect.bottom > HEIGHT:
       terminated = True
   ```
   - 玩家Y坐标超出屏幕底部

2. **碰撞障碍物**：
   ```python
   if platform.type == 'obstacle' and collision:
       terminated = True
   ```
   - 玩家踩到障碍物平台（红色平台）

### 终止状态返回值
```python
terminated = not game.running  # 布尔值
truncated = False              # 无时间限制
```

### 无终止条件
- 无最大步数限制
- 无时间限制
- 理论上可以无限向上

---

## 5. 环境动态 (Environment Dynamics)

### 物理规则

#### 运动学方程
```python
# 加速度更新
acc.x = PLAYER_ACC * action_direction + vel.x * PLAYER_FRICTION
acc.y = PLAYER_GRAVITY

# 速度更新
vel += acc

# 位置更新
pos += vel + 0.5 * acc
```

#### 碰撞检测
- **平台碰撞**：只在下落时检测 (`vel.y > 0`)
- **碰撞响应**：
  - 普通平台：`vel.y = -PLAYER_JUMP`
  - 奖励平台：`vel.y = -PLAYER_JUMP * 1.5`
  - 障碍物：`terminated = True`

#### 屏幕滚动
- **触发条件**：玩家到达屏幕1/4高度 (`player.rect.top <= HEIGHT/4`)
- **滚动距离**：`scroll_dist = abs(vel.y)`
- **分数增加**：`score += scroll_dist`

### 平台生成规则

#### 生成时机
- **触发条件**：平台数量 < 目标数量
- **目标数量**：`MIN_PLATFORMS_LIMIT` (3) 到 `MAX_PLATFORMS` (8)
- **动态调整**：基于稀疏因子 `get_sparse_factor()`

#### 生成位置约束
1. **高度限制**：
   - 预生成区域：`[highest_y - 150, highest_y - 50]`
   - 可见区域：`[-50, player.y + HEIGHT*0.2]`
   - **禁止区域**：`y > max_height` (已到达区域)

2. **可达性检查**：
   - 基于物理轨迹计算 (`calculate_jump_trajectory`)
   - 考虑重力、跳跃高度、水平移动距离
   - 最大跳跃高度：`MAX_JUMP_HEIGHT = 40` 像素
   - 最大水平距离：`MAX_HORIZONTAL_REACH = 288` 像素

3. **间距控制**：
   - 基础间距：`[30, 120]` 像素
   - 动态调整：随高度增加，间距增大（重叠区域减小）
   - 调整范围：分数1000-10000，间距增加30%

#### 平台类型概率
```python
# 基础概率
BASE_NORMAL_PROB = 0.85   # 普通平台
BASE_BOOST_PROB = 0.1     # 奖励平台
BASE_OBSTACLE_PROB = 0.05 # 障碍物（初始）

# 动态调整
# 1. 稀疏因子影响（normal和boost）
adjusted_normal = BASE_NORMAL_PROB * sparse_factor
adjusted_boost = BASE_BOOST_PROB * sparse_factor

# 2. 障碍物概率随高度增加
obstacle_prob = f(score)  # 0.05 -> 0.15 (分数1000-8000)

# 3. 归一化确保总和=1
```

#### 难度递增机制
1. **平台稀疏度**：
   - 分数 < 2000：不稀疏
   - 分数 2000-10000：线性稀疏
   - 分数 ≥ 10000：稀疏因子 = 0.5

2. **障碍物概率**：
   - 分数 < 1000：5%
   - 分数 1000-8000：5% → 15%
   - 分数 ≥ 8000：15%

3. **平台间距**：
   - 分数 < 1000：基础间距
   - 分数 1000-10000：间距逐渐增大
   - 分数 ≥ 10000：间距增加30%

#### 安全约束
- **障碍物安全检查**：
  - 与玩家距离 ≥ 150像素
  - 不在玩家下落路径上
  - 可见区域内最多2个（低高度时最多1个）
  - 确保上方有可通行路径

- **水平重叠控制**：
  - 普通/奖励平台：最大重叠30%
  - 障碍物：无重叠限制

---

## 关键参数速查表

| 参数类别 | 参数名 | 值 | 说明 |
|---------|--------|-----|------|
| **屏幕** | WIDTH | 480 | 屏幕宽度（像素） |
| | HEIGHT | 600 | 屏幕高度（像素） |
| | FPS | 60 | 帧率 |
| **玩家** | PLAYER_ACC | 0.5 | 水平加速度 |
| | PLAYER_FRICTION | -0.12 | 摩擦力系数 |
| | PLAYER_GRAVITY | 0.8 | 重力加速度 |
| | PLAYER_JUMP | 20 | 跳跃初速度 |
| **平台** | PLATFORM_WIDTH | 100 | 平台宽度 |
| | PLATFORM_HEIGHT | 20 | 平台高度 |
| | MAX_JUMP_HEIGHT | 40 | 最大跳跃高度 |
| | MAX_HORIZONTAL_REACH | 288 | 最大水平距离 |
| **难度** | HEIGHT_SPARSE_START | 2000 | 开始稀疏的分数 |
| | HEIGHT_SPARSE_LIMIT | 10000 | 极限稀疏的分数 |
| | OBSTACLE_PROB_START | 1000 | 障碍物开始增加的分数 |
| | OBSTACLE_PROB_LIMIT | 8000 | 障碍物最大概率的分数 |
| | SPACING_ADJUST_START | 1000 | 间距开始调整的分数 |
| | SPACING_ADJUST_LIMIT | 10000 | 间距最大调整的分数 |

---

## DQN模型设计建议

### 网络输入
- **当前状态**：5维向量（可扩展至10-15维）
- **建议扩展**：包含最近3个平台的相对位置和类型

### 网络输出
- **Q值**：3个动作的Q值 `[Q(left), Q(noop), Q(right)]`

### 训练建议
- **经验回放**：Buffer size ≥ 10,000
- **目标网络**：使用Double DQN减少高估
- **探索策略**：ε-greedy，ε从1.0衰减到0.01
- **奖励裁剪**：限制在`[-10, 10]`避免极端值
- **状态归一化**：确保所有特征在`[-1, 1]`或`[0, 1]`范围

### 超参数建议
```python
LEARNING_RATE = 5e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
BUFFER_SIZE = 100000
UPDATE_EVERY = 4
TAU = 1e-3  # 软更新系数
```

