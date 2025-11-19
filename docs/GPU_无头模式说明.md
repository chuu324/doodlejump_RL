# GPU和无头模式设置说明

## 问题解决

### 1. 训练时出现可视化界面的问题

**问题原因**：
- `Game` 类在初始化时总是会调用 `pygame.display.set_mode()` 创建窗口
- 即使设置了 `render_mode=None`，窗口仍然会被创建

**解决方案**：
- 修改了 `rl_env.py`，在 `render_mode=None` 时：
  - Linux/Mac: 设置 `SDL_VIDEODRIVER='dummy'` 环境变量
  - Windows: 创建窗口后立即关闭 `pygame.display.quit()`
- 修改了 `game.py` 的 `draw()` 和 `game_step_for_rl()` 方法，检查 `screen` 是否为 `None`，如果是则跳过绘制操作

**效果**：
- 训练时不会弹出游戏窗口
- 训练速度更快（无需渲染）
- 测试/演示时仍可正常显示

### 2. GPU设置

**自动检测**：
- 代码会自动检测是否有可用的CUDA GPU
- 如果有GPU，自动使用GPU进行训练
- 如果没有GPU，使用CPU（会显示警告）

**GPU信息显示**：
训练开始时会自动显示：
- GPU名称
- CUDA版本
- GPU内存大小

**手动指定GPU**（如果需要）：
如果有多块GPU，可以在 `dqn_model.py` 中修改：
```python
device = torch.device("cuda:0")  # 使用第一块GPU
device = torch.device("cuda:1")  # 使用第二块GPU
```

## 使用方法

### 训练（无头模式，自动使用GPU）

```bash
python train_dqn.py --mode train
```

训练时会：
- ✅ 不显示游戏窗口（无头模式）
- ✅ 自动使用GPU（如果可用）
- ✅ 显示GPU信息

### 测试（显示窗口）

```bash
python train_dqn.py --mode test --episodes 5
```

测试时会：
- ✅ 显示游戏窗口
- ✅ 使用训练好的模型
- ✅ 使用GPU进行推理（如果可用）

### 检查GPU是否可用

在Python中运行：
```python
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
```

## 性能对比

### CPU vs GPU

- **CPU训练**：较慢，适合小规模测试
- **GPU训练**：快10-100倍（取决于GPU），推荐用于正式训练

### 有渲染 vs 无渲染

- **有渲染**：训练速度慢，适合调试和演示
- **无渲染**：训练速度快3-5倍，适合正式训练

## 常见问题

### Q: 训练时仍然出现窗口？

**A**: 
1. 确保使用 `render_mode=None`（训练脚本已设置）
2. 在Windows上，窗口可能会短暂出现然后关闭，这是正常的
3. 如果窗口一直存在，检查是否有其他代码创建了pygame窗口

### Q: 如何确认正在使用GPU？

**A**: 
1. 训练开始时查看输出，应该显示 "✓ 使用GPU: ..."
2. 使用 `nvidia-smi` 命令（Linux/Windows）查看GPU使用情况
3. 如果显示 "⚠ 使用CPU"，说明没有检测到GPU

### Q: GPU内存不足怎么办？

**A**:
1. 减小 `BATCH_SIZE`（在 `dqn_model.py` 中，默认64）
2. 减小网络大小（隐藏层神经元数量）
3. 减小经验回放缓冲区大小 `BUFFER_SIZE`

### Q: 如何在Windows上完全禁用窗口？

**A**: 
Windows上pygame可能会短暂显示窗口。如果完全不想看到窗口：
1. 使用WSL（Windows Subsystem for Linux）
2. 使用远程服务器训练
3. 或者接受窗口会快速关闭的事实（不影响训练）

## 技术细节

### 无头模式实现

1. **Linux/Mac**: 使用 `SDL_VIDEODRIVER='dummy'` 环境变量
2. **Windows**: 创建窗口后立即关闭，并设置 `screen = None`
3. **游戏逻辑**: 检查 `screen` 是否为 `None`，如果是则跳过所有绘制操作

### GPU使用

- 所有PyTorch张量自动移动到GPU
- 网络模型在GPU上训练
- 经验回放数据在CPU上（减少GPU内存占用）
- 批次数据在训练时临时移动到GPU

## 修改的文件

1. **rl_env.py**: 添加无头模式支持
2. **game.py**: 添加无头模式检查和numpy导入
3. **dqn_model.py**: 改进GPU检测和显示
4. **train_dqn.py**: 添加GPU信息显示

