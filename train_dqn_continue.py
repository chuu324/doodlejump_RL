"""
DQN继续训练脚本
从600回合的最佳模型继续训练，使用调整后的超参数
"""

import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
import dqn_model
from dqn_model import DQNAgent
from rl_env import DoodleJumpEnv
import time
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ========== 继续训练的超参数配置 ==========

# 训练配置
START_EPISODE = 601        # 从601回合开始
NUM_EPISODES = 900         # 训练到900回合
CONTINUE_FROM_MODEL = 'models/dqn_best.pth'  # 加载的模型路径

# 探索率策略 (关键调整)
EPSILON_START = 0.20       # 重置探索率到20%
EPSILON_END = 0.10         # 最低10%(避免过度利用)
EPSILON_DECAY = 0.9992     # 极慢衰减(原0.998)

# 学习率
LEARNING_RATE = 1e-4       # 降低学习率(原3e-4)

# 更新策略
UPDATE_EVERY = 3           # 每3步更新(原2,更稳定)
TAU = 0.005                # 目标网络软更新系数(原0.003,加快同步)

# 批次大小
BATCH_SIZE = 128           # 保持

# 缓冲区
BUFFER_SIZE = 200000       # 保持

# 折扣因子
GAMMA = 0.95               # 保持

# 梯度裁剪
GRADIENT_CLIP = 0.5        # 保持

# 训练参数
MAX_STEPS = 20000          # 每个回合最大步数
SCORE_WINDOW = 100         # 计算平均分数的窗口大小
SAVE_INTERVAL = 50         # 每隔多少回合保存一次检查点
BEST_CHECK_INTERVAL = 50   # 每隔多少回合检查一次最佳平均分
PRINT_INTERVAL = 10        # 每隔多少回合打印一次信息

# 模型保存路径
MODEL_DIR = "models/continue"
CONTINUE_CHECKPOINT_PREFIX = "continue_checkpoint_ep"
CONTINUE_BEST_PREFIX = "continue_best_ep"


def train_continue():
    """从已有模型继续训练DQN智能体"""
    
    # 创建模型保存目录
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # ========== 覆盖模块级常量 ==========
    # 临时修改dqn_model模块中的常量，以便在agent中使用新的超参数
    original_update_every = dqn_model.UPDATE_EVERY
    original_tau = dqn_model.TAU
    original_gamma = dqn_model.GAMMA
    dqn_model.UPDATE_EVERY = UPDATE_EVERY
    dqn_model.TAU = TAU
    dqn_model.GAMMA = GAMMA
    
    # 创建环境（不使用渲染，加快训练速度）
    env = DoodleJumpEnv(render_mode=None)
    
    # 创建智能体
    state_size = env.observation_space.shape[0]  # 27 (使用周期性编码后)
    action_size = env.action_space.n  # 3
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # ========== 加载已训练的模型 ==========
    print(f"\n{'='*80}")
    print(f"加载模型: {CONTINUE_FROM_MODEL}")
    if not agent.load(CONTINUE_FROM_MODEL):
        print(f"❌ 无法加载模型: {CONTINUE_FROM_MODEL}")
        print("请确保模型文件存在！")
        # 恢复原始常量
        dqn_model.UPDATE_EVERY = original_update_every
        dqn_model.TAU = original_tau
        dqn_model.GAMMA = original_gamma
        return None, None
    
    # 覆盖探索率(重置到EPSILON_START)
    agent.epsilon = EPSILON_START
    agent.epsilon_decay = EPSILON_DECAY
    agent.epsilon_min = EPSILON_END
    print(f"✓ 模型加载成功")
    print(f"  重置探索率: {agent.epsilon:.3f}")
    print(f"  探索率衰减: {EPSILON_DECAY}")
    print(f"  最低探索率: {EPSILON_END}")
    
    # 调整学习率
    for param_group in agent.optimizer.param_groups:
        param_group['lr'] = LEARNING_RATE
    print(f"  设置学习率: {LEARNING_RATE}")
    
    # 重置损失记录
    agent.losses = []
    
    # 显示超参数配置
    print(f"\n超参数配置:")
    print(f"  UPDATE_EVERY: {UPDATE_EVERY}")
    print(f"  TAU: {TAU}")
    print(f"  GAMMA: {GAMMA}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  BUFFER_SIZE: {BUFFER_SIZE}")
    print(f"{'='*80}\n")
    
    # 训练统计
    scores = []  # 每个回合的分数
    scores_window = deque(maxlen=SCORE_WINDOW)  # 最近100回合的分数
    best_score = -np.inf  # 单个回合最佳分数
    best_mean_score = -np.inf  # 历史最佳平均分数
    display_best_mean_score = -np.inf  # 显示用的最佳平均分数
    episode_durations = []  # 每个回合的持续时间
    
    print("=" * 80)
    print("开始继续训练DQN智能体")
    print(f"训练回合: {START_EPISODE} → {NUM_EPISODES}")
    print(f"状态空间维度: {state_size}")
    print(f"动作空间大小: {action_size}")
    # 显示设备信息
    if torch.cuda.is_available():
        print(f"✓ 使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠ 使用CPU（建议使用GPU加速训练）")
    print("=" * 80)
    
    start_time = time.time()
    
    for episode in range(START_EPISODE, NUM_EPISODES + 1):
        # 更新智能体的当前回合数（用于分阶段梯度裁剪）
        agent.current_episode = episode
        
        # 重置环境
        state, info = env.reset()
        score = 0
        episode_start_time = time.time()
        
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
            
            # 如果游戏结束，跳出循环
            if terminated or truncated:
                break
        
        # 更新探索率（使用新的衰减策略）
        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)
        
        # 记录统计信息
        scores.append(score)
        scores_window.append(score)
        episode_duration = time.time() - episode_start_time
        episode_durations.append(episode_duration)
        
        # 更新单个回合最佳分数
        if score > best_score:
            best_score = score
        
        # 计算当前平均分数（用于显示和检查）
        mean_score = np.mean(scores_window) if len(scores_window) > 0 else 0
        
        # 每10回合更新显示用的最佳平均分数
        if episode % PRINT_INTERVAL == 0 and len(scores_window) >= BEST_CHECK_INTERVAL:
            if mean_score > display_best_mean_score:
                display_best_mean_score = mean_score
        
        # 在第一次有足够数据时初始化最佳平均分数并保存模型
        if best_mean_score == -np.inf and len(scores_window) >= BEST_CHECK_INTERVAL:
            best_mean_score = mean_score
            display_best_mean_score = mean_score
            best_model_path = os.path.join(MODEL_DIR, f"{CONTINUE_BEST_PREFIX}{episode}_score{mean_score:.1f}.pth")
            agent.save(best_model_path)
            print(f"\n🎉 初始化最佳平均分数: {best_mean_score:.1f} (回合 {episode}, 最近{len(scores_window)}回合平均)")
        
        # 每50回合检查一次平均分，如果超过历史最佳平均分则保存为最佳模型
        best_model_saved = False
        if episode % BEST_CHECK_INTERVAL == 0 and len(scores_window) >= BEST_CHECK_INTERVAL:
            if mean_score > best_mean_score:
                best_mean_score = mean_score
                display_best_mean_score = mean_score
                best_model_path = os.path.join(MODEL_DIR, f"{CONTINUE_BEST_PREFIX}{episode}_score{mean_score:.1f}.pth")
                agent.save(best_model_path)
                best_model_saved = True
                print(f"\n🎉 新的最佳平均分数: {best_mean_score:.1f} (回合 {episode}, 最近{len(scores_window)}回合平均)")
        
        # ========== 性能下降预警 ==========
        if episode >= START_EPISODE + 100 and len(scores) >= 100:
            recent_50 = np.mean(scores[-50:])
            baseline_100 = np.mean(scores[-100:-50])
            
            if recent_50 < baseline_100 * 0.95:
                print(f"\n⚠️  警告: 性能下降 {baseline_100:.0f} → {recent_50:.0f} (下降 {((baseline_100 - recent_50) / baseline_100 * 100):.1f}%)")
                print(f"   自动提升探索率: {agent.epsilon:.3f} → ", end="")
                agent.epsilon = min(agent.epsilon * 1.15, 0.25)
                print(f"{agent.epsilon:.3f}")
        
        # 打印训练信息
        if episode % PRINT_INTERVAL == 0:
            mean_duration = np.mean(episode_durations[-PRINT_INTERVAL:])
            elapsed_time = time.time() - start_time
            
            # 计算平均损失（最近1000次更新，如果不足1000次则使用全部）
            if len(agent.losses) > 0:
                recent_losses = agent.losses[-1000:] if len(agent.losses) >= 1000 else agent.losses
                avg_loss = np.mean(recent_losses)
                loss_str = f"{avg_loss:.2f}"
            else:
                loss_str = "N/A"
            
            # 获取当前学习率
            current_lr = agent.optimizer.param_groups[0]['lr']
            
            # 格式化最佳平均分数显示
            best_mean_display = display_best_mean_score if display_best_mean_score != -np.inf else mean_score
            
            print(f"回合 {episode:4d} | "
                  f"平均分: {mean_score:6.1f} | "
                  f"当前分: {score:6.1f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Loss: {loss_str:>6} | "
                  f"LR: {current_lr:.6f}")
        
        # 定期保存模型checkpoint
        if episode % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(MODEL_DIR, f"{CONTINUE_CHECKPOINT_PREFIX}{episode}.pth")
            agent.save(checkpoint_path)
            if best_model_saved:
                print(f"Checkpoint已保存: {checkpoint_path} (同时已保存最佳模型)")
            else:
                print(f"Checkpoint已保存: {checkpoint_path}")
    
    # 训练结束，保存最终模型
    final_checkpoint_path = os.path.join(MODEL_DIR, f"{CONTINUE_CHECKPOINT_PREFIX}{NUM_EPISODES}_final.pth")
    agent.save(final_checkpoint_path)
    total_time = time.time() - start_time
    
    # 恢复原始常量
    dqn_model.UPDATE_EVERY = original_update_every
    dqn_model.TAU = original_tau
    dqn_model.GAMMA = original_gamma
    
    print("\n" + "=" * 80)
    print("继续训练完成!")
    print(f"  训练回合: {START_EPISODE} → {NUM_EPISODES}")
    print(f"  总训练时间: {total_time/60:.2f} 分钟")
    print(f"  最佳单回合分数: {best_score:.1f}")
    print(f"  最佳平均分数: {best_mean_score:.1f}")
    print(f"  最后100回合平均分数: {np.mean(scores_window):.1f}")
    print(f"  最终探索率: {agent.epsilon:.3f}")
    print(f"  最终学习率: {agent.optimizer.param_groups[0]['lr']:.6f}")
    print(f"  模型保存于: {MODEL_DIR}/{CONTINUE_BEST_PREFIX}*.pth")
    print("=" * 80)
    
    # 绘制训练曲线
    plot_training_curve(scores, scores_window, START_EPISODE)
    
    env.close()
    return agent, scores


def plot_training_curve(scores, scores_window, start_episode):
    """绘制训练曲线"""
    try:
        plt.figure(figsize=(14, 6))
        
        # 子图1: 所有回合的分数
        plt.subplot(1, 2, 1)
        episode_numbers = range(start_episode, start_episode + len(scores))
        plt.plot(episode_numbers, scores, alpha=0.3, color='blue', label='每回合分数')
        if len(scores_window) > 0:
            window_scores = [np.mean(list(scores_window)[:i+1]) for i in range(len(scores_window))]
            window_start = start_episode + len(scores) - len(scores_window)
            plt.plot(range(window_start, window_start + len(window_scores)), 
                    window_scores, color='red', linewidth=2, label=f'平均分数 ({SCORE_WINDOW}回合)')
        plt.xlabel('回合')
        plt.ylabel('分数')
        plt.title(f'继续训练过程 - 分数曲线 (回合 {start_episode}-{start_episode + len(scores) - 1})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 最近100回合的平均分数
        plt.subplot(1, 2, 2)
        if len(scores_window) > 0:
            window_scores = list(scores_window)
            window_start = start_episode + len(scores) - len(scores_window)
            plt.plot(range(window_start, window_start + len(window_scores)), 
                    window_scores, alpha=0.5, color='blue', label='最近100回合分数')
            if len(window_scores) >= 10:
                # 移动平均
                moving_avg = np.convolve(window_scores, np.ones(10)/10, mode='valid')
                plt.plot(range(window_start + 9, window_start + len(window_scores)), 
                        moving_avg, color='red', linewidth=2, label='10回合移动平均')
        plt.xlabel('回合')
        plt.ylabel('分数')
        plt.title('最近100回合分数趋势')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_curve_continue.png', dpi=150, bbox_inches='tight')
        print("训练曲线已保存到: training_curve_continue.png")
        plt.close()
    except Exception as e:
        print(f"绘制训练曲线时出错: {e}")


if __name__ == "__main__":
    train_continue()

