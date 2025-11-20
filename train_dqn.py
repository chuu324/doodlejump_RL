"""
DQN训练脚本
根据RL环境核心要素总结.md文档进行训练
"""

import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
from dqn_model import DQNAgent
from rl_env import DoodleJumpEnv
import time
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 训练参数
N_EPISODES = 600  # 训练回合数
MAX_STEPS = 20000  # 每个回合最大步数
SCORE_WINDOW = 100  # 计算平均分数的窗口大小
SAVE_INTERVAL = 100  # 每隔多少回合保存一次模型
BEST_CHECK_INTERVAL = 50  # 每隔多少回合检查一次最佳平均分
PRINT_INTERVAL = 10  # 每隔多少回合打印一次信息

# 模型保存路径
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_checkpoint.pth")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "dqn_best.pth")

def train():
    """训练DQN智能体"""
    
    # 创建模型保存目录
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 创建环境（不使用渲染，加快训练速度）
    env = DoodleJumpEnv(render_mode=None)
    
    # 创建智能体
    state_size = env.observation_space.shape[0]  # 27 (使用周期性编码后)
    action_size = env.action_space.n  # 3
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # 训练统计
    scores = []  # 每个回合的分数
    scores_window = deque(maxlen=SCORE_WINDOW)  # 最近100回合的分数
    best_score = -np.inf  # 单个回合最佳分数
    best_mean_score = -np.inf  # 历史最佳平均分数（用于保存模型，每50回合更新）
    display_best_mean_score = -np.inf  # 显示用的最佳平均分数（每10回合更新）
    episode_durations = []  # 每个回合的持续时间
    
    print("=" * 60)
    print("开始训练DQN智能体")
    print(f"状态空间维度: {state_size}")
    print(f"动作空间大小: {action_size}")
    # 显示设备信息
    if torch.cuda.is_available():
        print(f"✓ 使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠ 使用CPU（建议使用GPU加速训练）")
    print("=" * 60)
    
    start_time = time.time()
    
    for episode in range(1, N_EPISODES + 1):
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
        
        # 更新探索率
        agent.update_epsilon()
        
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
        
        # 每10回合更新显示用的最佳平均分数（仅用于显示，不保存模型）
        if episode % PRINT_INTERVAL == 0 and len(scores_window) >= BEST_CHECK_INTERVAL:
            if mean_score > display_best_mean_score:
                display_best_mean_score = mean_score
        
        # 在第一次有足够数据时初始化最佳平均分数并保存模型
        if best_mean_score == -np.inf and len(scores_window) >= BEST_CHECK_INTERVAL:
            best_mean_score = mean_score
            display_best_mean_score = mean_score
            agent.save(BEST_MODEL_PATH)
            print(f"\n🎉 初始化最佳平均分数: {best_mean_score:.1f} (回合 {episode}, 最近{len(scores_window)}回合平均)")
        
        # 每50回合检查一次平均分，如果超过历史最佳平均分则保存为最佳模型
        best_model_saved = False
        if episode % BEST_CHECK_INTERVAL == 0 and len(scores_window) >= BEST_CHECK_INTERVAL:
            if mean_score > best_mean_score:
                best_mean_score = mean_score
                display_best_mean_score = mean_score  # 同步更新显示值
                agent.save(BEST_MODEL_PATH)
                best_model_saved = True
                print(f"\n🎉 新的最佳平均分数: {best_mean_score:.1f} (回合 {episode}, 最近{len(scores_window)}回合平均)")
        
        # 打印训练信息
        if episode % PRINT_INTERVAL == 0:
            mean_duration = np.mean(episode_durations[-PRINT_INTERVAL:])
            elapsed_time = time.time() - start_time
            
            # 计算平均损失（最近1000次更新，如果不足1000次则使用全部）
            if len(agent.losses) > 0:
                recent_losses = agent.losses[-1000:] if len(agent.losses) >= 1000 else agent.losses
                avg_loss = np.mean(recent_losses)
                loss_str = f"Avg Loss: {avg_loss:.4f}"
            else:
                loss_str = "Avg Loss: N/A"
            
            # 格式化最佳平均分数显示（使用显示用的最佳平均分，如果还是-inf则显示当前平均分）
            best_mean_display = display_best_mean_score if display_best_mean_score != -np.inf else mean_score
            
            print(f"回合 {episode:4d} | "
                  f"平均分数: {mean_score:7.1f} | "
                  f"当前分数: {score:7.1f} | "
                  f"最佳分数: {best_score:7.1f} | "
                  f"最佳平均: {best_mean_display:7.1f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"步数: {step+1:4d} | "
                  f"时间: {mean_duration:.2f}s | "
                  f"{loss_str}")
        
        # 定期保存模型checkpoint（不覆盖历史checkpoint）
        if episode % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(MODEL_DIR, f"dqn_checkpoint_{episode}.pth")
            agent.save(checkpoint_path)
            if best_model_saved:
                print(f"Checkpoint已保存: {checkpoint_path} (同时已保存最佳模型)")
            else:
                print(f"Checkpoint已保存: {checkpoint_path}")
    
    # 训练结束，保存最终模型
    agent.save(MODEL_PATH)
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"总训练时间: {total_time/60:.2f} 分钟")
    print(f"最佳单回合分数: {best_score:.1f}")
    print(f"最佳平均分数: {best_mean_score:.1f}")
    print(f"最后100回合平均分数: {np.mean(scores_window):.1f}")
    print("=" * 60)
    
    # 绘制训练曲线
    plot_training_curve(scores, scores_window)
    
    env.close()
    return agent, scores


def plot_training_curve(scores, scores_window):
    """绘制训练曲线"""
    try:
        plt.figure(figsize=(12, 5))
        
        # 子图1: 所有回合的分数
        plt.subplot(1, 2, 1)
        plt.plot(scores, alpha=0.3, color='blue', label='每回合分数')
        if len(scores_window) > 0:
            window_scores = [np.mean(list(scores_window)[:i+1]) for i in range(len(scores_window))]
            plt.plot(range(len(scores) - len(scores_window) + 1, len(scores) + 1), 
                    window_scores, color='red', linewidth=2, label=f'平均分数 ({SCORE_WINDOW}回合)')
        plt.xlabel('回合')
        plt.ylabel('分数')
        plt.title('训练过程 - 分数曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 最近100回合的平均分数
        plt.subplot(1, 2, 2)
        if len(scores_window) > 0:
            window_scores = list(scores_window)
            plt.plot(window_scores, alpha=0.5, color='blue', label='最近100回合分数')
            if len(window_scores) >= 10:
                # 移动平均
                moving_avg = np.convolve(window_scores, np.ones(10)/10, mode='valid')
                plt.plot(range(9, len(window_scores)), moving_avg, 
                        color='red', linewidth=2, label='10回合移动平均')
        plt.xlabel('回合 (最近100回合)')
        plt.ylabel('分数')
        plt.title('最近100回合分数趋势')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_curve.png', dpi=150, bbox_inches='tight')
        print("训练曲线已保存到: training_curve.png")
        plt.close()
    except Exception as e:
        print(f"绘制训练曲线时出错: {e}")


def test_agent(model_path=None, n_episodes=5, render=True):
    """
    测试训练好的智能体
    
    Args:
        model_path: 模型路径，如果为None则使用最佳模型
        n_episodes: 测试回合数
        render: 是否渲染
    """
    if model_path is None:
        model_path = BEST_MODEL_PATH
    
    # 创建环境
    render_mode = "human" if render else None
    env = DoodleJumpEnv(render_mode=render_mode)
    
    # 创建智能体并加载模型
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    if not agent.load(model_path):
        print("无法加载模型，使用随机策略")
        return
    
    # 设置探索率为0（完全贪婪）
    agent.epsilon = 0.0
    
    print(f"\n开始测试智能体 (模型: {model_path})")
    print("=" * 60)
    
    test_scores = []
    
    for episode in range(1, n_episodes + 1):
        state, info = env.reset()
        score = 0
        step = 0
        
        while True:
            # 使用贪婪策略（不探索）
            action = agent.act(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            score = info['score']
            step += 1
            
            if render:
                # 处理pygame事件
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
            
            if terminated or truncated:
                break
        
        test_scores.append(score)
        print(f"测试回合 {episode}: 分数 = {score:.1f}, 步数 = {step}")
    
    env.close()
    
    print("=" * 60)
    print(f"测试完成！")
    print(f"平均分数: {np.mean(test_scores):.1f}")
    print(f"最高分数: {np.max(test_scores):.1f}")
    print(f"最低分数: {np.min(test_scores):.1f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DQN训练和测试')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                       help='运行模式: train 或 test')
    parser.add_argument('--model', type=str, default=None,
                       help='测试时使用的模型路径')
    parser.add_argument('--episodes', type=int, default=5,
                       help='测试回合数')
    parser.add_argument('--no-render', action='store_true',
                       help='测试时不渲染')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test_agent(model_path=args.model, n_episodes=args.episodes, render=not args.no_render)

