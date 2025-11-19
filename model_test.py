"""
模型测试脚本
用于测试已训练好的DQN模型性能
支持交互式选择模型、自定义测试次数、可视化游戏窗口
"""

import os
import numpy as np
import torch
import pygame
from dqn_model import DQNAgent, get_device
from rl_env import DoodleJumpEnv
import time

device = get_device()


def find_model_files(models_dir="models"):
    """
    查找models目录中的所有模型文件(.pth)
    
    Args:
        models_dir: 模型目录路径
        
    Returns:
        model_files: 模型文件路径列表
    """
    if not os.path.exists(models_dir):
        print(f"错误: 模型目录 '{models_dir}' 不存在")
        return []
    
    model_files = []
    for file in os.listdir(models_dir):
        if file.endswith('.pth'):
            model_path = os.path.join(models_dir, file)
            model_files.append(model_path)
    
    return sorted(model_files)


def display_model_list(model_files):
    """
    显示模型文件列表
    
    Args:
        model_files: 模型文件路径列表
    """
    if not model_files:
        print("未找到任何模型文件!")
        return
    
    print("\n" + "=" * 60)
    print("可用的模型文件:")
    print("=" * 60)
    for i, model_path in enumerate(model_files, 1):
        # 获取文件大小
        file_size = os.path.getsize(model_path) / 1024  # KB
        # 获取文件名
        filename = os.path.basename(model_path)
        print(f"  {i}. {filename} ({file_size:.1f} KB)")
    print("=" * 60)


def load_model(model_path, state_size=27, action_size=3):
    """
    加载DQN模型
    
    Args:
        model_path: 模型文件路径
        state_size: 状态空间维度
        action_size: 动作空间大小
        
    Returns:
        agent: 加载好的DQNAgent实例，如果失败返回None
    """
    try:
        agent = DQNAgent(state_size=state_size, action_size=action_size)
        if agent.load(model_path):
            # 测试时使用贪婪策略（不探索）
            agent.epsilon = 0.0
            return agent
        else:
            return None
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None


def test_model(agent, env, n_episodes=3):
    """
    测试模型性能
    
    Args:
        agent: DQNAgent实例
        env: DoodleJumpEnv环境实例
        n_episodes: 测试回合数
        
    Returns:
        results: 测试结果字典，包含分数、步数等信息
    """
    print(f"\n开始测试模型 ({n_episodes} 个回合)...")
    print("=" * 60)
    
    scores = []
    steps_list = []
    durations = []
    
    for episode in range(1, n_episodes + 1):
        print(f"\n回合 {episode}/{n_episodes} 进行中...")
        
        # 重置环境
        state, info = env.reset()
        score = 0
        step = 0
        episode_start_time = time.time()
        
        while True:
            # 使用贪婪策略选择动作（不探索）
            action = agent.act(state, training=False)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # 更新状态和分数
            state = next_state
            score = info['score']
            step += 1
            
            # 处理pygame事件（允许关闭窗口）
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\n用户关闭了游戏窗口，测试中断")
                    env.close()
                    return {
                        'scores': scores,
                        'steps': steps_list,
                        'durations': durations,
                        'interrupted': True
                    }
            
            # 游戏结束
            if terminated or truncated:
                break
        
        # 记录结果
        episode_duration = time.time() - episode_start_time
        scores.append(score)
        steps_list.append(step)
        durations.append(episode_duration)
        
        print(f"  分数: {score:.1f} | 步数: {step} | 时长: {episode_duration:.2f}秒")
    
    env.close()
    
    # 计算统计数据
    results = {
        'scores': scores,
        'steps': steps_list,
        'durations': durations,
        'avg_score': np.mean(scores),
        'max_score': np.max(scores),
        'min_score': np.min(scores),
        'std_score': np.std(scores),
        'avg_steps': np.mean(steps_list),
        'avg_duration': np.mean(durations),
        'interrupted': False
    }
    
    return results


def display_results(results, model_name, n_episodes):
    """
    显示测试结果
    
    Args:
        results: 测试结果字典
        model_name: 模型名称
        n_episodes: 测试回合数
    """
    if results['interrupted']:
        print("\n" + "=" * 60)
        print("测试被中断")
        if results['scores']:
            print(f"已完成 {len(results['scores'])}/{n_episodes} 回合")
            print(f"已完成回合的平均分数: {np.mean(results['scores']):.1f}")
        print("=" * 60)
        return
    
    print("\n" + "=" * 60)
    print(f"测试结果 - {model_name}")
    print("=" * 60)
    print(f"测试回合数: {len(results['scores'])}")
    print(f"\n分数统计:")
    print(f"  平均分数: {results['avg_score']:.1f}")
    print(f"  最高分数: {results['max_score']:.1f}")
    print(f"  最低分数: {results['min_score']:.1f}")
    print(f"  标准差:   {results['std_score']:.1f}")
    print(f"\n步数统计:")
    print(f"  平均步数: {results['avg_steps']:.1f}")
    print(f"\n时长统计:")
    print(f"  平均时长: {results['avg_duration']:.2f}秒")
    print(f"\n各回合详情:")
    for i, (score, steps, duration) in enumerate(zip(results['scores'], 
                                                      results['steps'], 
                                                      results['durations']), 1):
        print(f"  回合 {i}: 分数={score:.1f}, 步数={steps}, 时长={duration:.2f}秒")
    print("=" * 60)


def main():
    """主函数"""
    print("=" * 60)
    print("DQN模型测试工具")
    print("=" * 60)
    
    # 1. 查找模型文件
    model_files = find_model_files()
    
    if not model_files:
        print("错误: 未找到任何模型文件!")
        print("请确保 models 目录中存在 .pth 模型文件")
        return
    
    # 2. 显示模型列表
    display_model_list(model_files)
    
    # 3. 让用户选择模型
    while True:
        try:
            choice = input(f"\n请选择要测试的模型 (1-{len(model_files)}, 输入 'q' 退出): ").strip()
            
            if choice.lower() == 'q':
                print("退出测试")
                return
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(model_files):
                selected_model = model_files[choice_idx]
                break
            else:
                print(f"错误: 请输入 1-{len(model_files)} 之间的数字")
        except ValueError:
            print("错误: 请输入有效的数字或 'q' 退出")
        except KeyboardInterrupt:
            print("\n\n测试取消")
            return
    
    # 4. 获取测试次数
    while True:
        try:
            n_episodes_input = input("\n请输入测试次数 (默认 3, 按回车使用默认值): ").strip()
            
            if n_episodes_input == '':
                n_episodes = 3
            else:
                n_episodes = int(n_episodes_input)
                if n_episodes <= 0:
                    print("错误: 测试次数必须大于 0")
                    continue
            
            break
        except ValueError:
            print("错误: 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n测试取消")
            return
    
    # 5. 创建环境（可视化模式）
    print("\n正在创建游戏环境（可视化模式）...")
    env = DoodleJumpEnv(render_mode='human')
    
    # 获取状态和动作空间大小
    state_size = env.observation_space.shape[0]  # 27
    action_size = env.action_space.n  # 3
    
    # 6. 加载模型
    print(f"正在加载模型: {os.path.basename(selected_model)}")
    agent = load_model(selected_model, state_size=state_size, action_size=action_size)
    
    if agent is None:
        print("错误: 无法加载模型")
        env.close()
        return
    
    print("✓ 模型加载成功")
    
    # 7. 测试模型
    try:
        results = test_model(agent, env, n_episodes=n_episodes)
        
        # 8. 显示结果
        display_results(results, os.path.basename(selected_model), n_episodes)
        
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    main()

