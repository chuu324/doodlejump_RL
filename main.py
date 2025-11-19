import pygame
import imageio
import numpy as np
from game import Game
from rl_env import DoodleJumpEnv
import gymnasium as gym

def run_manual_play():
    """运行手动游戏模式"""
    print("正在运行: 手动游戏模式")
    print("=" * 50)
    print("游戏控制说明:")
    print("  ← 左箭头键: 向左移动")
    print("  → 右箭头键: 向右移动")
    print("  空格键: 跳跃（需要踩在平台上）")
    print("  关闭窗口: 退出游戏")
    print("=" * 50)
    print("提示: 玩家会自动跳跃，使用左右键控制方向即可")
    g = Game()
    g.new_game()
    g.run_manual()
    pygame.quit()

def run_rl_agent(agent, env, steps=1000, record_path=None):
    """
    运行一个RL智能体 (或随机智能体)
    agent: 'random' 或一个DQNAgent实例
    env: DoodleJumpEnv 实例
    steps: 运行多少步
    record_path: (可选) 视频保存路径
    """
    
    if record_path:
        print(f"准备录制视频: {record_path}")
        # 使用 imageio 开始录制
        writer = imageio.get_writer(record_path, fps=env.metadata["render_fps"])
    
    obs, info = env.reset()
    terminated = False
    
    for step in range(steps):
        if terminated:
            print(f"游戏结束，总分: {info['score']}")
            obs, info = env.reset()
            terminated = False
            
        # 1. 获取动作
        if agent == 'random':
            # 随机智能体 (对比视频 "之前")
            action = env.action_space.sample()
        elif hasattr(agent, 'act'):
            # DQN智能体
            action = agent.act(obs, training=False)  # 测试模式，不使用探索
        else:
            # 未知类型，使用随机动作
            print("警告: 未知的agent类型, 使用随机动作代替。")
            action = env.action_space.sample() 

        # 2. 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 3. 录制视频帧
        if record_path:
            # 获取RGB图像数据并写入
            rgb_frame = env.render()
            writer.append_data(rgb_frame)
            
        # 如果是 human 模式，需要额外处理事件
        if env.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
        
    print("运行结束。")
    if record_path:
        writer.close()
        print(f"视频已保存至: {record_path}")
    env.close()


if __name__ == "__main__":
    print("请选择运行模式:")
    print("1: 手动游戏 (Human Play)")
    print("2: 录制 '强化学习前' (随机Agent) 对比视频 (before_rl.mp4)")
    print("3: 录制 '强化学习后' (训练好的DQN Agent) 对比视频 (after_rl.mp4)")
    print("4: 运行RL环境 (Human模式，用于调试)")
    print("5: 使用训练好的DQN模型玩游戏 (Human模式)")
    
    choice = input("输入选项 (1/2/3/4/5): ")

    if choice == '1':
        run_manual_play()
        
    elif choice == '2':
        print("录制 '强化学习前' 视频...")
        # 'rgb_array' 模式在后台渲染，速度快，用于录制
        env = DoodleJumpEnv(render_mode='rgb_array')
        run_rl_agent(agent='random', env=env, steps=2000, record_path='before_rl.mp4')
        
    elif choice == '3':
        print("录制 '强化学习后' 视频...")
        # 加载训练好的DQN模型
        try:
            from dqn_model import DQNAgent
            import os
            
            model_path = "models/dqn_best.pth"
            if not os.path.exists(model_path):
                model_path = "models/dqn_checkpoint.pth"
            
            if os.path.exists(model_path):
                env = DoodleJumpEnv(render_mode='rgb_array')
                state_size = env.observation_space.shape[0]
                action_size = env.action_space.n
                trained_agent = DQNAgent(state_size=state_size, action_size=action_size)
                trained_agent.load(model_path)
                trained_agent.epsilon = 0.0  # 测试时使用贪婪策略
                print(f"已加载模型: {model_path}")
                run_rl_agent(agent=trained_agent, env=env, steps=2000, record_path='after_rl.mp4')
            else:
                print(f"错误: 找不到模型文件 {model_path}")
                print("请先运行 train_dqn.py 训练模型")
        except ImportError:
            print("错误: 无法导入DQNAgent，请确保dqn_model.py存在")
        except Exception as e:
            print(f"错误: {e}")

    elif choice == '4':
        print("运行RL环境 (Human模式)...")
        # 'human' 模式会弹出一个窗口
        env = DoodleJumpEnv(render_mode='human')
        # 用随机Agent跑
        run_rl_agent(agent='random', env=env, steps=5000)
    
    elif choice == '5':
        print("使用训练好的DQN模型玩游戏...")
        try:
            from dqn_model import DQNAgent
            import os
            
            model_path = "models/dqn_best.pth"
            if not os.path.exists(model_path):
                model_path = "models/dqn_checkpoint.pth"
            
            if os.path.exists(model_path):
                env = DoodleJumpEnv(render_mode='human')
                state_size = env.observation_space.shape[0]
                action_size = env.action_space.n
                trained_agent = DQNAgent(state_size=state_size, action_size=action_size)
                trained_agent.load(model_path)
                trained_agent.epsilon = 0.0  # 测试时使用贪婪策略
                print(f"已加载模型: {model_path}")
                print("按关闭窗口退出")
                run_rl_agent(agent=trained_agent, env=env, steps=10000)
            else:
                print(f"错误: 找不到模型文件 {model_path}")
                print("请先运行 train_dqn.py 训练模型")
        except ImportError:
            print("错误: 无法导入DQNAgent，请确保dqn_model.py存在")
        except Exception as e:
            print(f"错误: {e}")
        
    else:
        print("无效选项。")