import pygame
from game import Game
from model_test import find_model_files, display_model_list, load_model, test_model, display_results
from rl_env import DoodleJumpEnv
import os

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

def run_model_test():
    """运行模型测试功能（从models文件夹中选择模型进行测试）"""
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
    print("请选择运行模式:")
    print("1: 手动游戏 (Human Play)")
    print("2: 测试DQN模型 (从models文件夹中选择模型)")
    
    choice = input("输入选项 (1/2): ")

    if choice == '1':
        run_manual_play()
    elif choice == '2':
        run_model_test()
    else:
        print("无效选项。")