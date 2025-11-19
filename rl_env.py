import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os
from game import Game # 导入你的核心游戏

class DoodleJumpEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # 如果不需要渲染，设置环境变量禁用pygame显示（无头模式）
        if render_mode is None:
            # 在Linux/Mac上，dummy驱动可以避免创建窗口
            # 在Windows上，我们稍后关闭display
            if os.name != 'nt':  # 非Windows系统
                os.environ['SDL_VIDEODRIVER'] = 'dummy'
        
        self.game = Game() # 创建一个游戏实例
        self.game.running = True # 确保游戏开始时是 running 状态
        
        # 如果不需要渲染，关闭屏幕（避免创建窗口）
        if render_mode is None:
            if hasattr(self.game, 'screen') and self.game.screen is not None:
                try:
                    pygame.display.quit()
                except:
                    pass
                self.game.screen = None
        
        # 1. 定义动作空间 (Action Space)
        # 0: 向左, 1: 不动, 2: 向右
        self.action_space = spaces.Discrete(3)

        # 2. 定义状态空间 (Observation Space)
        # 扩展后的状态空间（使用周期性编码处理环形边界）：
        # - 玩家基础状态：x_sin, x_cos, y, vel_y, vel_x (5维，x从1维变为2维周期性编码)
        # - 最近障碍物：dx, dy, dist, type (4维，dx使用环形距离)
        # - 最近平台：dx, dy, type (3维，dx使用环形距离)
        # - 上方5个平台：每个平台(dx, dy, type) (15维，dx使用环形距离)
        # 总计：5 + 4 + 3 + 15 = 27维
        state_dim = 27
        low = np.array([-np.inf] * state_dim, dtype=np.float32)
        high = np.array([np.inf] * state_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        if self.render_mode == "human":
            # 如果之前设置了dummy驱动，需要重新初始化
            if 'SDL_VIDEODRIVER' in os.environ and os.environ['SDL_VIDEODRIVER'] == 'dummy':
                del os.environ['SDL_VIDEODRIVER']
            pygame.display.init()
            self.game.screen = pygame.display.set_mode((self.game.WIDTH, self.game.HEIGHT))
        elif self.render_mode == "rgb_array":
            # rgb_array模式需要屏幕但不显示窗口
            if 'SDL_VIDEODRIVER' in os.environ and os.environ['SDL_VIDEODRIVER'] == 'dummy':
                del os.environ['SDL_VIDEODRIVER']
            if self.game.screen is None:
                pygame.display.init()
                self.game.screen = pygame.display.set_mode((self.game.WIDTH, self.game.HEIGHT))

    def _get_obs(self):
        """
        扩展的状态空间（使用周期性编码处理环形边界）
        包含：玩家状态、水平速度、障碍物信息、多平台信息
        """
        player = self.game.player
        
        # ========== 1. 玩家基础状态 ==========
        # 使用周期性编码处理x坐标（sin/cos对），保持边界连续性
        player_x_sin, player_x_cos = self.game.periodic_encode(player.pos.x, self.game.WIDTH)
        player_y = player.pos.y / self.game.HEIGHT  # [0, 1]
        player_vel_y = np.clip(player.vel.y / 30.0, -1.0, 1.0)  # 归一化垂直速度，假设最大30
        player_vel_x = np.clip(player.vel.x / 5.0, -1.0, 1.0)  # ⭐ 新增：水平速度，假设最大5像素/帧
        
        obs = [player_x_sin, player_x_cos, player_y, player_vel_y, player_vel_x]
        
        # ========== 2. 找到最近的障碍物 ==========
        closest_obstacle = None
        min_obstacle_dist = float('inf')
        
        for plat in self.game.platforms:
            if plat.type == 'obstacle':
                # 使用环形距离计算
                dist_x = self.game.toroidal_distance(plat.rect.centerx, player.pos.x, self.game.WIDTH)
                dist_y = plat.rect.y - player.pos.y
                dist = dist_x**2 + dist_y**2
                if dist < min_obstacle_dist:
                    min_obstacle_dist = dist
                    closest_obstacle = plat
        
        if closest_obstacle:
            # 使用环形相对位置
            obs_dx = self.game.toroidal_dx(player.pos.x, closest_obstacle.rect.centerx, self.game.WIDTH) / self.game.WIDTH
            # 使用平台顶部位置（玩家站在平台顶部）
            obs_dy = (closest_obstacle.rect.y - player.pos.y) / self.game.HEIGHT
            obs_dist = np.sqrt(min_obstacle_dist) / np.sqrt(self.game.WIDTH**2 + self.game.HEIGHT**2)  # 归一化距离
            obs_type = 1.0  # 障碍物类型编码
        else:
            obs_dx, obs_dy, obs_dist, obs_type = 0.0, 0.0, 1.0, 0.0  # 无障碍物时设为默认值
        
        obs.extend([obs_dx, obs_dy, obs_dist, obs_type])
        
        # ========== 3. 找到最近的平台（非障碍物） ==========
        closest_platform = None
        min_platform_dist = float('inf')
        
        for plat in self.game.platforms:
            if plat.type != 'obstacle':
                # 使用环形距离计算
                dist_x = self.game.toroidal_distance(plat.rect.centerx, player.pos.x, self.game.WIDTH)
                dist_y = plat.rect.y - player.pos.y
                dist = dist_x**2 + dist_y**2
                if dist < min_platform_dist:
                    min_platform_dist = dist
                    closest_platform = plat
        
        if closest_platform:
            # 使用环形相对位置
            plat_dx = self.game.toroidal_dx(player.pos.x, closest_platform.rect.centerx, self.game.WIDTH) / self.game.WIDTH
            # 使用平台顶部位置（玩家站在平台顶部）
            plat_dy = (closest_platform.rect.y - player.pos.y) / self.game.HEIGHT
            plat_type = 1.0 if closest_platform.type == 'boost' else 0.0  # boost=1.0, normal=0.0
        else:
            plat_dx, plat_dy, plat_type = 0.0, 0.0, 0.0
        
        obs.extend([plat_dx, plat_dy, plat_type])
        
        # ========== 4. 上方3-5个平台的信息 ==========
        # 找到玩家上方的平台，按距离排序
        platforms_above = []
        for plat in self.game.platforms:
            if plat.rect.y < player.pos.y:  # 在玩家上方（使用rect.y，平台顶部）
                # 使用环形距离计算
                dist_x = self.game.toroidal_distance(plat.rect.centerx, player.pos.x, self.game.WIDTH)
                dist_y = plat.rect.y - player.pos.y
                dist = dist_x**2 + dist_y**2
                platforms_above.append((dist, plat))
        
        platforms_above.sort(key=lambda x: x[0])  # 按距离排序
        platforms_above = platforms_above[:5]  # 取最近的5个
        
        # 为每个平台添加：dx, dy, type
        for i in range(5):
            if i < len(platforms_above):
                plat = platforms_above[i][1]
                # 使用环形相对位置
                plat_dx = self.game.toroidal_dx(player.pos.x, plat.rect.centerx, self.game.WIDTH) / self.game.WIDTH
                # 使用平台顶部位置（玩家站在平台顶部）
                plat_dy = (plat.rect.y - player.pos.y) / self.game.HEIGHT
                # 类型编码：normal=0.0, boost=0.5, obstacle=1.0
                if plat.type == 'normal':
                    plat_type = 0.0
                elif plat.type == 'boost':
                    plat_type = 0.5
                else:  # obstacle
                    plat_type = 1.0
                obs.extend([plat_dx, plat_dy, plat_type])
            else:
                # 如果平台不足，用0填充
                obs.extend([0.0, 0.0, 0.0])
        
        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        # 返回额外信息，比如分数
        return {"score": self.game.score}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # 处理随机种子
        
        self.game.new_game() # 重置游戏
        self.last_score = 0
        self.last_obstacle_dist = None  # 用于跟踪上一帧的障碍物距离（奖励塑形）
        self.last_obstacle_close = False  # 用于判断是否成功规避障碍物
        
        obs = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.game.draw()
            
        return obs, info

    def step(self, action):
        # 执行一个动作
        
        # 1. 执行游戏步骤
        self.game.game_step_for_rl(action)
        
        # 2. 计算奖励 (Reward Shaping - 根据DQN训练瓶颈分析文档)
        terminated = not self.game.running # 游戏是否结束
        reward = 0.0
        
        if terminated:
            # 死亡惩罚
            reward = -100.0
        else:
            # ========== 基础奖励 ==========
            # 存活奖励（每步）
            reward += 0.05
            
            # 高度奖励（缩小尺度，避免与死亡惩罚不平衡）
            score_diff = self.game.score - self.last_score
            reward += score_diff * 0.9
            self.last_score = self.game.score
            
            # ========== 奖励塑形：障碍物相关 ==========
            player = self.game.player
            
            # 找到最近的障碍物
            closest_obstacle = None
            min_obstacle_dist = float('inf')
            
            for plat in self.game.platforms:
                if plat.type == 'obstacle':
                    # 使用环形距离计算
                    dist_x = self.game.toroidal_distance(plat.rect.centerx, player.pos.x, self.game.WIDTH)
                    dist_y = plat.rect.centery - player.pos.y
                    dist = np.sqrt(dist_x**2 + dist_y**2)
                    if dist < min_obstacle_dist:
                        min_obstacle_dist = dist
                        closest_obstacle = plat
            
            if closest_obstacle:
                # 归一化距离（0-1，1表示很远，0表示很近）
                max_dist = np.sqrt(self.game.WIDTH**2 + self.game.HEIGHT**2)
                normalized_dist = min(min_obstacle_dist / max_dist, 1.0)
                
                # 接近障碍物惩罚（距离越近，惩罚越大）
                # 当距离 < 200像素时，开始有惩罚
                danger_threshold = 200.0
                if min_obstacle_dist < danger_threshold:
                    danger_factor = 1.0 - (min_obstacle_dist / danger_threshold)
                    obstacle_penalty = -0.1 * danger_factor  # 最大惩罚-0.1
                    reward += obstacle_penalty
                
                # 成功规避奖励：如果上一帧障碍物很近，这一帧变远了，给予奖励
                if self.last_obstacle_close and min_obstacle_dist > danger_threshold:
                    reward += 1.0  # ⭐ 成功规避奖励
                    self.last_obstacle_close = False
                elif min_obstacle_dist < danger_threshold:
                    self.last_obstacle_close = True
                else:
                    self.last_obstacle_close = False
                
                self.last_obstacle_dist = min_obstacle_dist
            else:
                # 无障碍物时，如果上一帧有障碍物且很近，说明成功规避了
                if self.last_obstacle_close:
                    reward += 1.0  # ⭐ 成功规避奖励
                self.last_obstacle_close = False
                self.last_obstacle_dist = None
            
            # ========== 奖励塑形：位置相关 ==========
            # 接近屏幕底部惩罚（鼓励向上）
            bottom_threshold = self.game.HEIGHT * 0.8  # 屏幕底部80%位置
            if player.pos.y > bottom_threshold:
                bottom_penalty = -0.5 * ((player.pos.y - bottom_threshold) / (self.game.HEIGHT - bottom_threshold))
                reward += bottom_penalty

        # 3. 获取新的观测值和信息
        obs = self._get_obs()
        info = self._get_info()
        
        # 4. 渲染 (如果需要)
        if self.render_mode == "human":
            self.game.draw()
        
        # Gymnasium API 要求返回 (obs, reward, terminated, truncated, info)
        # truncated 在这个游戏中可以设为 False
        return obs, reward, terminated, False, info

    def render(self):
        # 渲染
        if self.render_mode == "rgb_array":
            return self.game.get_screen_rgb()
        elif self.render_mode == "human":
            self.game.draw() # game_step_for_rl 内部已经调用了 draw
            return None

    def close(self):
        pygame.quit()