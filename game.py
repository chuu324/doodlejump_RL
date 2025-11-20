import pygame
import random
import os
import numpy as np
from settings import *
from sprites import Player, Platform

class Game:
    def __init__(self, enable_sound=True):
        """
        初始化游戏
        
        Args:
            enable_sound: 是否启用音效（训练时建议设为False以提高性能）
        """
        self.enable_sound = enable_sound
        
        # !! 初始化 Pygame 和 Mixer !!
        if enable_sound:
            pygame.mixer.pre_init(44100, -16, 2, 512) # 优化音效设置
        pygame.init()
        if enable_sound:
            pygame.mixer.init() # 正式启动音效模块
        
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(TITLE)
        
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.score = 0
        self.font = pygame.font.SysFont(None, 30)
        
        # 调用资源加载
        self.load_assets()

    def load_assets(self):
        """
        在游戏开始时预加载所有图片和音效资源
        如果资源文件不存在，将使用默认图形
        """
        print("正在加载资源...")
        
        # 加载玩家图片
        self.player_image = None
        if PLAYER_IMG_PATH and os.path.exists(PLAYER_IMG_PATH):
            try:
                self.player_image = pygame.image.load(PLAYER_IMG_PATH).convert_alpha()
                print(f"✓ 已加载玩家图片: {PLAYER_IMG_PATH}")
            except Exception as e:
                print(f"✗ 加载玩家图片失败: {e}，将使用默认图形")
        else:
            print(f"✗ 玩家图片不存在: {PLAYER_IMG_PATH}，将使用默认图形")
        
        # 加载普通平台图片
        self.platform_normal_images = []
        for img_path in PLATFORM_NORMAL_IMGS:
            if os.path.exists(img_path):
                try:
                    img = pygame.image.load(img_path).convert_alpha()
                    self.platform_normal_images.append(img)
                    print(f"✓ 已加载普通平台图片: {img_path}")
                except Exception as e:
                    print(f"✗ 加载普通平台图片失败: {img_path}, {e}")
        if not self.platform_normal_images:
            print("✗ 未找到普通平台图片，将使用默认蓝色方块")
        
        # 加载奖励平台图片
        self.reward_images = []
        for img_path in REWARD_IMGS_LIST:
            if os.path.exists(img_path):
                try:
                    img = pygame.image.load(img_path).convert_alpha()
                    self.reward_images.append(img)
                    print(f"✓ 已加载奖励平台图片: {img_path}")
                except Exception as e:
                    print(f"✗ 加载奖励平台图片失败: {img_path}, {e}")
        if not self.reward_images:
            print("✗ 未找到奖励平台图片，将使用默认浅绿色方块")

        # 加载障碍物图片
        self.obstacle_images = []
        for img_path in OBSTACLE_IMGS_LIST:
            if os.path.exists(img_path):
                try:
                    img = pygame.image.load(img_path).convert_alpha()
                    self.obstacle_images.append(img)
                    print(f"✓ 已加载障碍物图片: {img_path}")
                except Exception as e:
                    print(f"✗ 加载障碍物图片失败: {img_path}, {e}")
        if not self.obstacle_images:
            print("✗ 未找到障碍物图片，将使用默认红色方块")

        # 加载奖励音效列表（仅在启用音效时加载）
        self.reward_sounds = []
        if self.enable_sound:
            for snd_path in REWARD_SOUNDS_LIST:
                if os.path.exists(snd_path):
                    try:
                        self.reward_sounds.append(pygame.mixer.Sound(snd_path))
                        print(f"✓ 已加载奖励音效: {snd_path}")
                    except Exception as e:
                        print(f"✗ 加载奖励音效失败: {snd_path}, {e}")
            if not self.reward_sounds:
                print("✗ 未找到奖励音效文件")
        else:
            print("音效已禁用，跳过音效加载")

        # 加载游戏结束音效（仅在启用音效时加载）
        self.game_over_sound = None
        if self.enable_sound:
            if os.path.exists(GAME_OVER_SOUND):
                try:
                    self.game_over_sound = pygame.mixer.Sound(GAME_OVER_SOUND)
                    print(f"✓ 已加载游戏结束音效: {GAME_OVER_SOUND}")
                except Exception as e:
                    print(f"✗ 加载游戏结束音效失败: {GAME_OVER_SOUND}, {e}")
            else:
                print(f"✗ 游戏结束音效不存在: {GAME_OVER_SOUND}")

        # 加载里程碑音效（仅在启用音效时加载）
        self.milestone_sound = None
        if self.enable_sound:
            if os.path.exists(MILESTONE_SOUND):
                try:
                    self.milestone_sound = pygame.mixer.Sound(MILESTONE_SOUND)
                    print(f"✓ 已加载里程碑音效: {MILESTONE_SOUND}")
                except Exception as e:
                    print(f"✗ 加载里程碑音效失败: {MILESTONE_SOUND}, {e}")
            else:
                print(f"✗ 里程碑音效不存在: {MILESTONE_SOUND}")
        
        print("资源加载完成！")


    def new_game(self):
        # 开始新游戏
        self.score = 0
        self.max_height = HEIGHT  # 追踪玩家到达的最高位置（Y坐标，越小越高）
        self.all_sprites = pygame.sprite.Group()
        self.platforms = pygame.sprite.Group()
        
        self.player = Player(self)
        self.all_sprites.add(self.player)
        
        # 重置分数里程碑 
        self.next_score_milestone = SCORE_MILESTONE_TARGET
        
        # 创建初始平台
        for plat in PLATFORM_LIST:
            p = Platform(self, plat[0], plat[1], PLATFORM_WIDTH, PLATFORM_HEIGHT)
            self.all_sprites.add(p)
            self.platforms.add(p)
            
        self.running = True

    def run_manual(self):
        # 手动模式的主循环
        self.playing = True
        while self.playing:
            self.clock.tick(FPS)
            self.events_manual()
            self.update()
            self.draw()

    def periodic_encode(self, x, width):
        """
        周期性编码：将线性x坐标转换为周期性编码(sin/cos对)
        用于处理环形边界的连续性
        
        Args:
            x: 线性x坐标（像素）
            width: 屏幕宽度（用于归一化）
        
        Returns:
            (sin_value, cos_value): 周期性编码的sin和cos值
        """
        # 将x归一化到[0, 1]，然后映射到[0, 2π]
        normalized = (x % width) / width
        angle = normalized * 2 * np.pi
        return np.sin(angle), np.cos(angle)
    
    def toroidal_distance(self, x1, x2, width):
        """
        计算环形距离（考虑边界环绕）
        返回两个x坐标之间的最短环形距离
        
        Args:
            x1: 第一个x坐标
            x2: 第二个x坐标
            width: 屏幕宽度
        
        Returns:
            环形距离（像素）
        """
        # 计算直接距离
        direct_dist = abs(x2 - x1)
        # 计算环绕距离（通过边界）
        wrap_dist = width - direct_dist
        # 返回较小的距离
        return min(direct_dist, wrap_dist)
    
    def toroidal_dx(self, x1, x2, width):
        """
        计算环形相对位置（考虑边界环绕）
        返回从x1到x2的环形相对位置，范围在[-width/2, width/2]
        
        Args:
            x1: 起点x坐标
            x2: 终点x坐标
            width: 屏幕宽度
        
        Returns:
            环形相对位置（像素），正数表示x2在x1右侧（考虑环绕），负数表示左侧
        """
        # 计算直接距离
        direct_dx = x2 - x1
        # 计算环绕距离
        if direct_dx > width / 2:
            # 通过左边界更近
            return direct_dx - width
        elif direct_dx < -width / 2:
            # 通过右边界更近
            return direct_dx + width
        else:
            return direct_dx
    
    def get_sparse_factor(self):
        """
        根据当前高度计算稀疏因子
        返回0-1之间的值，1表示不稀疏，0表示最稀疏
        """
        if self.score < HEIGHT_SPARSE_START:
            return 1.0  # 低高度时不稀疏
        elif self.score >= HEIGHT_SPARSE_LIMIT:
            return SPARSE_FACTOR  # 达到极限稀疏度
        else:
            # 线性插值
            progress = (self.score - HEIGHT_SPARSE_START) / (HEIGHT_SPARSE_LIMIT - HEIGHT_SPARSE_START)
            return 1.0 - progress * (1.0 - SPARSE_FACTOR)
    
    def get_obstacle_probability(self):
        """
        根据当前高度计算障碍物概率
        初始较低，随高度增加而缓慢增加
        返回0-1之间的概率值
        """
        if self.score < OBSTACLE_PROB_START:
            return BASE_OBSTACLE_PROB  # 低高度时使用初始低概率
        elif self.score >= OBSTACLE_PROB_LIMIT:
            return BASE_OBSTACLE_PROB_MAX  # 高高度时使用最大概率
        else:
            # 线性插值，缓慢增加
            progress = (self.score - OBSTACLE_PROB_START) / (OBSTACLE_PROB_LIMIT - OBSTACLE_PROB_START)
            return BASE_OBSTACLE_PROB + progress * (BASE_OBSTACLE_PROB_MAX - BASE_OBSTACLE_PROB)
    
    def get_target_platform_count(self):
        """
        根据稀疏因子计算目标平台数量
        """
        sparse_factor = self.get_sparse_factor()
        target_count = MIN_PLATFORMS_LIMIT + (MAX_PLATFORMS - MIN_PLATFORMS_LIMIT) * sparse_factor
        return max(int(target_count), MIN_PLATFORMS_LIMIT)
    
    def get_spacing_adjust_factor(self):
        """
        根据当前高度计算间距调整因子
        返回0.7-1.0之间的值，值越小间距越大（重叠区域越小）
        """
        if self.score < SPACING_ADJUST_START:
            return MAX_SPACING_ADJUST_FACTOR  # 低高度时不调整
        elif self.score >= SPACING_ADJUST_LIMIT:
            return MIN_SPACING_ADJUST_FACTOR  # 高高度时达到最大调整
        else:
            # 线性插值，缓慢渐进
            progress = (self.score - SPACING_ADJUST_START) / (SPACING_ADJUST_LIMIT - SPACING_ADJUST_START)
            return MAX_SPACING_ADJUST_FACTOR - progress * (MAX_SPACING_ADJUST_FACTOR - MIN_SPACING_ADJUST_FACTOR)
    
    def get_platform_spacing(self):
        """
        根据稀疏因子和高度动态计算平台间距范围
        随着高度增加，间距逐渐增大（重叠区域减小）
        """
        sparse_factor = self.get_sparse_factor()
        spacing_adjust_factor = self.get_spacing_adjust_factor()
        
        # 基础间距（基于稀疏因子）
        base_min_spacing = MIN_PLATFORM_SPACING + (MIN_PLATFORM_SPACING_LIMIT - MIN_PLATFORM_SPACING) * (1 - sparse_factor)
        base_max_spacing = MAX_PLATFORM_SPACING + (MAX_PLATFORM_SPACING_LIMIT - MAX_PLATFORM_SPACING) * (1 - sparse_factor)
        
        # 应用间距调整因子（值越小，间距越大）
        # spacing_adjust_factor从1.0到0.7，间距从基础值增加到基础值的1/0.7倍
        adjusted_min_spacing = base_min_spacing / spacing_adjust_factor
        adjusted_max_spacing = base_max_spacing / spacing_adjust_factor
        
        # 确保间距在可达性范围内（不超过最大跳跃高度的80%）
        max_allowed_spacing = MAX_JUMP_HEIGHT * 0.8
        adjusted_min_spacing = min(adjusted_min_spacing, max_allowed_spacing * 0.5)
        adjusted_max_spacing = min(adjusted_max_spacing, max_allowed_spacing)
        
        # 确保最小间距不超过最大间距
        if adjusted_min_spacing > adjusted_max_spacing:
            adjusted_min_spacing = adjusted_max_spacing * 0.7
        
        return adjusted_min_spacing, adjusted_max_spacing
    
    def calculate_jump_trajectory(self, start_x, start_y, jump_velocity_y, horizontal_velocity, target_x, target_y):
        """
        计算跳跃轨迹，判断是否能到达目标位置
        考虑重力加速度的影响
        返回(是否可达, 到达时间)
        """
        # 初始速度
        v0_y = jump_velocity_y
        v0_x = horizontal_velocity
        
        # 目标相对位置（使用环形距离）
        dx = self.toroidal_dx(start_x, target_x, self.WIDTH)
        dy = target_y - start_y
        
        # 如果目标在起点下方，需要检查是否在合理范围内
        if dy > 0:
            # 目标在下方，检查水平距离是否合理（使用环形距离）
            toroidal_dist_x = abs(dx)
            if toroidal_dist_x > MAX_HORIZONTAL_REACH:
                return False, 0
            # 如果目标在下方不太远，认为可达（玩家可以水平移动后下落）
            if dy < MAX_JUMP_HEIGHT * 0.5:
                return True, toroidal_dist_x / max(abs(v0_x), 1.0)
            return False, 0
        
        # 目标在上方，计算垂直运动
        # 运动方程: y = v0_y * t + 0.5 * g * t^2
        # 其中 g = PLAYER_GRAVITY (正值，向下)
        # 当到达最高点时: v = v0_y + g * t = 0, 所以 t = -v0_y / g
        # 最高点高度: h_max = v0_y^2 / (2 * g)
        
        # 计算到达目标高度所需的时间
        # dy = v0_y * t + 0.5 * g * t^2
        # 0.5 * g * t^2 + v0_y * t - dy = 0
        # 使用二次方程求解
        a = 0.5 * PLAYER_GRAVITY
        b = v0_y
        c = -dy
        
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return False, 0
        
        # 计算两个解（上升和下降阶段）
        t1 = (-b + (discriminant ** 0.5)) / (2 * a)
        t2 = (-b - (discriminant ** 0.5)) / (2 * a)
        
        # 选择正数解（时间必须为正）
        t = None
        if t1 > 0 and t2 > 0:
            t = min(t1, t2)  # 选择较早的时间（上升阶段）
        elif t1 > 0:
            t = t1
        elif t2 > 0:
            t = t2
        else:
            return False, 0
        
        # 计算水平位置（使用环形距离）
        # 考虑水平加速度和摩擦力
        # 简化：假设水平速度在跳跃过程中基本保持
        horizontal_dist = abs(v0_x * t)
        
        # 检查水平距离是否可达（使用环形距离）
        # 注意：这里我们使用直接距离，因为v0_x的方向已经考虑了环形特性
        if horizontal_dist > MAX_HORIZONTAL_REACH:
            return False, 0
        
        # 检查是否在最大跳跃高度内
        max_height = (v0_y * v0_y) / (2 * PLAYER_GRAVITY)
        if abs(dy) > max_height * 1.1:  # 留10%余量
            return False, 0
        
        return True, t
    
    def is_platform_reachable(self, new_x, new_y, from_platform=None):
        """
        检查新平台是否可达
        基于物理轨迹计算，考虑重力加速度的影响
        """
        if not self.platforms:
            return True  # 如果没有现有平台，允许生成
        
        # 如果指定了参考平台，使用它；否则找到最近的平台
        if from_platform:
            reference_platforms = [from_platform]
        else:
            # 找到最近的几个可通行平台作为参考（最多3个）
            candidate_platforms = []
            for plat in self.platforms:
                if plat.type != 'obstacle':  # 只考虑可踩的平台
                    # 使用环形距离计算
                    dist_x = self.toroidal_distance(plat.rect.centerx, new_x, self.WIDTH)
                    dist_y = abs(plat.rect.y - new_y)
                    dist = (dist_x ** 2 + dist_y ** 2) ** 0.5
                    candidate_platforms.append((dist, plat))
            
            if not candidate_platforms:
                return True  # 如果没有可通行平台，允许生成
            
            # 按距离排序，取最近的几个
            candidate_platforms.sort(key=lambda x: x[0])
            reference_platforms = [plat for _, plat in candidate_platforms[:3]]
        
        # 检查是否至少有一个参考平台可以到达新平台
        for ref_plat in reference_platforms:
            start_x = ref_plat.rect.centerx
            start_y = ref_plat.rect.top  # 玩家站在平台顶部
            
            # 计算从平台起跳的初始速度
            jump_velocity_y = -PLAYER_JUMP  # 向上跳跃
            # 估算水平速度（考虑加速度和摩擦力）
            # 简化：假设玩家可以调整到最大水平速度
            # 使用环形相对位置确定方向
            toroidal_dx = self.toroidal_dx(start_x, new_x, self.WIDTH)
            max_horizontal_speed = PLAYER_ACC / abs(PLAYER_FRICTION) if PLAYER_FRICTION != 0 else PLAYER_ACC * 10
            horizontal_velocity = max_horizontal_speed if toroidal_dx > 0 else -max_horizontal_speed
            
            # 计算轨迹
            is_reachable, _ = self.calculate_jump_trajectory(
                start_x, start_y, jump_velocity_y, horizontal_velocity, new_x, new_y
            )
            
            if is_reachable:
                return True
        
        return False
    
    def check_upward_path_available(self, check_y_range=250):
        """
        检查玩家上方是否有可通行的路径
        check_y_range: 检查的范围（像素）
        返回(是否有可通行平台, 可通行平台数量, 障碍物数量)
        """
        if not self.platforms:
            return True, 0, 0  # 没有平台时，允许生成
        
        player_y = self.player.pos.y
        check_min_y = player_y - check_y_range
        check_max_y = player_y - 20  # 玩家上方一点，避免检查玩家当前位置
        
        # 统计上方区域内的平台
        safe_platforms = 0  # 可通行平台（normal或boost）
        obstacle_platforms = 0  # 障碍物平台
        
        for plat in self.platforms:
            if check_min_y <= plat.rect.y <= check_max_y:
                if plat.type == 'obstacle':
                    obstacle_platforms += 1
                else:
                    safe_platforms += 1
        
        return safe_platforms > 0, safe_platforms, obstacle_platforms
    
    def is_in_visible_area(self, y):
        """
        检查Y坐标是否在可见区域内
        """
        return PLATFORM_VISIBLE_Y_MIN <= y <= PLATFORM_VISIBLE_Y_MAX
    
    def calculate_horizontal_overlap(self, x1, width1, x2, width2):
        """
        计算两个平台的水平重叠长度
        返回重叠长度
        """
        # 计算两个区间的重叠
        left1, right1 = x1, x1 + width1
        left2, right2 = x2, x2 + width2
        
        # 计算重叠区间
        overlap_left = max(left1, left2)
        overlap_right = min(right1, right2)
        
        # 如果有重叠，返回重叠长度；否则返回0
        if overlap_left < overlap_right:
            return overlap_right - overlap_left
        return 0
    
    def check_excessive_horizontal_overlap(self, new_x, new_y, platform_type):
        """
        检查新平台与现有可通行平台的水平重叠是否过大
        只对normal和boost平台进行检查，鼓励左右移动
        返回True如果重叠过大，False如果重叠可接受
        """
        # 只对可通行平台（normal和boost）进行检查
        if platform_type == 'obstacle':
            return False  # 障碍物不检查重叠
        
        # 检查附近的可通行平台
        for plat in self.platforms:
            # 只检查可通行平台
            if plat.type == 'obstacle':
                continue
            
            # 只检查垂直距离在范围内的平台
            vertical_dist = abs(plat.rect.y - new_y)
            if vertical_dist > OVERLAP_CHECK_Y_RANGE:
                continue
            
            # 计算水平重叠
            overlap = self.calculate_horizontal_overlap(
                new_x, PLATFORM_WIDTH,
                plat.rect.x, PLATFORM_WIDTH
            )
            
            # 计算重叠比例（相对于平台宽度）
            overlap_ratio = overlap / PLATFORM_WIDTH
            
            # 如果重叠超过阈值，认为重叠过大
            if overlap_ratio > MAX_HORIZONTAL_OVERLAP_RATIO:
                return True
        
        return False
    
    def check_obstacle_safety(self, new_x, new_y):
        """
        检查障碍物生成是否安全
        返回(是否安全, 原因)
        """
        # 1. 检查与玩家的距离
        player_x = self.player.pos.x
        player_y = self.player.pos.y
        
        # 计算与玩家的距离（使用环形距离）
        obstacle_center_x = new_x + PLATFORM_WIDTH / 2
        dist_x = self.toroidal_distance(obstacle_center_x, player_x, self.WIDTH)
        dist_y = abs(new_y - player_y)
        total_dist = (dist_x ** 2 + dist_y ** 2) ** 0.5
        
        # 如果障碍物太靠近玩家，不安全
        if total_dist < OBSTACLE_SAFE_DISTANCE:
            return False, "too_close_to_player"
        
        # 2. 检查是否在玩家下落路径上
        # 如果障碍物在玩家下方，且水平位置重叠，可能阻挡下落
        if new_y > player_y:
            # 检查水平位置是否重叠（考虑环形边界）
            obstacle_left = new_x
            obstacle_right = new_x + PLATFORM_WIDTH
            player_left = player_x - 20  # 玩家宽度的一半估算
            player_right = player_x + 20
            
            # 检查水平重叠（考虑环形边界）
            # 计算障碍物中心到玩家中心的环形距离
            obstacle_center = obstacle_left + PLATFORM_WIDTH / 2
            toroidal_dx = self.toroidal_dx(player_x, obstacle_center, self.WIDTH)
            # 如果环形距离小于平台宽度的一半+玩家宽度的一半，认为重叠
            if abs(toroidal_dx) < (PLATFORM_WIDTH / 2 + 20):
                # 检查垂直距离
                if new_y - player_y < OBSTACLE_SAFE_DISTANCE:
                    return False, "in_fall_path"
        
        # 3. 检查可见区域内的障碍物数量
        if self.is_in_visible_area(new_y):
            visible_obstacles = 0
            for plat in self.platforms:
                if plat.type == 'obstacle' and self.is_in_visible_area(plat.rect.y):
                    visible_obstacles += 1
            
            # 如果可见区域内障碍物太多，不安全
            if visible_obstacles >= OBSTACLE_MAX_IN_VISIBLE:
                return False, "too_many_in_visible"
            
            # 在低高度时，进一步限制可见区域内的障碍物
            if self.score < OBSTACLE_MIN_HEIGHT_FOR_MULTIPLE:
                if visible_obstacles >= 1:  # 低高度时最多1个
                    return False, "too_many_at_low_height"
        
        # 4. 检查玩家上方是否有足够的可通行平台
        # 如果障碍物在玩家上方，确保上方还有其他可通行平台
        if new_y < player_y:
            upward_path_available, safe_count, obstacle_count = self.check_upward_path_available()
            # 如果上方只有障碍物，不安全
            if not upward_path_available and obstacle_count > 0:
                return False, "blocks_upward_path"
        
        return True, "safe"
    
    def count_obstacles_in_range(self, min_y, max_y):
        """
        统计指定Y范围内障碍物的数量
        """
        count = 0
        for plat in self.platforms:
            if plat.type == 'obstacle' and min_y <= plat.rect.y <= max_y:
                count += 1
        return count
    
    def generate_platform(self, min_y, max_y, force_safe=False, preferred_reference=None):
        """
        生成一个平台，考虑可达性和稀疏度
        force_safe: 如果为True，强制生成可通行平台（normal或boost）
        preferred_reference: 优先使用的参考平台（用于可达性检查）
        """
        max_attempts = 30  # 增加尝试次数
        sparse_factor = self.get_sparse_factor()
        min_spacing, max_spacing = self.get_platform_spacing()
        
        # 确保min_y和max_y是整数，并且min_y < max_y
        min_y = int(min_y)
        max_y = int(max_y)
        if min_y >= max_y:
            max_y = min_y + 1  # 确保至少有一个整数的范围
        
        # 找到玩家上方的参考平台（用于可达性检查）
        reference_platforms = []
        if self.platforms:
            for plat in self.platforms:
                if plat.type != 'obstacle' and plat.rect.y < self.player.pos.y:
                    reference_platforms.append(plat)
            # 按距离玩家排序
            reference_platforms.sort(key=lambda p: abs(p.rect.y - self.player.pos.y))
        
        # 如果指定了优先参考平台，放在最前面
        if preferred_reference and preferred_reference in reference_platforms:
            reference_platforms.remove(preferred_reference)
            reference_platforms.insert(0, preferred_reference)
        
        for attempt in range(max_attempts):
            # 生成位置
            x = random.randrange(0, WIDTH - PLATFORM_WIDTH)
            y = random.randrange(min_y, max_y)
            
            # 优化5: 禁止在玩家历史最高高度以下的区域生成新平台
            # max_height是玩家到达的最高位置（Y坐标越小越高）
            # 如果新平台的Y坐标大于max_height，说明在玩家已到达过的区域下方，禁止生成
            if hasattr(self, 'max_height') and y > self.max_height:
                continue  # 跳过，不在已到达区域生成
            
            # 检查是否在可见区域内，如果是，检查是否已有平台（避免重复生成）
            if self.is_in_visible_area(y):
                # 检查该区域是否已有平台（使用环形距离）
                has_platform_nearby = False
                for plat in self.platforms:
                    if abs(plat.rect.y - y) < 50 and self.toroidal_distance(plat.rect.centerx, x, self.WIDTH) < PLATFORM_WIDTH * 2:
                        has_platform_nearby = True
                        break
                if has_platform_nearby:
                    continue  # 跳过，避免在可见区域重复生成
            
            # 检查是否与现有平台太近（使用环形距离）
            too_close = False
            for plat in self.platforms:
                dist_y = abs(plat.rect.y - y)
                dist_x = self.toroidal_distance(plat.rect.centerx, x, self.WIDTH)
                if dist_y < min_spacing and dist_x < PLATFORM_WIDTH * 1.5:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # 确定平台类型
            rand = random.random()
            
            # 如果强制生成可通行平台，或者平台在玩家上方且上方没有可通行路径
            if force_safe or (y < self.player.pos.y and not self.check_upward_path_available()[0]):
                # 强制生成可通行平台
                if rand < BASE_BOOST_PROB / (BASE_NORMAL_PROB + BASE_BOOST_PROB):
                    p_type = 'boost'
                else:
                    p_type = 'normal'
            else:
                # 获取动态障碍物概率
                obstacle_prob = self.get_obstacle_probability()
                
                # 计算调整后的概率（普通平台和奖励平台都乘以sparse_factor）
                adjusted_normal_prob = BASE_NORMAL_PROB * sparse_factor
                adjusted_boost_prob = BASE_BOOST_PROB * sparse_factor
                
                # 归一化概率，确保总和为1
                total_prob = adjusted_normal_prob + adjusted_boost_prob + obstacle_prob
                if total_prob > 0:
                    adjusted_obstacle_prob = obstacle_prob / total_prob
                    adjusted_normal_prob = adjusted_normal_prob / total_prob
                    adjusted_boost_prob = adjusted_boost_prob / total_prob
                else:
                    # 如果总概率为0（不应该发生），使用默认值
                    adjusted_obstacle_prob = 0.0
                    adjusted_normal_prob = 0.7
                    adjusted_boost_prob = 0.3
                
                # 根据概率选择平台类型
                if rand < adjusted_obstacle_prob:
                    p_type = 'obstacle'
                elif rand < adjusted_obstacle_prob + adjusted_boost_prob:
                    p_type = 'boost'
                else:
                    p_type = 'normal'
            
            # 障碍物安全检查
            if p_type == 'obstacle':
                is_safe, reason = self.check_obstacle_safety(x, y)
                if not is_safe:
                    # 如果安全检查失败，尝试降低障碍物概率或改为普通平台
                    if attempt < max_attempts - 5:
                        # 在前面的尝试中，直接跳过，继续尝试
                        continue
                    else:
                        # 在后面的尝试中，改为生成普通平台
                        p_type = 'normal'
            
            # 检查可达性
            # 如果平台在玩家上方，使用参考平台检查可达性
            if y < self.player.pos.y and reference_platforms:
                # 使用最近的参考平台检查可达性
                is_reachable = False
                for ref_plat in reference_platforms[:2]:  # 检查最近的2个参考平台
                    if self.is_platform_reachable(x, y, from_platform=ref_plat):
                        is_reachable = True
                        break
                if not is_reachable:
                    # 障碍物可以稍微放宽要求
                    if p_type == 'obstacle' and attempt < max_attempts - 10:
                        continue
                    elif p_type != 'obstacle':
                        continue
            else:
                # 使用默认的可达性检查
                if not self.is_platform_reachable(x, y):
                    # 障碍物可以稍微放宽要求
                    if p_type == 'obstacle' and attempt < max_attempts - 10:
                        continue
                    elif p_type != 'obstacle':
                        continue
            
            # 检查水平重叠（只对普通平台和奖励平台）
            if p_type != 'obstacle':
                if self.check_excessive_horizontal_overlap(x, y, p_type):
                    # 对于可通行平台，如果重叠过大，尝试重新生成位置
                    # 在后半段尝试中，更积极地避免重叠
                    if attempt > max_attempts // 2:
                        # 尝试找到一个与现有平台水平位置差异较大的位置
                        nearby_safe_platforms = []
                        for plat in self.platforms:
                            if plat.type != 'obstacle' and abs(plat.rect.y - y) < OVERLAP_CHECK_Y_RANGE:
                                nearby_safe_platforms.append(plat.rect.centerx)
                        
                        if nearby_safe_platforms:
                            # 计算当前x与所有附近平台的平均距离（使用环形距离）
                            avg_distance = sum(self.toroidal_distance(x, px, self.WIDTH) for px in nearby_safe_platforms) / len(nearby_safe_platforms)
                            
                            # 如果平均距离太小，尝试调整x位置
                            if avg_distance < PLATFORM_WIDTH * 0.8:
                                # 优先选择屏幕左侧或右侧（远离现有平台）
                                if x < WIDTH / 2:
                                    # 当前在左侧，尝试右侧
                                    x = random.randrange(WIDTH // 2, WIDTH - PLATFORM_WIDTH)
                                else:
                                    # 当前在右侧，尝试左侧
                                    x = random.randrange(0, WIDTH // 2)
                                # 重新检查重叠
                                if self.check_excessive_horizontal_overlap(x, y, p_type):
                                    if attempt < max_attempts - 3:
                                        continue  # 继续尝试
            
            # 创建平台
            p = Platform(self, x, y, PLATFORM_WIDTH, PLATFORM_HEIGHT, p_type)
            return p
        
        # 如果所有尝试都失败，生成一个基础可通行平台
        x = random.randrange(0, WIDTH - PLATFORM_WIDTH)
        y = random.randrange(min_y, max_y)
        p = Platform(self, x, y, PLATFORM_WIDTH, PLATFORM_HEIGHT, 'normal')
        return p

    def update(self):
        # 游戏循环 - 更新
        self.all_sprites.update()
        
        # 更新最高高度追踪（Y坐标越小表示越高）
        if self.player.pos.y < self.max_height:
            self.max_height = self.player.pos.y
        
        # 检查玩家是否踩到平台 (只在下落时)
        if self.player.vel.y > 0:
            hits = pygame.sprite.spritecollide(self.player, self.platforms, False)
            if hits:
                # 检查是哪种平台
                if hits[0].type == 'obstacle':
                    # 碰到障碍物，游戏结束
                    self.playing = False
                    self.running = False
                    # !! 触发：播放失败音效 !!
                    if self.enable_sound and self.game_over_sound:
                        try:
                            self.game_over_sound.play()
                        except:
                            pass  # 无头模式下可能无法播放音效
                else:
                    # 踩到平台
                    self.player.pos.y = hits[0].rect.top
                    self.player.vel.y = 0
                    if hits[0].type == 'boost':
                        self.player.vel.y = -PLAYER_JUMP * 1.5 # 强力跳跃
                        # !! 触发：随机播放奖励音效 !!
                        if self.enable_sound and self.reward_sounds:
                            try:
                                random.choice(self.reward_sounds).play()
                            except:
                                pass  # 无头模式下可能无法播放音效
                    else:
                        self.player.jump() # 普通跳跃

        # 屏幕滚动 (当玩家到达屏幕 1/4 高度时)
        if self.player.rect.top <= HEIGHT / 4:
            scroll_dist = abs(self.player.vel.y)
            self.player.pos.y += scroll_dist
            
            # !! 在分数增加后检查里程碑 !!
            old_score = self.score
            self.score += int(scroll_dist) # 分数增加
            
            # !! 触发：检查是否达到分数里程碑 !!
            if old_score < self.next_score_milestone and self.score >= self.next_score_milestone:
                if self.enable_sound and self.milestone_sound:
                    try:
                        self.milestone_sound.play()
                    except:
                        pass  # 无头模式下可能无法播放音效
                self.next_score_milestone += SCORE_MILESTONE_TARGET # 设置下一个目标 (2000, 3000...)
            
            for plat in self.platforms:
                plat.rect.y += scroll_dist
                if plat.rect.top >= HEIGHT:
                    plat.kill() # 移除滚出屏幕的平台

        # 游戏结束 (掉落)
        if self.player.rect.bottom > HEIGHT:
            # 确保失败音效只播一次
            if self.running: 
                # !! 触发：播放失败音效 !!
                if self.enable_sound and self.game_over_sound:
                    try:
                        self.game_over_sound.play()
                    except:
                        pass  # 无头模式下可能无法播放音效
            
            self.playing = False
            self.running = False

        # 生成新平台（基于高度的动态稀疏度）
        target_count = self.get_target_platform_count()
        
        # 检查玩家上方是否有可通行路径
        upward_path_available, safe_count, obstacle_count = self.check_upward_path_available()
        
        # 计算生成区域：主要在屏幕上方预生成，让平台自然进入视野
        if self.platforms:
            # 找到最高的平台（Y坐标最小）
            highest_y = min(plat.rect.y for plat in self.platforms)
            # 上方生成区域（在最高平台上方，预生成区域）
            # 让平台在屏幕外预生成，然后自然进入视野
            above_min_y = highest_y + PLATFORM_GEN_Y_MIN
            above_max_y = highest_y + PLATFORM_GEN_Y_MAX
            
            # 屏幕可见区域（从屏幕顶部到玩家下方一定距离）
            visible_min_y = PLATFORM_VISIBLE_Y_MIN  # 屏幕顶部稍上方
            visible_max_y = self.player.pos.y + HEIGHT * 0.2  # 玩家下方20%屏幕高度
            
            # 合并两个区域，优先在可见区域生成
            min_y = min(above_min_y, visible_min_y)
            max_y = max(above_max_y, visible_max_y)
        else:
            # 初始生成：覆盖整个屏幕高度
            min_y = PLATFORM_VISIBLE_Y_MIN  # 屏幕顶部稍上方
            max_y = HEIGHT - 50  # 屏幕底部稍上方
        
        # 优化5: 确保生成区域在玩家历史最高高度以上
        # max_height是玩家到达的最高位置（Y坐标越小越高）
        # 生成区域的最小Y值应该小于等于max_height（在玩家已到达高度之上）
        if hasattr(self, 'max_height'):
            min_y = min(min_y, self.max_height)  # 确保不在已到达区域下方生成
        
        # 确保生成区域在合理范围内
        if self.platforms:
            min_y = max(int(above_min_y), -300)  # 允许在屏幕上方更远的地方预生成
            max_y = max(int(above_max_y), int(visible_max_y))
            # 再次确保不在已到达区域下方
            if hasattr(self, 'max_height'):
                min_y = min(int(min_y), int(self.max_height))
        else:
            min_y = max(int(min_y), -50)
            max_y = min(int(max_y), HEIGHT - 50)
        
        # 检查屏幕可见区域内的平台数量
        visible_platforms = 0
        for plat in self.platforms:
            if self.is_in_visible_area(plat.rect.y):  # 在屏幕可见范围内
                visible_platforms += 1
        
        # 如果可见区域平台太少，优先在可见区域生成
        min_visible_platforms = max(3, int(target_count * 0.4))  # 至少40%的平台在可见区域
        
        # 找到玩家当前站立的平台（如果有）
        current_platform = None
        if self.player.vel.y >= 0:  # 如果玩家在下落或静止，检查是否在平台上
            for plat in self.platforms:
                # 使用环形距离检查水平位置
                dist_x = self.toroidal_distance(plat.rect.centerx, self.player.pos.x, self.WIDTH)
                if (plat.type != 'obstacle' and 
                    dist_x < PLATFORM_WIDTH and
                    plat.rect.top <= self.player.pos.y <= plat.rect.top + 10):
                    current_platform = plat
                    break
        
        while len(self.platforms) < target_count:
            # 如果上方没有可通行路径，优先在上方区域生成可通行平台
            if not upward_path_available:
                # 在玩家上方区域强制生成可通行平台
                gen_min_y = max(-50, int(self.player.pos.y - HEIGHT * 0.4))
                gen_max_y = int(self.player.pos.y - 20)  # 玩家上方一点
                if gen_min_y >= gen_max_y:
                    gen_max_y = gen_min_y + 50
                # 使用当前平台作为参考（如果存在）
                p = self.generate_platform(gen_min_y, gen_max_y, force_safe=True, preferred_reference=current_platform)
            # 如果可见区域平台不足，优先在可见区域生成
            elif visible_platforms < min_visible_platforms and self.platforms:
                # 在可见区域生成
                gen_min_y = max(-50, int(self.player.pos.y - HEIGHT * 0.2))
                gen_max_y = min(HEIGHT - 50, int(self.player.pos.y + HEIGHT * 0.5))
                p = self.generate_platform(gen_min_y, gen_max_y, preferred_reference=current_platform)
            else:
                # 正常生成（上方预生成区域，让平台自然进入视野）
                gen_min_y = int(min_y)
                gen_max_y = int(max_y)
                p = self.generate_platform(gen_min_y, gen_max_y, preferred_reference=current_platform)
            
            if p:
                self.platforms.add(p)
                self.all_sprites.add(p)
                # 更新可见平台计数
                if self.is_in_visible_area(p.rect.y):
                    visible_platforms += 1
                # 如果生成了可通行平台在上方，更新路径可用性
                if not upward_path_available and p.rect.y < self.player.pos.y and p.type != 'obstacle':
                    upward_path_available = True
            else:
                break  # 如果生成失败，退出循环避免无限循环

    def events_manual(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.playing = False
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.player.jump()

    def draw(self):
        # 如果screen为None（无头模式），跳过绘制
        if self.screen is None:
            return
        self.screen.fill(WHITE)
        self.all_sprites.draw(self.screen)
        score_text = self.font.render(f"Score: {self.score}", True, BLACK)
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()

    def game_step_for_rl(self, action):
        if not self.running:
            return
        self.player.rl_action = action
        # 无头模式下不需要tick时钟
        if self.screen is not None:
            self.clock.tick(FPS) 
        self.update() 
        self.draw() 

    def get_screen_rgb(self):
        """获取屏幕RGB数组（用于录制视频）"""
        if self.screen is None:
            # 无头模式下返回黑色图像
            return np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        return pygame.surfarray.array3d(pygame.display.get_surface())