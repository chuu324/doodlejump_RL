import pygame
import random
from settings import *
vec = pygame.math.Vector2 # 2D向量

class Player(pygame.sprite.Sprite):
    def __init__(self, game):
        super().__init__()
        self.game = game
        
        # 使用游戏对象中已加载的图片，如果不存在则使用默认图形
        if hasattr(game, 'player_image') and game.player_image is not None:
            self.image = game.player_image
        else:
            # 默认使用一个绿色的方块
            self.image = pygame.Surface((30, 40))
            self.image.fill(GREEN)
            
        self.rect = self.image.get_rect()
        self.rect.center = (WIDTH / 2, HEIGHT / 2)
        
        self.pos = vec(WIDTH / 2, HEIGHT / 2)
        self.vel = vec(0, 0)
        self.acc = vec(0, 0)

    def jump(self):
        # 只有在踩到平台上时才能跳
        self.rect.y += 2
        hits = pygame.sprite.spritecollide(self, self.game.platforms, False)
        self.rect.y -= 2
        if hits:
            self.vel.y = -PLAYER_JUMP

    def update(self):
        # 应用重力
        self.acc = vec(0, PLAYER_GRAVITY)
        
        # RL Agent 控制的移动（优先）
        if hasattr(self, 'rl_action'):
            if self.rl_action == 0: # 向左
                self.acc.x = -PLAYER_ACC
            elif self.rl_action == 2: # 向右
                self.acc.x = PLAYER_ACC
            # rl_action == 1 (不动) 则 self.acc.x 默认为 0 (受摩擦力影响)
        else:
            # 手动控制（仅在非RL模式且video系统初始化时）
            try:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    self.acc.x = -PLAYER_ACC
                if keys[pygame.K_RIGHT]:
                    self.acc.x = PLAYER_ACC
            except pygame.error:
                # 无头模式下，video系统未初始化，跳过键盘输入
                pass

        # 应用摩擦力
        self.acc.x += self.vel.x * PLAYER_FRICTION
        # 运动学公式
        self.vel += self.acc
        self.pos += self.vel + 0.5 * self.acc
        
        # 环绕屏幕
        if self.pos.x > WIDTH:
            self.pos.x = 0
        if self.pos.x < 0:
            self.pos.x = WIDTH
            
        self.rect.midbottom = self.pos

class Platform(pygame.sprite.Sprite):
    def __init__(self, game, x, y, w, h, p_type='normal'):
        super().__init__()
        self.game = game 
        self.type = p_type

        image_to_use = None
        fill_color = BLUE 
        
        if self.type == 'normal':
            if self.game.platform_normal_images:
                image_to_use = random.choice(self.game.platform_normal_images)
            else:
                fill_color = BLUE
        
        elif self.type == 'boost': # 奖励
            if self.game.reward_images: # <-- 使用复数列表
                image_to_use = random.choice(self.game.reward_images) # <-- 随机选择
            else:
                fill_color = (150, 255, 150) # 浅绿

        elif self.type == 'obstacle': # 障碍
            if self.game.obstacle_images: # <-- 使用复数列表
                image_to_use = random.choice(self.game.obstacle_images) # <-- 随机选择
            else:
                fill_color = RED
        
        if image_to_use:
            self.image = image_to_use
        else:
            self.image = pygame.Surface((w, h))
            self.image.fill(fill_color)
            
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y