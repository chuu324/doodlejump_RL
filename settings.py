# 游戏设置
TITLE = "Doodle Jump RL"
WIDTH = 480
HEIGHT = 600
FPS = 60

# 玩家属性（方案A优化：提升水平移动能力，减少跳跃高度）
PLAYER_ACC = 0.65  # 加速度（原0.5，+30%，提升水平移动能力）
PLAYER_FRICTION = -0.12 # 摩擦力（保持不变）
PLAYER_GRAVITY = 0.75    # 重力（原0.8，-6.25%，略微增加下落时间）
PLAYER_JUMP = 18        # 跳跃力度（原20，-10%，减少跳跃高度）

# 平台设置 - 初始平台列表，覆盖整个屏幕高度
# Y坐标：0在顶部，HEIGHT在底部
PLATFORM_LIST = [
    # 底部区域（玩家起始位置附近，Y值较大）
    (0, HEIGHT - 60),                    # 最底部平台 (540)
    (WIDTH / 2 - 50, HEIGHT * 3 / 4),   # 底部偏上 (450)
    (WIDTH - 100, HEIGHT - 150),         # 右侧底部 (450)
    
    # 中部区域
    (125, HEIGHT - 300),                 # 左侧中部 (300)
    (350, HEIGHT - 350),                 # 右侧中部 (250)
    (WIDTH / 2 - 50, HEIGHT - 400),      # 中央中部 (200)
    
    # 上部区域（玩家能看到的上方，Y值较小）
    (175, HEIGHT - 500),                 # 左侧上部 (100)
    (300, HEIGHT - 550),                 # 右侧上部 (50)
    (50, 50),                            # 左上角
    (WIDTH - 100, 30),                   # 右上角
]
PLATFORM_WIDTH = 100
PLATFORM_HEIGHT = 20

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# 贴图路径 
# 玩家 
PLAYER_IMG_PATH = 'assets/player.png'

# 普通平台 
PLATFORM_NORMAL_IMGS = [
    'assets/platform_blue1.png',
    'assets/platform_blue2.png',
    'assets/platform_cloud.png'
]

# 奖励平台 
REWARD_IMGS_LIST = [
    'assets/reward_spring1.png',
    'assets/reward_spring_yellow.png'
]

# 障碍平台 
OBSTACLE_IMGS_LIST = [
    'assets/obstacle_monster1.png',
    'assets/obstacle_monster_broken.png'
]

# 奖励音效 
REWARD_SOUNDS_LIST = [
    'assets/reward1.wav',
    'assets/reward2.wav',
    'assets/powerup.wav'
]

# 游戏结束/失败音效 (单个)
GAME_OVER_SOUND = 'assets/game_over.wav'

# 分数里程碑音效 (单个)
MILESTONE_SOUND = 'assets/milestone.wav'

# 分数里程碑的目标值 (每 1000 分响一次)
SCORE_MILESTONE_TARGET = 1000

# 平台生成设置
MIN_PLATFORMS = 4  # 最小平台数量（保证游戏可玩性）
MAX_PLATFORMS = 8  # 最大平台数量（初始值）
MIN_PLATFORMS_LIMIT = 3  # 高度极限时的最小平台数量

# 平台间距设置
MIN_PLATFORM_SPACING = 30  # 最小垂直间距（像素）
MAX_PLATFORM_SPACING = 120  # 最大垂直间距（像素）
MIN_PLATFORM_SPACING_LIMIT = 80  # 高度极限时的最小间距
MAX_PLATFORM_SPACING_LIMIT = 200  # 高度极限时的最大间距

# 平台间距动态调整参数
SPACING_ADJUST_START = 1000  # 开始调整间距的高度（分数）
SPACING_ADJUST_LIMIT = 10000  # 达到最大间距调整的高度（分数）
MIN_SPACING_ADJUST_FACTOR = 0.7  # 最小间距调整因子（0.7表示间距增加30%，重叠区域减小）
MAX_SPACING_ADJUST_FACTOR = 1.0  # 最大间距调整因子（1.0表示不调整）

# 平台生成高度范围（相对于屏幕顶部）
PLATFORM_GEN_Y_MIN = -150  # 预生成区域：生成平台的最小Y坐标（屏幕上方更远）
PLATFORM_GEN_Y_MAX = -50  # 预生成区域：生成平台的最大Y坐标（屏幕上方）
PLATFORM_VISIBLE_Y_MIN = -50  # 可见区域：屏幕顶部稍上方
PLATFORM_VISIBLE_Y_MAX = HEIGHT + 50  # 可见区域：屏幕底部稍下方

# 高度稀疏度参数
HEIGHT_SPARSE_START = 2000  # 开始稀疏的高度（分数）
HEIGHT_SPARSE_LIMIT = 10000  # 达到极限稀疏的高度（分数）
SPARSE_FACTOR = 0.5  # 稀疏因子（0-1，越小越稀疏）

# 平台类型概率（基础值，会根据高度调整）
BASE_OBSTACLE_PROB = 0.05  # 障碍物初始概率（较低）
BASE_OBSTACLE_PROB_MAX = 0.15  # 障碍物最大概率（高高度时）
BASE_BOOST_PROB = 0.1  # 奖励平台基础概率
BASE_NORMAL_PROB = 0.85  # 普通平台基础概率（调整以匹配新的障碍物概率）

# 障碍物概率增长参数
OBSTACLE_PROB_START = 1000  # 开始增加障碍物概率的高度（分数）
OBSTACLE_PROB_LIMIT = 8000  # 达到最大障碍物概率的高度（分数）

# 可达性检查参数
MAX_JUMP_HEIGHT = PLAYER_JUMP * 2  # 最大跳跃高度估算（像素）
MAX_HORIZONTAL_REACH = WIDTH * 0.6  # 最大水平可达距离（像素）

# 平台水平重叠控制参数
MAX_HORIZONTAL_OVERLAP_RATIO = 0.3  # 最大允许水平重叠比例（30%），只对普通平台和奖励平台
OVERLAP_CHECK_Y_RANGE = 200  # 检查重叠的垂直范围（像素）

# 障碍物安全生成参数
OBSTACLE_SAFE_DISTANCE = 140  # 障碍物与玩家的最小安全距离（像素，原150，-6.7%，因水平移动能力增强）
OBSTACLE_MAX_IN_VISIBLE = 3  # 可见区域内最大障碍物数量
OBSTACLE_MIN_HEIGHT_FOR_MULTIPLE = 3000  # 允许出现多个障碍物的最小高度（分数）