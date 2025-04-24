#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class GameConfig:

    """
    Specify the training lineup in CAMP_HEROES. The battle lineup will be paired in all possible combinations.
    To train a single agent, comment out the other agents.
    1. 133 DiRenjie
    2. 199 Arli
    3. 508 Garo
    """

    """
    在CAMP_HEROES中指定训练阵容, 对战阵容会两两组合, 训练单智能体则注释其他智能体。此配置会在阵容生成器中使用
    1. 133 狄仁杰
    2. 199 公孙离
    3. 508 伽罗
    """
    CAMP_HEROES = [
        [{"hero_id": 133}],
        [{"hero_id": 199}],
        [{"hero_id": 508}],
    ]
    # Set the weight of each reward item and use it in reward_manager
    # 设置各个回报项的权重，在reward_manager中使用
    REWARD_WEIGHT_DICT = {
        "hp_point": 2.0,        # 保持不变，生命值对智能体生存重要
        "tower_hp_point": 5.0,  # 保持不变，更加强调防御塔血量重要性
        "money": 0.006,         # 保持不变，鼓励获取资源
        "exp": 0.006,           # 保持不变，鼓励获取经验
        "ep_rate": 0.75,        # 保持不变，鼓励技能释放
        "death": -1.0,          # 保持不变，避免死亡惩罚
        "kill": 0.5,            # 改为正向奖励，鼓励击杀敌方
        "last_hit": 0.5,        # 提高补刀奖励
        "forward": 0.01,        # 保持不变，鼓励更主动的进攻
    }
    # Time decay factor, used in reward_manager
    # 时间衰减因子，在reward_manager中使用
    TIME_SCALE_ARG = 3000       # 增加时间衰减因子，鼓励智能体更快取得胜利
    # Evaluation frequency and model save interval configuration, used in workflow
    # 评估频率和模型保存间隔配置，在workflow中使用
    EVAL_FREQ = 10
    MODEL_SAVE_INTERVAL = 1800


# Dimension configuration, used when building the model
# 维度配置，构建模型时使用
class DimConfig:
    # main camp soldier
    DIM_OF_SOLDIER_1_10 = [18, 18, 18, 18]
    # enemy camp soldier
    DIM_OF_SOLDIER_11_20 = [18, 18, 18, 18]
    # main camp organ
    DIM_OF_ORGAN_1_2 = [18, 18]
    # enemy camp organ
    DIM_OF_ORGAN_3_4 = [18, 18]
    # main camp hero
    DIM_OF_HERO_FRD = [235]
    # enemy camp hero
    DIM_OF_HERO_EMY = [235]
    # public hero info
    DIM_OF_HERO_MAIN = [14]
    # global info
    DIM_OF_GLOBAL_INFO = [25]


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DATA_SPLIT_SHAPE = [
        810,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        512,
        512,
    ]
    SERI_VEC_SPLIT_SHAPE = [(725,), (85,)]
    INIT_LEARNING_RATE_START = 0.00015  # 增加初始学习率，加速初期收敛
    BETA_START = 0.03                  # 增加熵正则化系数，鼓励探索
    LOG_EPSILON = 1e-6
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [
        True,
        True,
        True,
        True,
        True,
        True,
    ]  # means each task whether need reinforce

    CLIP_PARAM = 0.15  # 降低裁剪参数，使策略更新更稳定

    MIN_POLICY = 0.00001

    TARGET_EMBED_DIM = 32

    data_shapes = [
        [(725 + 85) * 16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [192],
        [256],
        [256],
        [256],
        [256],
        [144],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [512],
        [512],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.998        # 增加折扣因子，更加重视长期回报
    LAMDA = 0.97         # 增加GAE系数，让优势估计更平滑

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # 以下是优化PPO算法的新增参数
    # PPO优化参数
    PPO_EPOCH = 4        # PPO每批数据的训练轮数
    BATCH_SIZE = 1024    # 批次大小
    
    # 自适应KL惩罚参数
    USE_KL_PENALTY = True    # 启用KL散度惩罚
    KL_TARGET = 0.01         # 目标KL散度
    KL_COEFF = 0.5           # KL惩罚系数初始值
    
    # 动态学习率调整
    LR_DECAY = True          # 启用学习率衰减
    LR_DECAY_RATE = 0.9995   # 学习率衰减率
    MIN_LR = 0.00001         # 最小学习率

    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # For instance, the dimension for ppo is 15584,
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样, 比如示例代码中ppo的维度是15584
    # **注意**，此项必须正确配置，应该与definition.py中的NumpyData2SampleData函数数据对齐，否则可能报样本维度错误
    SAMPLE_DIM = 15584
