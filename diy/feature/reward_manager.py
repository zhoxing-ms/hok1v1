#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import math
from ppo.config import GameConfig

# Used to record various reward information
# 用于记录各个奖励信息
class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


# Used to initialize various reward information
# 用于初始化各个奖励信息
def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.main_hero_hp = -1
        self.main_hero_organ_hp = -1
        self.m_reward_value = {}
        self.m_last_frame_no = -1
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.m_init_calc_frame_map = {}
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_main_hero_config_id = -1
        self.m_each_level_max_exp = {}
        # 新增奖励相关变量
        self.last_enemy_tower_hp = -1  # 记录上一帧敌方防御塔血量
        self.tower_damage_done = 0     # 累计对防御塔造成的伤害
        self.progress_stages = {}      # 记录进度阶段奖励是否已发放
        self.consecutive_attacks = 0   # 连续攻击防御塔次数
        self.last_position = None      # 上一帧位置

    # Used to initialize the maximum experience value for each agent level
    # 用于初始化智能体各个等级的最大经验值
    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        self.m_each_level_max_exp[1] = 160
        self.m_each_level_max_exp[2] = 298
        self.m_each_level_max_exp[3] = 446
        self.m_each_level_max_exp[4] = 524
        self.m_each_level_max_exp[5] = 613
        self.m_each_level_max_exp[6] = 713
        self.m_each_level_max_exp[7] = 825
        self.m_each_level_max_exp[8] = 950
        self.m_each_level_max_exp[9] = 1088
        self.m_each_level_max_exp[10] = 1240
        self.m_each_level_max_exp[11] = 1406
        self.m_each_level_max_exp[12] = 1585
        self.m_each_level_max_exp[13] = 1778
        self.m_each_level_max_exp[14] = 1984

    def result(self, frame_data):
        self.init_max_exp_of_each_hero()
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)

        # 添加新的奖励项：防御塔伤害奖励和进度奖励
        self.calculate_tower_damage_reward(frame_data)
        self.calculate_progress_reward(frame_data)
        
        frame_no = frame_data["frameNo"]
        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                if key not in ["tower_damage", "progress", "positional"]:  # 不对新增奖励项应用时间衰减
                    self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

        return self.m_reward_value

    # 计算对防御塔造成伤害的奖励
    def calculate_tower_damage_reward(self, frame_data):
        # 获取敌方防御塔
        enemy_tower = None
        enemy_camp = -1
        if self.main_hero_camp != -1:
            enemy_camp = 1 if self.main_hero_camp == 0 else 0
            
        for npc in frame_data["npc_states"]:
            if npc["camp"] == enemy_camp and npc["sub_type"] == "ACTOR_SUB_TOWER":
                enemy_tower = npc
                break
                
        if enemy_tower is not None:
            current_tower_hp = enemy_tower["hp"]
            
            # 初始化上一帧防御塔血量
            if self.last_enemy_tower_hp == -1:
                self.last_enemy_tower_hp = current_tower_hp
            
            # 计算本帧造成的伤害
            damage_done = max(0, self.last_enemy_tower_hp - current_tower_hp)
            
            # 更新总伤害
            self.tower_damage_done += damage_done
            
            # 根据伤害计算奖励
            if damage_done > 0:
                # 增加连续攻击计数并给予额外奖励
                self.consecutive_attacks += 1
                combo_multiplier = min(2.0, 1.0 + (self.consecutive_attacks * 0.1))  # 最高2倍奖励
                tower_damage_reward = damage_done * 0.01 * combo_multiplier
                
                # 对塔血量降到特定阈值给予额外奖励
                hp_percent = current_tower_hp / enemy_tower["max_hp"]
                if hp_percent < 0.5 and "half_tower" not in self.progress_stages:
                    tower_damage_reward += 1.0
                    self.progress_stages["half_tower"] = True
                if hp_percent < 0.25 and "quarter_tower" not in self.progress_stages:
                    tower_damage_reward += 2.0
                    self.progress_stages["quarter_tower"] = True
            else:
                # 如果没有造成伤害，重置连击计数
                self.consecutive_attacks = max(0, self.consecutive_attacks - 0.5)
                tower_damage_reward = 0
                
            # 更新上一帧防御塔血量
            self.last_enemy_tower_hp = current_tower_hp
            
            # 将塔伤害奖励添加到奖励字典
            self.m_reward_value["tower_damage"] = tower_damage_reward
        else:
            self.m_reward_value["tower_damage"] = 0

    # 计算游戏进度奖励，鼓励智能体向着目标前进
    def calculate_progress_reward(self, frame_data):
        main_hero = None
        enemy_tower = None
        
        # 获取主英雄和敌方防御塔
        for hero in frame_data["hero_states"]:
            if hero["player_id"] == self.main_hero_player_id:
                main_hero = hero
                break
                
        enemy_camp = 1 if self.main_hero_camp == 0 else 0
        for npc in frame_data["npc_states"]:
            if npc["camp"] == enemy_camp and npc["sub_type"] == "ACTOR_SUB_TOWER":
                enemy_tower = npc
                break
                
        if main_hero is not None and enemy_tower is not None:
            # 计算英雄与敌方防御塔的距离
            hero_pos = (main_hero["actor_state"]["location"]["x"], main_hero["actor_state"]["location"]["z"])
            tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
            
            distance = math.dist(hero_pos, tower_pos)
            
            # 基于距离的位置奖励
            # 距离越近，奖励越高
            max_distance = 100  # 假设地图最大距离为100
            normalized_distance = max(0, 1.0 - (distance / max_distance))
            positional_reward = normalized_distance * 0.02
            
            # 根据英雄朝向防御塔的方向移动给予额外奖励
            if self.last_position is not None:
                last_distance = math.dist(self.last_position, tower_pos)
                # 如果距离变小，说明朝着防御塔方向移动
                if distance < last_distance:
                    positional_reward += 0.01
                    
                # 如果非常接近防御塔（在攻击范围内）
                if distance < 10 and "near_tower" not in self.progress_stages:
                    positional_reward += 0.5
                    self.progress_stages["near_tower"] = True
            
            self.last_position = hero_pos
            
            # 将位置奖励添加到奖励字典
            self.m_reward_value["positional"] = positional_reward
        else:
            self.m_reward_value["positional"] = 0

    # Calculate the value of each reward item in each frame
    # 计算每帧的每个奖励子项的信息
    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):

        # Get both agents
        # 获取双方智能体
        main_hero, enemy_hero = None, None
        hero_list = frame_data["hero_states"]
        for hero in hero_list:
            hero_camp = hero["actor_state"]["camp"]
            if hero_camp == camp:
                main_hero = hero
            else:
                enemy_hero = hero
        main_hero_hp = main_hero["actor_state"]["hp"]
        main_hero_max_hp = main_hero["actor_state"]["max_hp"]
        main_hero_ep = main_hero["actor_state"]["values"]["ep"]
        main_hero_max_ep = main_hero["actor_state"]["values"]["max_ep"]

        # Get both defense towers
        # 获取双方防御塔
        main_tower, main_spring, enemy_tower, enemy_spring = None, None, None, None
        npc_list = frame_data["npc_states"]
        for organ in npc_list:
            organ_camp = organ["camp"]
            organ_subtype = organ["sub_type"]
            if organ_camp == camp:
                if organ_subtype == "ACTOR_SUB_TOWER":  # 21 is ACTOR_SUB_TOWER, normal tower
                    main_tower = organ
                elif organ_subtype == "ACTOR_SUB_CRYSTAL":  # 24 is ACTOR_SUB_CRYSTAL, base crystal
                    main_spring = organ
            else:
                if organ_subtype == "ACTOR_SUB_TOWER":  # 21 is ACTOR_SUB_TOWER, normal tower
                    enemy_tower = organ
                elif organ_subtype == "ACTOR_SUB_CRYSTAL":  # 24 is ACTOR_SUB_CRYSTAL, base crystal
                    enemy_spring = organ

        for reward_name, reward_struct in cul_calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value
            # Money
            # 金钱
            if reward_name == "money":
                reward_struct.cur_frame_value = main_hero["moneyCnt"]
            # Health points
            # 生命值
            elif reward_name == "hp_point":
                reward_struct.cur_frame_value = math.sqrt(math.sqrt(1.0 * main_hero_hp / main_hero_max_hp))
            # Energy points
            # 法力值
            elif reward_name == "ep_rate":
                if main_hero_max_ep == 0 or main_hero_hp <= 0:
                    reward_struct.cur_frame_value = 0
                else:
                    reward_struct.cur_frame_value = main_hero_ep / float(main_hero_max_ep)
            # Kills
            # 击杀
            elif reward_name == "kill":
                reward_struct.cur_frame_value = main_hero["killCnt"]
            # Deaths
            # 死亡
            elif reward_name == "death":
                reward_struct.cur_frame_value = main_hero["deadCnt"]
            # Tower health points
            # 塔血量
            elif reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = 1.0 * main_tower["hp"] / main_tower["max_hp"]
            # Last hit
            # 补刀
            elif reward_name == "last_hit":
                reward_struct.cur_frame_value = 0.0
                frame_action = frame_data["frame_action"]
                if "dead_action" in frame_action:
                    dead_actions = frame_action["dead_action"]
                    for dead_action in dead_actions:
                        if (
                            dead_action["killer"]["runtime_id"] == main_hero["actor_state"]["runtime_id"]
                            and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
                        ):
                            # 增加补刀奖励
                            reward_struct.cur_frame_value += 1.0
                        elif (
                            dead_action["killer"]["runtime_id"] == enemy_hero["actor_state"]["runtime_id"]
                            and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
                        ):
                            reward_struct.cur_frame_value -= 1.0
            # Experience points
            # 经验值
            elif reward_name == "exp":
                reward_struct.cur_frame_value = self.calculate_exp_sum(main_hero)
            # Forward
            # 前进
            elif reward_name == "forward":
                # 改进前进奖励计算，更好地鼓励向敌方防御塔移动
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)

    # Calculate the total amount of experience gained using agent level and current experience value
    # 用智能体等级和当前经验值，计算获得经验值的总量
    def calculate_exp_sum(self, this_hero_info):
        exp_sum = 0.0
        for i in range(1, this_hero_info["level"]):
            exp_sum += self.m_each_level_max_exp[i]
        exp_sum += this_hero_info["exp"]
        return exp_sum

    # Calculate the forward reward based on the distance between the agent and both defensive towers
    # 用智能体到双方防御塔的距离，计算前进奖励
    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        hero_pos = (
            main_hero["actor_state"]["location"]["x"],
            main_hero["actor_state"]["location"]["z"],
        )
        
        # 计算智能体到敌方防御塔的距离
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        # 计算我方防御塔到敌方防御塔的距离
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        
        forward_value = 0
        # 修改前进奖励计算逻辑
        hero_hp_ratio = main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"]
        
        # 在生命值较高时鼓励更积极的推进
        if hero_hp_ratio > 0.7:
            # 距离敌方防御塔越近，奖励越高
            forward_value = (dist_main2emy - dist_hero2emy) / dist_main2emy
            # 增加指数放大效果，使得靠近敌塔时奖励更明显
            forward_value = math.pow(max(0, forward_value), 0.8) * 1.5
        # 生命值适中时保持适度推进
        elif hero_hp_ratio > 0.4:
            forward_value = (dist_main2emy - dist_hero2emy) / dist_main2emy
            forward_value = max(0, forward_value)
        # 生命值过低时，减少激进推进的奖励
        else:
            # 低血量时仍然鼓励适度推进，但强度降低
            forward_value = (dist_main2emy - dist_hero2emy) / dist_main2emy * 0.5
            forward_value = max(0, forward_value)
            
        return forward_value

    # Calculate the reward item information for both sides using frame data
    # 用帧数据来计算两边的奖励子项信息
    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1

        for hero in frame_data["hero_states"]:
            if hero["player_id"] == self.main_hero_player_id:
                main_camp = hero["actor_state"]["camp"]
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero["actor_state"]["camp"]
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    # Use the values obtained in each frame to calculate the corresponding reward value
    # 用每一帧得到的奖励子项信息来计算对应的奖励值
    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        reward_sum, weight_sum = 0.0, 0.0
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            if reward_name == "hp_point":
                if (
                    self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0
                    and self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0
                ):
                    reward_struct.cur_frame_value = 0
                    reward_struct.last_frame_value = 0
                elif self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0:
                    reward_struct.cur_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    reward_struct.last_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                elif self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0:
                    reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - 0
                    reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - 0
                else:
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            elif reward_name == "ep_rate":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
                if reward_struct.last_frame_value > 0:
                    # 更改EP计算逻辑，鼓励技能使用
                    ep_change = reward_struct.cur_frame_value - reward_struct.last_frame_value
                    if ep_change < 0:  # EP减少，表示使用了技能
                        reward_struct.value = abs(ep_change)  # 使用技能获得正向奖励
                    else:
                        reward_struct.value = 0  # EP增加或不变，不给奖励
                else:
                    reward_struct.value = 0
            elif reward_name == "exp":
                main_hero = None
                for hero in frame_data["hero_states"]:
                    if hero["player_id"] == self.main_hero_player_id:
                        main_hero = hero
                if main_hero and main_hero["level"] >= 15:
                    reward_struct.value = 0
                else:
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
                    # 对经验值奖励根据英雄等级进行加权
                    level_factor = 1.0
                    if main_hero:
                        level = main_hero["level"]
                        # 低等级时经验值更重要
                        level_factor = max(0.5, 2.0 - (level * 0.1))
                    reward_struct.value = (reward_struct.cur_frame_value - reward_struct.last_frame_value) * level_factor
            elif reward_name == "forward":
                # 前进奖励直接使用当前值而不是差值
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "last_hit":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "tower_hp_point":
                # 防御塔血量采用差值计算，更敏感地反映血量变化
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                hp_change = reward_struct.cur_frame_value - reward_struct.last_frame_value
                # 加大对防御塔血量变化的敏感度
                if hp_change < 0:  # 我方防御塔血量下降或敌方防御塔血量上升
                    reward_struct.value = hp_change * 2.0  # 加大惩罚力度
                else:  # 我方防御塔血量上升或敌方防御塔血量下降
                    reward_struct.value = hp_change * 3.0  # 更大的奖励
            else:
                # Calculate zero-sum reward
                # 计算零和奖励
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            weight_sum += reward_struct.weight
            reward_sum += reward_struct.value * reward_struct.weight
            reward_dict[reward_name] = reward_struct.value
        reward_dict["reward_sum"] = reward_sum
