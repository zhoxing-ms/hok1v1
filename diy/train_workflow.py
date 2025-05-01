#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import os
import time
import random
import numpy as np
from diy.feature.definition import (
    sample_process,
    build_frame,
    lineup_iterator_roundrobin_camp_heroes,
    FrameCollector,
    NONE_ACTION,
)
from kaiwu_agent.utils.common_func import attached
from diy.config import GameConfig, Config
from tools.model_pool_utils import get_valid_model_pool


@attached
def workflow(envs, agents, logger=None, monitor=None):
    # Whether the agent is training, corresponding to do_predicts
    # 智能体是否进行训练
    do_learns = [True, True]
    last_save_model_time = time.time()
    
    # 添加训练相关计数器和追踪变量
    episode_count = 0
    win_count = 0
    train_win_rate = 0.0
    eval_win_rate = 0.0
    last_eval_time = time.time()
    last_win_rate_update_time = time.time()
    eval_results = []
    
    # 获取有效的模型池
    model_pool = get_valid_model_pool(logger)
    if not model_pool:
        model_pool = []
    
    # 初始化模型选择概率，优先选择更强的模型作为对手
    model_selection_probs = initialize_model_selection_probs(model_pool)
    
    # 用于自动保存不同阶段的模型
    model_save_milestones = [100, 500, 1000, 2000, 5000]

    while True:
        for g_data in run_episodes(envs, agents, logger, monitor, model_selection_probs):
            # 更新训练计数器和胜率
            episode_result = g_data.get('episode_result', {})
            if episode_result:
                episode_count += 1
                if episode_result.get('win', False):
                    win_count += 1
                
                # 每100局更新一次胜率
                if episode_count % 100 == 0:
                    train_win_rate = win_count / 100
                    win_count = 0
                    logger.info(f"Episode {episode_count}, training win rate: {train_win_rate:.2f}")
                    
                    # 根据胜率动态调整模型选择概率
                    if train_win_rate > 0.7:
                        # 胜率过高，增加更强对手的选择概率
                        update_model_selection_probs(model_selection_probs, stronger=True)
                    elif train_win_rate < 0.3:
                        # 胜率过低，增加更弱对手的选择概率
                        update_model_selection_probs(model_selection_probs, stronger=False)
                
                # 如果是评估局，记录结果
                if episode_result.get('is_eval', False):
                    eval_results.append(episode_result.get('win', False))
                    if len(eval_results) >= 20:
                        eval_win_rate = sum(eval_results) / len(eval_results)
                        logger.info(f"Evaluation win rate (last 20 games): {eval_win_rate:.2f}")
                        eval_results = []
                
                # 保存训练里程碑模型
                if episode_count in model_save_milestones:
                    milestone_id = f"milestone_{episode_count}"
                    agents[0].save_model(id=milestone_id)
                    logger.info(f"Saved milestone model at episode {episode_count}")
            
            for index, (d_learn, agent) in enumerate(zip(do_learns, agents)):
                if d_learn and 'samples' in g_data and len(g_data['samples'][index]) > 0:
                    # The learner trains in a while true loop, here learn actually sends samples
                    # learner 采用 while true 训练，此处 learn 实际为发送样本
                    agent.learn(g_data['samples'][index])
            
            if 'samples' in g_data:
                g_data['samples'].clear()

            now = time.time()
            if now - last_save_model_time > GameConfig.MODEL_SAVE_INTERVAL:
                agents[0].save_model()
                last_save_model_time = now


# 初始化模型选择概率分布
def initialize_model_selection_probs(model_pool):
    probs = {
        'selfplay': 0.6,    # 自我对弈概率
        'common_ai': 0.1,   # 规则AI对弈概率
        'models': {}        # 各历史模型的选择概率
    }
    
    # 如果模型池非空，分配剩余概率给历史模型
    if model_pool:
        remaining_prob = 0.3
        model_count = len(model_pool)
        
        # 较新的模型获得更高的概率（假设模型ID越大越新）
        sorted_models = sorted([int(m) for m in model_pool])
        for i, model_id in enumerate(sorted_models):
            # 使用递增权重，越新的模型权重越高
            weight = (i + 1) / sum(range(1, model_count + 1))
            probs['models'][str(model_id)] = remaining_prob * weight
    
    return probs

# 动态更新模型选择概率
def update_model_selection_probs(probs, stronger=True):
    if stronger:
        # 增加更强对手的概率
        probs['selfplay'] = max(0.4, probs['selfplay'] - 0.05)
        probs['common_ai'] = max(0.05, probs['common_ai'] - 0.05)
        
        # 增加较新模型的概率
        if probs['models']:
            model_ids = sorted([int(m) for m in probs['models'].keys()])
            newest_models = [str(m) for m in model_ids[-2:] if m in model_ids]  # 最新的两个模型
            
            for model_id in newest_models:
                probs['models'][model_id] = min(0.3, probs['models'][model_id] + 0.05)
    else:
        # 增加更弱对手的概率
        probs['selfplay'] = min(0.8, probs['selfplay'] + 0.05)
        probs['common_ai'] = min(0.2, probs['common_ai'] + 0.05)
        
        # 减少较新模型的概率
        if probs['models']:
            model_ids = sorted([int(m) for m in probs['models'].keys()])
            newest_models = [str(m) for m in model_ids[-2:] if m in model_ids]
            
            for model_id in newest_models:
                probs['models'][model_id] = max(0.05, probs['models'][model_id] - 0.03)
    
    # 确保所有概率总和为1
    model_prob_sum = sum(probs['models'].values()) if probs['models'] else 0
    total = probs['selfplay'] + probs['common_ai'] + model_prob_sum
    
    # 归一化
    if total != 1.0:
        factor = 1.0 / total
        probs['selfplay'] *= factor
        probs['common_ai'] *= factor
        for model_id in probs['models']:
            probs['models'][model_id] *= factor
    
    return probs

# 根据概率分布选择对手类型
def select_opponent_type(probs):
    r = random.random()
    cumulative_prob = 0
    
    # 检查是否选择自我对弈
    cumulative_prob += probs['selfplay']
    if r < cumulative_prob:
        return "selfplay"
    
    # 检查是否选择规则AI
    cumulative_prob += probs['common_ai']
    if r < cumulative_prob:
        return "common_ai"
    
    # 否则选择历史模型
    if probs['models']:
        model_probs = list(probs['models'].items())
        model_ids = [m[0] for m in model_probs]
        model_weights = [m[1] for m in model_probs]
        
        # 归一化模型权重
        weight_sum = sum(model_weights)
        norm_weights = [w / weight_sum for w in model_weights]
        
        return np.random.choice(model_ids, p=norm_weights)
    
    # 默认返回自我对弈
    return "selfplay"


def run_episodes(envs, agents, logger, monitor, model_selection_probs=None):
    # hok1v1 environment
    # hok1v1环境
    env = envs[0]
    # Number of agents, in hok1v1 the value is 2
    # 智能体数量，在hok1v1中值为2
    agent_num = len(agents)
    # Episode counter
    # 对局数量计数器
    episode_cnt = 0
    # ID of Agent to training
    # 每一局要训练的智能体的id
    train_agent_id = 0
    # Lineup iterator
    # 阵容生成器
    lineup_iter = lineup_iterator_roundrobin_camp_heroes(camp_heroes=GameConfig.CAMP_HEROES)
    # Frame Collector
    # 帧收集器
    frame_collector = FrameCollector(agent_num)
    # Make eval matches as evenly distributed as possible
    # 引入随机因子，让eval对局尽可能平均分布
    random_eval_start = random.randint(0, GameConfig.EVAL_FREQ)
    
    # 对手模型选择策略
    if model_selection_probs is None:
        model_selection_probs = {
            'selfplay': 0.7,
            'common_ai': 0.2,
            'models': {
                # 如果没有历史模型，则此部分为空
            }
        }

    # Single environment process (30 frame/s)
    # 单局流程 (30 frame/s)
    while True:
        # Settings before starting a new environment
        # 以下是启动一个新对局前的设置

        # Set the id of the agent to be trained. id=0 means the blue side, id=1 means the red side.
        # 设置要训练的智能体的id，id=0表示蓝方，id=1表示红方，每一局都切换一次阵营。默认对手智能体是selfplay即自己
        train_agent_id = 1 - train_agent_id
        
        # 默认对手智能体是selfplay
        opponent_agent = "selfplay"

        # Evaluate at a certain frequency during training to reflect the improvement of the agent during training
        # 智能体支持边训练边评估，训练中按一定的频率进行评估，反映智能体在训练中的水平
        is_eval = (episode_cnt + random_eval_start) % GameConfig.EVAL_FREQ == 0
        if is_eval:
            # 评估模式，使用固定的对手
            opponent_agent = "common_ai"
        else:
            # 训练模式，根据概率选择对手
            opponent_agent = select_opponent_type(model_selection_probs)

        # Generate a new set of agent configurations
        # 生成一组新的智能体配置
        heroes_config = next(lineup_iter)

        usr_conf = {
            "diy": {
                # The side reporting the environment metrics
                # 上报对局指标的阵营
                "monitor_side": train_agent_id,
                # The label for reporting environment metrics: selfplay - "selfplay", common_ai - "common_ai", opponent model - model_id
                # 上报对局指标的标签： 自对弈 - "selfplay", common_ai - "common_ai", 对手模型 - model_id
                "monitor_label": opponent_agent,
                # Indicates the lineups used by both sides
                # 表示双方使用的阵容
                "lineups": heroes_config,
            }
        }

        if train_agent_id not in [0, 1]:
            raise Exception("monitor_side is not valid, valid monitor_side list is [0, 1], please check")

        # Start a new environment
        # 启动新对局，返回初始环境状态
        _, state_dicts = env.reset(usr_conf=usr_conf)
        if state_dicts is None:
            logger.info(f"episode {episode_cnt}, reset is None happened!")
            continue

        # Game variables
        # 对局变量
        episode_cnt += 1
        frame_no = 0
        step = 0
        # Record the cumulative rewards of the agent in the environment
        # 记录对局中智能体的累积回报，用于上报监控
        total_reward_dicts = [{}, {}]
        logger.info(f"Episode {episode_cnt} start, usr_conf is {usr_conf}")

        # Reset agent
        # 重置agent

        # The 'do_predicts' specifies which agents are to perform model predictions.
        # Since the default opponent model is 'selfplay', it is set to [True, True] by default.
        # do_predicts指定哪些智能体要进行模型预测，由于默认对手模型是selfplay，默认设置[True, True]
        do_predicts = [True, True]
        for i, agent in enumerate(agents):
            player_id = state_dicts[i]["player_id"]
            camp = state_dicts[i]["player_camp"]
            agent.reset(camp, player_id)

            # The agent to be trained should load the latest model
            # 要训练的智能体应加载最新的模型
            if i == train_agent_id:
                # train_agent_id uses the latest model
                # train_agent_id 使用最新模型
                agent.load_model(id="latest")
            else:
                if opponent_agent == "common_ai":
                    # common_ai does not need to load a model, no need to predict
                    # 如果对手是 common_ai 则不需要加载模型, 也不需要进行预测
                    do_predicts[i] = False
                elif opponent_agent == "selfplay":
                    # Training model, "latest" - latest model, "random" - random model from the model pool
                    # 加载训练过的模型，可以选择最新模型，也可以选择随机模型 "latest" - 最新模型, "random" - 模型池中随机模型
                    agent.load_model(id="latest")
                else:
                    # Opponent model, model_id is checked from kaiwu.json
                    # 选择kaiwu.json中设置的对手模型, model_id 即 opponent_agent，必须设置正确否则报错
                    eval_candidate_model = get_valid_model_pool(logger)
                    if int(opponent_agent) not in eval_candidate_model:
                        raise Exception(f"model_id {opponent_agent} not in {eval_candidate_model}")
                    else:
                        agent.load_model(id=opponent_agent)

            logger.info(f"agent_{i} reset playerid:{player_id} camp:{camp}")

        # Reward initialization
        # 回报初始化，作为当前环境状态state_dicts的一部分
        for i in range(agent_num):
            reward = agents[i].reward_manager.result(state_dicts[i]["frame_state"])
            state_dicts[i]["reward"] = reward
            for key, value in reward.items():
                if key in total_reward_dicts[i]:
                    total_reward_dicts[i][key] += value
                else:
                    total_reward_dicts[i][key] = value

        # Reset environment frame collector
        # 重置环境帧收集器
        frame_collector.reset(num_agents=agent_num)
        
        # 记录对局开始时间以及是否胜利
        game_start_time = time.time()
        is_win = False

        while True:
            # Initialize the default actions. If the agent does not make a decision, env.step uses the default action.
            # 初始化默认的actions，如果智能体不进行决策，则env.step使用默认action
            actions = [
                NONE_ACTION,
            ] * agent_num

            for index, (d_predict, agent) in enumerate(zip(do_predicts, agents)):
                if d_predict:
                    if not is_eval:
                        actions[index] = agent.train_predict(state_dicts[index])
                    else:
                        actions[index] = agent.eval_predict(state_dicts[index])

                    # Only when do_predict=True and is_eval=False, the agent's environment data is saved.
                    # 仅do_predict=True且is_eval=False时，智能体的对局数据保存。即评估对局数据不训练，不是最新模型产生的数据不训练
                    if not is_eval and index == train_agent_id:
                        frame = build_frame(agent, state_dicts[index])
                        frame_collector.save_frame(frame, agent_id=index)

            """
            The format of action is like [[2, 10, 1, 14, 8, 0], [1, 3, 10, 10, 9, 0]]
            There are 2 agents, so the length of the array is 2, and the order of values in
            each element is: button, move (2), skill (2), target
            action格式形如[[2, 10, 1, 14, 8, 0], [1, 3, 10, 10, 9, 0]]
            2个agent, 故数组的长度为2, 每个元素里面的值的顺序是:button, move(2个), skill(2个), target
            """

            # Step forward
            # 推进环境到下一帧，得到新的状态
            frame_no, _, _, terminated, truncated, state_dicts = env.step(actions)

            # Disaster recovery
            # 容灾
            if state_dicts is None:
                logger.info(f"episode {episode_cnt}, step({step}) is None happened!")
                break

            # Reward generation
            # 计算回报，作为当前环境状态state_dicts的一部分
            for i in range(agent_num):
                reward = agents[i].reward_manager.result(state_dicts[i]["frame_state"])
                state_dicts[i]["reward"] = reward
                for key, value in reward.items():
                    if key in total_reward_dicts[i]:
                        total_reward_dicts[i][key] += value
                    else:
                        total_reward_dicts[i][key] = value

            step += 1
            
            # 检查是否胜利（实时更新胜负状态）
            if train_agent_id < len(state_dicts) and "termination_info" in state_dicts[train_agent_id]:
                term_info = state_dicts[train_agent_id]["termination_info"]
                if term_info.get("win", False):
                    is_win = True

            # Normal end or timeout exit
            # 正常结束或超时退出
            if terminated or truncated:
                # 计算对局时长
                game_duration = time.time() - game_start_time
                
                logger.info(
                    f"episode_{episode_cnt} terminated in fno_{frame_no}, time:{game_duration:.1f}s, truncated:{truncated}, eval:{is_eval}, total_reward_dicts:{total_reward_dicts}"
                )
                # 记录更详细的胜负信息
                if train_agent_id < len(state_dicts) and "termination_info" in state_dicts[train_agent_id]:
                    term_info = state_dicts[train_agent_id]["termination_info"]
                    is_win = term_info.get("win", False)
                    logger.info(
                        f"Episode {episode_cnt} result: {'WIN' if is_win else 'LOSE'}, opponent: {opponent_agent}"
                    )
                
                # Reward for saving the last state of the environment
                # 保存环境最后状态的reward
                for index, (d_predict, agent) in enumerate(zip(do_predicts, agents)):
                    if d_predict and not is_eval and index == train_agent_id:
                        frame_collector.save_last_frame(
                            agent_id=index,
                            reward=state_dicts[index]["reward"]["reward_sum"],
                        )

                monitor_data = {
                    "reward": round(total_reward_dicts[train_agent_id]["reward_sum"], 2),
                    "diy_1": round(total_reward_dicts[train_agent_id]["forward"], 2),
                    "game_duration": round(game_duration, 2),
                    "win": 1 if is_win else 0,
                }

                if monitor and is_eval:
                    monitor.put_data({os.getpid(): monitor_data})

                # Sample process
                # 进行样本处理，准备训练
                if len(frame_collector) > 0 and not is_eval:
                    list_agents_samples = sample_process(frame_collector)
                    
                    # 返回样本和对局结果
                    yield {
                        'samples': list_agents_samples,
                        'episode_result': {
                            'win': is_win,
                            'duration': game_duration,
                            'is_eval': is_eval,
                            'opponent': opponent_agent
                        }
                    }
                else:
                    # 返回仅包含对局结果的字典
                    yield {
                        'episode_result': {
                            'win': is_win, 
                            'duration': game_duration,
                            'is_eval': is_eval,
                            'opponent': opponent_agent
                        }
                    }
                break
