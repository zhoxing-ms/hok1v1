#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
from diy.model.model import Model
from diy.feature.definition import *
import numpy as np
from kaiwu_agent.agent.base_agent import (
    BaseAgent,
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)

from diy.config import Config
from kaiwu_agent.utils.common_func import attached
from diy.feature.reward_manager import GameRewardManager


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.cur_model_name = ""
        self.device = device
        # Create Model and convert the model to achannel-last memory format to achieve better performance.
        # 创建模型, 将模型转换为通道后内存格式，以获得更好的性能。
        self.model = Model().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        # config info
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.cut_points = [value[0] for value in Config.data_shapes]

        # env info
        self.hero_camp = 0
        self.player_id = 0
        self.game_id = None

        # learning info
        self.train_step = 0
        self.initial_lr = Config.INIT_LEARNING_RATE_START
        self.current_lr = self.initial_lr
        parameters = self.model.parameters()
        self.optimizer = torch.optim.Adam(params=parameters, lr=self.initial_lr, betas=(0.9, 0.999), eps=1e-8)
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        
        # 自适应KL散度惩罚系数
        self.kl_coeff = Config.KL_COEFF if hasattr(Config, 'KL_COEFF') else 0.5
        self.kl_target = Config.KL_TARGET if hasattr(Config, 'KL_TARGET') else 0.01
        
        # PPO多轮次学习参数
        self.ppo_epoch = Config.PPO_EPOCH if hasattr(Config, 'PPO_EPOCH') else 4
        self.batch_size = Config.BATCH_SIZE if hasattr(Config, 'BATCH_SIZE') else 1024
        
        # 动态学习率衰减
        self.lr_decay = Config.LR_DECAY if hasattr(Config, 'LR_DECAY') else False
        self.lr_decay_rate = Config.LR_DECAY_RATE if hasattr(Config, 'LR_DECAY_RATE') else 0.9995
        self.min_lr = Config.MIN_LR if hasattr(Config, 'MIN_LR') else 0.00001
        
        # 用于离线训练的经验缓冲区
        self.experience_buffer = []
        self.buffer_size = 10000  # 最大经验缓冲区大小

        # tools
        self.reward_manager = None
        self.logger = logger
        self.monitor = monitor

        super().__init__(agent_type, device, logger, monitor)

    def _model_inference(self, list_obs_data):
        # 使用网络进行推理
        # Using the network for inference
        feature = [obs_data.feature for obs_data in list_obs_data]
        legal_action = [obs_data.legal_action for obs_data in list_obs_data]
        lstm_cell = [obs_data.lstm_cell for obs_data in list_obs_data]
        lstm_hidden = [obs_data.lstm_hidden for obs_data in list_obs_data]

        input_list = [np.array(feature), np.array(lstm_cell), np.array(lstm_hidden)]
        torch_inputs = [torch.from_numpy(nparr).to(torch.float32) for nparr in input_list]
        for i, data in enumerate(torch_inputs):
            data = data.reshape(-1)
            torch_inputs[i] = data.float()

        feature, lstm_cell, lstm_hidden = torch_inputs
        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = lstm_cell.reshape(-1, self.lstm_unit_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.model.set_eval_mode()
        with torch.no_grad():
            output_list = self.model(format_inputs, inference=True)

        np_output = []
        for output in output_list:
            np_output.append(output.numpy())

        logits, value, _lstm_cell, _lstm_hidden = np_output[:4]

        _lstm_cell = _lstm_cell.squeeze(axis=0)
        _lstm_hidden = _lstm_hidden.squeeze(axis=0)

        list_act_data = list()
        for i in range(len(legal_action)):
            prob, action, d_action = self._sample_masked_action(logits[i], legal_action[i])
            list_act_data.append(
                ActData(
                    action=action,
                    d_action=d_action,
                    prob=prob,
                    value=value,
                    lstm_cell=_lstm_cell[i],
                    lstm_hidden=_lstm_hidden[i],
                )
            )
        return list_act_data

    @predict_wrapper
    def predict(self, list_obs_data):
        return self._model_inference(list_obs_data)

    @exploit_wrapper
    def exploit(self, state_dict):
        # Evaluation task will not call agent.reset in the workflow. Users can use the game_id to determine whether a new environment
        # 评估任务不会在workflow中重置agent，用户可以通过game_id判断是否是新的对局，并根据新对局对agent进行重置
        game_id = state_dict["game_id"]
        if self.game_id != game_id:
            player_id = state_dict["player_id"]
            camp = state_dict["player_camp"]
            self.reset(camp, player_id)
            self.game_id = game_id

        # exploit is automatically called when submitting an evaluation task.
        # The parameter is the state_dict returned by env, and it returns the action used by env.step.
        # exploit在提交评估任务时自动调用，参数为env返回的state_dict, 返回env.step使用的action
        obs_data = self.observation_process(state_dict)
        # Call _model_inference for model inference, executing local model inference
        # 模型推理调用_model_inference, 执行本地模型推理
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(state_dict, act_data, False)

    def train_predict(self, state_dict):
        obs_data = self.observation_process(state_dict)
        # Call agent.predict for distributed model inference
        # 调用agent.predict，执行分布式模型推理
        act_data = self.predict([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(state_dict, act_data, True)

    def eval_predict(self, state_dict):
        obs_data = self.observation_process(state_dict)
        # Call agent.predict for distributed model inference
        # 调用agent.predict，执行分布式模型推理
        act_data = self.predict([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(state_dict, act_data, False)

    def action_process(self, state_dict, act_data, is_stochastic):
        if is_stochastic:
            # Use stochastic sampling action
            # 采用随机采样动作 action
            return act_data.action
        else:
            # Use the action with the highest probability
            # 采用最大概率动作 d_action
            return act_data.d_action

    def observation_process(self, state_dict):
        feature_vec, legal_action = (
            state_dict["observation"],
            state_dict["legal_action"],
        )
        return ObsData(
            feature=feature_vec, legal_action=legal_action, lstm_cell=self.lstm_cell, lstm_hidden=self.lstm_hidden
        )

    # 将输入数据转换为批次以供训练
    def _prepare_batches(self, input_datas, batch_size):
        """
        将输入数据划分为多个批次，用于mini-batch训练
        """
        indices = np.arange(input_datas.shape[0])
        np.random.shuffle(indices)
        
        for start_idx in range(0, input_datas.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, input_datas.shape[0])
            batch_indices = indices[start_idx:end_idx]
            batch_input = input_datas[batch_indices]
            yield batch_input
    
    # 计算新旧策略之间的KL散度
    def _compute_kl_divergence(self, old_logits, new_logits, legal_action):
        """
        计算新旧策略之间的KL散度
        """
        old_probs_list = []
        new_probs_list = []
        
        # 分割logits
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        old_logits_split = torch.split(old_logits, label_split_size[:-1], dim=1)
        new_logits_split = torch.split(new_logits, label_split_size[:-1], dim=1)
        legal_actions_split = torch.split(legal_action, label_split_size[:-1], dim=1)
        
        # 计算每个部分的KL散度
        kl_div_sum = 0.0
        for old_l, new_l, la in zip(old_logits_split, new_logits_split, legal_actions_split):
            # 将logits转换为概率分布
            old_policy = self._softmax_with_legal(old_l, la)
            new_policy = self._softmax_with_legal(new_l, la)
            
            # 计算KL散度: sum(p_old * log(p_old / p_new))
            # 添加一个小的常数防止数值不稳定
            ratio = old_policy / (new_policy + 1e-8)
            log_ratio = torch.log(ratio + 1e-8)
            kl = old_policy * log_ratio
            kl_div_sum += kl.sum(dim=1).mean()
            
        return kl_div_sum / len(old_logits_split)
    
    # 将logits和legal_action转换为合法的概率分布
    def _softmax_with_legal(self, logits, legal_action):
        """
        根据legal_action对logits进行mask后计算softmax概率
        """
        # 应用legal_action mask
        masked_logits = logits - 1e20 * (1.0 - legal_action)
        # 计算softmax
        max_logits = torch.max(masked_logits, dim=1, keepdim=True)[0]
        exp_logits = torch.exp(masked_logits - max_logits) * legal_action
        sum_exp_logits = torch.sum(exp_logits, dim=1, keepdim=True)
        probs = exp_logits / (sum_exp_logits + 1e-10)
        return probs
        
    # 动态调整学习率
    def _adjust_learning_rate(self):
        """
        根据训练步数动态调整学习率
        """
        if self.lr_decay:
            self.current_lr = max(self.min_lr, self.initial_lr * (self.lr_decay_rate ** self.train_step))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr
                
    # 自适应调整KL散度惩罚系数
    def _adjust_kl_coeff(self, kl_div):
        """
        根据实际KL散度与目标KL散度的差异动态调整KL惩罚系数
        """
        if kl_div > self.kl_target * 2.0:
            self.kl_coeff *= 1.5
        elif kl_div < self.kl_target / 2.0:
            self.kl_coeff *= 0.5
        # 限制kl_coeff在合理范围内
        self.kl_coeff = min(max(0.05, self.kl_coeff), 5.0)

    @learn_wrapper
    def learn(self, list_sample_data):
        # 将样本添加到经验缓冲区
        if hasattr(Config, 'PPO_EPOCH') and Config.PPO_EPOCH > 1:
            # 将新样本添加到缓冲区
            for sample_data in list_sample_data:
                self.experience_buffer.append(sample_data)
            # 如果缓冲区超过大小限制，移除最旧的样本
            if len(self.experience_buffer) > self.buffer_size:
                excess = len(self.experience_buffer) - self.buffer_size
                self.experience_buffer = self.experience_buffer[excess:]
            
            # 从经验缓冲区中随机采样进行训练
            buffer_size = min(len(self.experience_buffer), self.batch_size * 4)
            sample_indices = np.random.choice(len(self.experience_buffer), buffer_size, replace=False)
            list_npdata = [self.experience_buffer[i].npdata for i in sample_indices]
        else:
            # 常规模式，直接使用传入的样本
            list_npdata = [sample_data.npdata for sample_data in list_sample_data]
            
        _input_datas = np.stack(list_npdata, axis=0)
        _input_datas = torch.from_numpy(_input_datas).to(self.device)
        results = {}

        # 多轮次PPO训练
        n_epochs = self.ppo_epoch if hasattr(Config, 'PPO_EPOCH') else 1
        
        # 动态调整学习率
        self._adjust_learning_rate()
        
        # 存储累计损失
        total_loss_sum = 0
        value_loss_sum = 0
        policy_loss_sum = 0
        entropy_loss_sum = 0
        kl_div_sum = 0
        
        # 如果启用KL惩罚，先获取当前策略下的logits
        if hasattr(Config, 'USE_KL_PENALTY') and Config.USE_KL_PENALTY:
            data_list = list(_input_datas.split(self.cut_points, dim=1))
            for i, data in enumerate(data_list):
                data = data.reshape(-1)
                data_list[i] = data.float()
                
            seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
            feature, legal_action = seri_vec.split(
                [
                    np.prod(self.seri_vec_split_shape[0]),
                    np.prod(self.seri_vec_split_shape[1]),
                ],
                dim=1,
            )
            init_lstm_cell = data_list[-2]
            init_lstm_hidden = data_list[-1]

            feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
            lstm_hidden_state = init_lstm_hidden.reshape(-1, self.lstm_unit_size)
            lstm_cell_state = init_lstm_cell.reshape(-1, self.lstm_unit_size)

            format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

            self.model.set_eval_mode()
            with torch.no_grad():
                old_rst_list = self.model(format_inputs, inference=True)
                old_logits = old_rst_list[0]  # 获取当前策略的logits
        
        # 多轮次训练
        for epoch in range(n_epochs):
            # 将数据分成多个批次进行训练
            for batch_data in self._prepare_batches(_input_datas, self.batch_size):
                data_list = list(batch_data.split(self.cut_points, dim=1))
                for i, data in enumerate(data_list):
                    data = data.reshape(-1)
                    data_list[i] = data.float()

                seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
                feature, legal_action = seri_vec.split(
                    [
                        np.prod(self.seri_vec_split_shape[0]),
                        np.prod(self.seri_vec_split_shape[1]),
                    ],
                    dim=1,
                )
                init_lstm_cell = data_list[-2]
                init_lstm_hidden = data_list[-1]

                feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
                lstm_hidden_state = init_lstm_hidden.reshape(-1, self.lstm_unit_size)
                lstm_cell_state = init_lstm_cell.reshape(-1, self.lstm_unit_size)

                format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

                self.model.set_train_mode()
                self.optimizer.zero_grad()

                rst_list = self.model(format_inputs)
                
                # 计算KL散度惩罚
                kl_div = 0
                kl_loss = 0
                if hasattr(Config, 'USE_KL_PENALTY') and Config.USE_KL_PENALTY:
                    new_logits = rst_list[0]
                    kl_div = self._compute_kl_divergence(old_logits, new_logits, legal_action)
                    kl_loss = self.kl_coeff * kl_div
                    kl_div_sum += kl_div.item()
                
                total_loss, info_list = self.model.compute_loss(data_list, rst_list)
                
                # 如果启用KL惩罚，添加KL损失
                if hasattr(Config, 'USE_KL_PENALTY') and Config.USE_KL_PENALTY:
                    total_loss += kl_loss
                
                total_loss_sum += total_loss.item()
                
                # 收集各类损失
                _, (value_loss, policy_loss, entropy_loss) = info_list
                value_loss_sum += value_loss.item()
                policy_loss_sum += policy_loss.item()
                entropy_loss_sum += entropy_loss.item()

                total_loss.backward()

                # grad clip
                if Config.USE_GRAD_CLIP:
                    torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)

                self.optimizer.step()
            
            # 每个epoch结束后更新KL惩罚系数
            if hasattr(Config, 'USE_KL_PENALTY') and Config.USE_KL_PENALTY and epoch < n_epochs - 1:
                self._adjust_kl_coeff(kl_div_sum / (epoch + 1))
        
        # 计算平均损失
        avg_factor = max(1, n_epochs)
        results["total_loss"] = total_loss_sum / avg_factor
        
        # 更新训练步数
        self.train_step += 1
        
        if self.monitor:
            results["value_loss"] = round(value_loss_sum / avg_factor, 2)
            results["policy_loss"] = round(policy_loss_sum / avg_factor, 2)
            results["entropy_loss"] = round(entropy_loss_sum / avg_factor, 2)
            if hasattr(Config, 'USE_KL_PENALTY') and Config.USE_KL_PENALTY:
                results["kl_div"] = round(kl_div_sum / avg_factor, 4)
                results["kl_coeff"] = round(self.kl_coeff, 4)
            results["learning_rate"] = round(self.current_lr, 6)
            self.monitor.put_data({os.getpid(): results})

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files, and it is important to ensure that
        #  each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files, and it is important to ensure that
        # each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
        else:
            self.model.load_state_dict(
                torch.load(
                    model_file_path,
                    map_location=self.device,
                )
            )
            self.cur_model_name = model_file_path
            self.logger.info(f"load model {model_file_path} successfully")

    def reset(self, hero_camp, player_id):
        self.hero_camp = hero_camp
        self.player_id = player_id
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.reward_manager = GameRewardManager(player_id)

    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data
        self.lstm_cell = act_data.lstm_cell
        self.lstm_hidden = act_data.lstm_hidden

    # get final executable actions
    def _sample_masked_action(self, logits, legal_action):
        """
        Sample actions from predicted logits and legal actions
        return: probability, stochastic and deterministic actions with additional []
        """
        """
        从预测的logits和合法动作中采样动作
        返回：以列表形式概率、随机和确定性动作
        """

        prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])
        for index in range(0, len(self.label_size_list) - 1):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list += list(probs)
            sample_action = self._legal_sample(probs, use_max=False)
            action_list.append(sample_action)
            d_action = self._legal_sample(probs, use_max=True)
            d_action_list.append(d_action)

        # deals with the last prediction, target
        # 处理最后的预测，目标
        index = len(self.label_size_list) - 1
        target_legal_action_o = np.reshape(
            legal_actions[index],  # [12, 8]
            [
                self.legal_action_size[0],
                self.legal_action_size[-1] // self.legal_action_size[0],
            ],
        )
        one_hot_actions = np.eye(self.label_size_list[0])[action_list[0]]  # [12]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])  # [12, 1]
        target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        legal_actions[index] = target_legal_action  # [12]
        probs = self._legal_soft_max(logits_split[-1], target_legal_action)
        prob_list += list(probs)
        sample_action = self._legal_sample(probs, use_max=False)
        action_list.append(sample_action)

        # target_legal_action = tf.gather(target_legal_action, action_idx, axis=1)
        one_hot_actions = np.eye(self.label_size_list[0])[d_action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])
        target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        # legal_actions[index] = target_legal_action
        probs = self._legal_soft_max(logits_split[-1], target_legal_action_d)

        # prob_list.append(probs)
        d_action = self._legal_sample(probs, use_max=True)
        d_action_list.append(d_action)

        return [prob_list], action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        _lsm_const_w, _lsm_const_e = 1e20, 1e-5
        _lsm_const_e = 0.00001

        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        # Not necessary max clip 1
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        # tmp = tf.exp(tmp - tmp_max)* legal_action + _lsm_const_e
        tmp = (np.exp(tmp) + _lsm_const_e) * legal_action
        # tmp_sum = tf.reduce_sum(tmp, axis=1, keepdims=True)
        probs = tmp / np.sum(tmp, keepdims=True)
        return probs

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        # Sample with probability, input probs should be 1D array
        # 根据概率采样，输入的probs应该是一维数组
        if use_max:
            return np.argmax(probs)

        # 在随机采样时添加温度参数，控制探索
        # 随着训练步数增加，降低温度参数，减少随机性
        temperature = max(0.5, 1.0 - 0.0001 * self.train_step)  # 温度从1.0慢慢降到0.5
        if temperature != 1.0:
            # 应用温度缩放
            probs = np.power(probs, 1.0 / temperature)
            probs = probs / np.sum(probs)  # 重新归一化

        return np.argmax(np.random.multinomial(1, probs, size=1))
