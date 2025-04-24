#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
Enhanced network implementation for PPO in 1v1 scenarios
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional, Union
from ppo.config import Config

# 注意力机制模块，用于处理序列数据之间的依赖关系
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        多头注意力机制实现
        
        参数:
            d_model: 输入特征维度
            num_heads: 注意力头数量
            dropout: Dropout概率
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model必须能被num_heads整除"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # 投影并分割多头
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 输出投影
        output = self.out_proj(context)
        
        return output

# 残差连接和层归一化
class ResidualLayerNorm(nn.Module):
    def __init__(self, size, dropout=0.1):
        """
        残差连接和层归一化模块
        
        参数:
            size: 特征维度
            dropout: Dropout概率
        """
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """
        应用残差连接和层归一化
        
        参数:
            x: 输入特征
            sublayer: 要应用的子层函数
        """
        return x + self.dropout(sublayer(self.norm(x)))

# 位置编码器，用于为序列添加位置信息
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        """
        位置编码器，为序列添加位置信息
        
        参数:
            d_model: 特征维度
            max_len: 最大序列长度
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        添加位置编码到输入特征
        
        参数:
            x: 输入特征 [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1)]

# 前馈神经网络模块
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        前馈神经网络模块
        
        参数:
            d_model: 输入特征维度
            d_ff: 隐藏层维度
            dropout: Dropout概率
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        """
        应用前馈神经网络
        
        参数:
            x: 输入特征
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# 增强型LSTM模块，包含注意力机制
class EnhancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, bidirectional=False, attention=True):
        """
        增强型LSTM模块，可选添加注意力机制
        
        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏状态维度
            num_layers: LSTM层数
            dropout: Dropout概率
            bidirectional: 是否使用双向LSTM
            attention: 是否添加注意力机制
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attention = attention
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # 注意力机制
        if attention:
            attn_dim = hidden_size * 2 if bidirectional else hidden_size
            self.attention_layer = MultiHeadAttention(attn_dim, num_heads=4, dropout=dropout)
            self.residual_norm = ResidualLayerNorm(attn_dim, dropout=dropout)
    
    def forward(self, x, hidden=None):
        """
        前向传播
        
        参数:
            x: 输入特征 [batch_size, seq_len, input_size]
            hidden: 初始隐藏状态 (h_0, c_0)
        
        返回:
            output: LSTM输出
            hidden: 最终隐藏状态
        """
        # LSTM前向传播
        output, hidden = self.lstm(x, hidden)
        
        # 如果启用注意力机制
        if self.attention:
            # 应用自注意力
            def _attention_sublayer(x):
                return self.attention_layer(x, x, x)
            
            output = self.residual_norm(output, _attention_sublayer)
        
        return output, hidden

# 主网络
class EnhancedNetwork(nn.Module):
    def __init__(self):
        """
        增强型PPO网络，结合LSTM和注意力机制
        """
        super().__init__()
        
        # 配置
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.label_size_list = Config.LABEL_SIZE_LIST
        
        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(725, 512),  # 输入特征维度
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # LSTM层
        self.lstm = EnhancedLSTM(
            input_size=256, 
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            attention=True
        )
        
        # 动作头
        self.action_heads = nn.ModuleList()
        for size in self.label_size_list:
            self.action_heads.append(nn.Linear(self.lstm_unit_size, size))
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(self.lstm_unit_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # 共享特征提取器
        self.shared_feature_extractor = nn.Sequential(
            nn.Linear(self.lstm_unit_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 助手网络，用于预测敌人动作，提高模型的对抗性
        self.enemy_action_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, sum(self.label_size_list))
        )
    
    def forward(self, inputs, lstm_state=None, inference=False):
        """
        前向传播
        
        参数:
            inputs: 输入特征列表
            lstm_state: LSTM状态 (h_0, c_0)
            inference: 是否为推理模式
        
        返回:
            output_list: 输出列表 [logits, value, lstm_cell, lstm_hidden]
        """
        # 分解输入
        feature, lstm_hidden, lstm_cell = inputs
        
        # 特征编码
        encoded_features = self.feature_encoder(feature)
        
        # 准备LSTM状态
        hidden = (
            lstm_hidden.unsqueeze(0).contiguous(),  # hidden state
            lstm_cell.unsqueeze(0).contiguous()     # cell state
        )
        
        # LSTM前向传播
        lstm_out, (new_hidden, new_cell) = self.lstm(encoded_features.unsqueeze(1), hidden)
        lstm_out = lstm_out.squeeze(1)
        
        # 提取最终隐藏状态
        new_hidden = new_hidden.squeeze(0)
        new_cell = new_cell.squeeze(0)
        
        # 共享特征提取
        shared_features = self.shared_feature_extractor(lstm_out)
        
        # 多任务输出头
        logits_list = []
        for action_head in self.action_heads:
            logits = action_head(lstm_out)
            logits_list.append(logits)
        
        # 合并logits
        logits = torch.cat(logits_list, dim=1)
        
        # 价值预测
        value = self.value_head(lstm_out)
        
        # 辅助任务：预测敌人动作（训练时使用）
        if not inference:
            enemy_action_pred = self.enemy_action_predictor(shared_features)
        else:
            enemy_action_pred = None
        
        # 返回输出
        output_list = [logits, value, new_cell, new_hidden]
        
        if not inference and enemy_action_pred is not None:
            output_list.append(enemy_action_pred)
        
        return output_list

# 创建网络实例
def create_enhanced_network():
    """
    创建增强型PPO网络实例
    
    返回:
        network: 增强型网络实例
    """
    return EnhancedNetwork() 