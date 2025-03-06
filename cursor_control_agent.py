import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Tuple
import random
from collections import deque

# 导入我们的文本编辑环境
from text_editing_environment import TextEditingEnv
import re
import time

class CursorControlHead(nn.Module):
    """
    光标控制头，用于预测在给定文本状态下应该执行的操作
    
    输入:
    - 文本状态的嵌入表示
    - 光标位置信息
    - 选择区域信息
    - 剪贴板内容
    - 当前按键状态
    
    输出:
    - 动作概率分布
    """
    
    def __init__(self, text_embedding_dim, hidden_dim, n_actions):
        super(CursorControlHead, self).__init__()
        
        self.text_embedding_dim = text_embedding_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        
        # 文本处理网络
        self.text_rnn = nn.GRU(
            input_size=text_embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        
        # 位置编码
        self.position_embedding = nn.Linear(2, hidden_dim)
        
        # 选区编码
        self.selection_embedding = nn.Linear(2, hidden_dim)
        
        # 剪贴板处理网络
        self.clipboard_rnn = nn.GRU(
            input_size=text_embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # 按键状态编码
        self.key_embedding = nn.Linear(3, hidden_dim)
        
        # 组合所有特征 - 计算正确的输入维度
        # bidirectional GRU输出维度为hidden_dim*2
        # 其他特征维度均为hidden_dim
        total_feature_dim = (hidden_dim * 2) + (hidden_dim * 4)
        
        self.fc_combine = nn.Linear(total_feature_dim, hidden_dim)
        
        # 动作预测层
        self.fc_policy = nn.Linear(hidden_dim, n_actions)
        self.fc_value = nn.Linear(hidden_dim, 1)
    
    def forward(self, 
                text_features, 
                cursor_pos, 
                selection, 
                clipboard_features, 
                key_pressed):
        """前向传播"""
        # 处理文本特征
        text_output, _ = self.text_rnn(text_features)
        text_repr = text_output.mean(dim=1)  # 池化操作获得整体文本表示
        
        # 处理光标位置
        cursor_repr = self.position_embedding(cursor_pos)
        
        # 处理选区位置
        selection_repr = self.selection_embedding(selection)
        
        # 处理剪贴板内容
        clipboard_output, _ = self.clipboard_rnn(clipboard_features)
        clipboard_repr = clipboard_output.mean(dim=1)
        
        # 处理按键状态
        key_repr = self.key_embedding(key_pressed)
        
        # 调试维度信息
        # print(f"text_repr: {text_repr.shape}")
        # print(f"cursor_repr: {cursor_repr.shape}")
        # print(f"selection_repr: {selection_repr.shape}")
        # print(f"clipboard_repr: {clipboard_repr.shape}")
        # print(f"key_repr: {key_repr.shape}")
        
        # 组合所有特征
        combined = torch.cat([
            text_repr, 
            cursor_repr, 
            selection_repr,
            clipboard_repr,
            key_repr
        ], dim=1)
        
        # print(f"combined shape: {combined.shape}")
        
        combined_repr = F.relu(self.fc_combine(combined))
        
        # 预测动作分布和状态值
        action_logits = self.fc_policy(combined_repr)
        state_value = self.fc_value(combined_repr)
        
        return action_logits, state_value


class TextEncoder:
    """
    将文本转换为模型可处理的特征表示
    
    简化版本：使用字符级别的one-hot编码
    实际应用中可以替换为预训练语言模型的嵌入
    """
    
    def __init__(self, vocab_size=128, embedding_dim=64):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.char_to_index = {chr(i): i for i in range(vocab_size)}
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    def encode_text(self, text, max_length=1000):
        """将文本编码为模型输入"""
        # 字符级索引
        indices = []
        for char in text[:max_length]:
            if ord(char) < self.vocab_size:
                indices.append(self.char_to_index.get(char, 0))
            else:
                indices.append(0)  # 未知字符用0表示
        
        # 填充到固定长度
        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))
        
        # 转换为张量
        return torch.tensor(indices, dtype=torch.long)
    
    def get_embeddings(self, text_indices):
        """获取文本的嵌入表示"""
        return self.embedding(text_indices)


class PPOAgent:
    """
    基于PPO算法的强化学习代理，负责学习如何操控光标进行文本编辑
    
    这是一个简化版本，实际实现中需要更多的优化和细节处理
    """
    
    def __init__(self, env, text_embedding_dim=64, hidden_dim=128, gamma=0.99, clip_ratio=0.2, lr=3e-4, gpt2_path=None):
        """
        初始化PPO代理
        
        参数:
        env: 环境实例
        text_embedding_dim: 文本嵌入维度
        hidden_dim: 隐藏层维度
        gamma: 折扣因子
        clip_ratio: PPO裁剪比例
        lr: 学习率
        gpt2_path: GPT-2模型路径，如果提供则使用GPT-2进行文本编码
        """
        self.env = env
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.n_actions = env.action_space.n
        self.use_gpt2 = gpt2_path is not None
        
        # 初始化编码器（可选使用GPT-2）
        if self.use_gpt2:
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                from transformers import GPT2Tokenizer, GPT2Model
                self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path, device=device)
                
                # 设置pad token为eos token（GPT-2默认没有pad token）
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.gpt2_model = GPT2Model.from_pretrained(gpt2_path, device)
                print(f"GPT-2模型成功加载: {gpt2_path}")
                # GPT-2嵌入维度是768
                actual_embedding_dim = 768
            except Exception as e:
                print(f"加载GPT-2模型失败: {e}")
                print("回退到使用基本文本编码器")
                self.use_gpt2 = False
                actual_embedding_dim = text_embedding_dim
                self.text_encoder = TextEncoder(vocab_size=128, embedding_dim=text_embedding_dim)
        else:
            # 使用简单编码器
            self.text_encoder = TextEncoder(vocab_size=128, embedding_dim=text_embedding_dim)
            actual_embedding_dim = text_embedding_dim
        
        # 确保隐藏维度与文本嵌入维度兼容
        if self.use_gpt2 and text_embedding_dim != 768:
            print(f"警告: GPT-2的嵌入维度为768，但提供的text_embedding_dim为{text_embedding_dim}")
            print("将使用768作为实际嵌入维度")
        
        # 初始化策略网络（光标控制头）
        self.policy = CursorControlHead(
            text_embedding_dim=actual_embedding_dim,
            hidden_dim=hidden_dim,
            n_actions=self.n_actions
        )
        
        # 优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 经验回放缓冲区
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': [],
            'values': []
        }
        
        # 记录额外参数
        self.hidden_dim = hidden_dim
        self.text_embedding_dim = actual_embedding_dim
        self.max_text_length = 1000  # 最大文本长度，可以根据需要调整
    
    def _preprocess_observation(self, obs):
        """将环境观察转换为模型输入"""
        # 获取文本
        text = obs['text']
        
        # 根据是否使用GPT-2选择不同的编码方式
        if hasattr(self, 'use_gpt2') and self.use_gpt2:
            # 使用GPT-2编码文本
            with torch.no_grad():
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    max_length=self.max_text_length, 
                    truncation=True,
                    padding="max_length"
                )
                # 获取GPT-2的隐藏状态
                outputs = self.gpt2_model(**inputs)
                text_features = outputs.last_hidden_state
        else:
            # 使用基本文本编码器
            text_indices = self.text_encoder.encode_text(text)
            text_features = self.text_encoder.get_embeddings(text_indices)
            text_features = text_features.unsqueeze(0)  # 添加批处理维度
        
        # 归一化光标位置
        cursor_pos = torch.tensor(obs['cursor_position'], dtype=torch.float32).unsqueeze(0)
        cursor_pos = cursor_pos / 1000.0  # 简单归一化
        
        # 归一化选择区域
        selection = torch.tensor(obs['selection'], dtype=torch.float32).unsqueeze(0)
        selection = selection / 1000.0  # 简单归一化
        
        # 编码剪贴板文本
        clipboard = obs['clipboard']
        
        # 同样根据是否使用GPT-2选择不同的剪贴板编码方式
        if hasattr(self, 'use_gpt2') and self.use_gpt2:
            # 剪贴板可能为空，需要特殊处理
            if not clipboard:
                # 使用占位符或创建零张量与text_features形状匹配
                # 方法1: 使用占位符代替空剪贴板
                clipboard = "<empty_clipboard>"
            
            # 使用GPT-2编码剪贴板
            with torch.no_grad():
                clipboard_inputs = self.tokenizer(
                    clipboard, 
                    return_tensors="pt", 
                    max_length=self.max_text_length, 
                    truncation=True,
                    padding="max_length"
                )
                clipboard_outputs = self.gpt2_model(**clipboard_inputs)
                clipboard_features = clipboard_outputs.last_hidden_state
        else:
            # 使用基本编码器
            clipboard_indices = self.text_encoder.encode_text(clipboard)
            clipboard_features = self.text_encoder.get_embeddings(clipboard_indices)
            clipboard_features = clipboard_features.unsqueeze(0)  # 添加批处理维度
        
        # 按键状态
        key_pressed = torch.tensor(obs['key_pressed'], dtype=torch.float32).unsqueeze(0)
        
        return {
            'text_features': text_features,
            'cursor_pos': cursor_pos,
            'selection': selection,
            'clipboard_features': clipboard_features,
            'key_pressed': key_pressed
        }
    
    def select_action(self, obs, training=True):
        """根据当前观察选择动作"""
        processed_obs = self._preprocess_observation(obs)
        
        # 前向传播获取动作概率分布和状态值
        with torch.no_grad():
            action_logits, state_value = self.policy(
                processed_obs['text_features'],
                processed_obs['cursor_pos'],
                processed_obs['selection'],
                processed_obs['clipboard_features'],
                processed_obs['key_pressed']
            )
        
        # 从概率分布中采样动作
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        if training:
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item(), state_value.item()
        else:
            # 在评估模式下，选择概率最高的动作
            action = torch.argmax(action_probs)
            return action.item()
    
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """存储一个转换样本到缓冲区"""
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['next_states'].append(next_state)
        self.buffer['dones'].append(done)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['values'].append(value)
    
    def compute_returns(self, final_value=0):
        """计算每个时间步的回报"""
        returns = []
        R = final_value
        
        # 从后往前计算折扣回报
        for i in reversed(range(len(self.buffer['rewards']))):
            R = self.buffer['rewards'][i] + self.gamma * R * (1 - self.buffer['dones'][i])
            returns.insert(0, R)
        
        return returns
    
    def update_policy(self, batch_size=64, epochs=10):
        """使用PPO算法更新策略网络"""
        # 计算折扣回报
        returns = self.compute_returns()
        
        # 转换为张量
        states = self.buffer['states']
        actions = torch.tensor(self.buffer['actions'], dtype=torch.long)
        old_log_probs = torch.tensor(self.buffer['log_probs'], dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # 计算优势函数
        values = torch.tensor(self.buffer['values'], dtype=torch.float32)
        advantages = returns - values
        
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 执行多个训练周期
        for _ in range(epochs):
            # 创建小批量
            indices = np.random.permutation(len(returns))
            
            for start_idx in range(0, len(returns), batch_size):
                # 获取小批量索引
                idx = indices[start_idx:start_idx + batch_size]
                
                # 处理小批量数据
                batch_obs_list = [self._preprocess_observation(states[i]) for i in idx]
                
                # 合并批处理
                batch_text_features = torch.cat([obs['text_features'] for obs in batch_obs_list], dim=0)
                batch_cursor_pos = torch.cat([obs['cursor_pos'] for obs in batch_obs_list], dim=0)
                batch_selection = torch.cat([obs['selection'] for obs in batch_obs_list], dim=0)
                batch_clipboard_features = torch.cat([obs['clipboard_features'] for obs in batch_obs_list], dim=0)
                batch_key_pressed = torch.cat([obs['key_pressed'] for obs in batch_obs_list], dim=0)
                
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                
                # 前向传播
                logits, values = self.policy(
                    batch_text_features,
                    batch_cursor_pos,
                    batch_selection,
                    batch_clipboard_features,
                    batch_key_pressed
                )
                
                # 新的动作概率分布
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                
                # 新的对数概率
                new_log_probs = dist.log_prob(batch_actions)
                
                # 策略比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 裁剪目标函数
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                
                # 策略损失
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 值函数损失
                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)
                
                # 熵奖励（鼓励探索）
                entropy = dist.entropy().mean()
                
                # 总损失
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # 清空缓冲区
        for key in self.buffer.keys():
            self.buffer[key] = []
    
    def train(self, num_episodes=1000, max_steps=100, update_frequency=20, render=False):
        """训练代理"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            # 执行一个回合
            while not done and step < max_steps:
                # 选择动作
                action, log_prob, value = self.select_action(obs)
                
                # 执行动作
                next_obs, reward, done, _ = self.env.step(action)
                
                # 存储转换
                self.store_transition(obs, action, reward, next_obs, done, log_prob, value)
                
                # 更新状态
                obs = next_obs
                episode_reward += reward
                step += 1
                
                # 可选渲染
                if render:
                    self.env.render()
            
            episode_rewards.append(episode_reward)
            
            # 周期性更新策略
            if (episode + 1) % update_frequency == 0:
                self.update_policy()
                print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {np.mean(episode_rewards[-update_frequency:]):.2f}")
        
        return episode_rewards
    
    def test(self, num_episodes=10, max_steps=100, render=True):
        """测试代理性能"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < max_steps:
                # 评估模式选择动作
                action = self.select_action(obs, training=False)
                
                # 执行动作
                next_obs, reward, done, _ = self.env.step(action)
                
                # 更新状态
                obs = next_obs
                episode_reward += reward
                step += 1
                
                # 可选渲染
                if render:
                    self.env.render()
                    
            episode_rewards.append(episode_reward)
            print(f"Test Episode {episode + 1}, Reward: {episode_reward:.2f}")
        
        print(f"Average Test Reward: {np.mean(episode_rewards):.2f}")
        return episode_rewards