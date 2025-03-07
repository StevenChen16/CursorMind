import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import deque
import random
import time
import math

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def transpose_for_scores(self, x):
        batch_size = x.size(0)
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, query, key, value, attention_mask=None):
        batch_size = query.size(0)
        
        # 线性变换
        query_layer = self.transpose_for_scores(self.query(query))
        key_layer = self.transpose_for_scores(self.key(key))
        value_layer = self.transpose_for_scores(self.value(value))
        
        # 注意力计算
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # 应用注意力掩码（如果有）
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # 注意力权重
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # 计算加权和
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.d_model,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # 输出投影
        output = self.out_proj(context_layer)
        
        return output


class CursorControlHead(nn.Module):
    """
    光标控制头部网络，使用注意力机制输出动作分布
    """
    def __init__(self, input_dim, hidden_dim, num_actions=12, num_heads=4):
        super(CursorControlHead, self).__init__()
        
        # 确保hidden_dim可以被注意力头数整除
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # 直接处理整个输入向量
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 多头自注意力层
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # 输出头
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 投影到隐藏维度
        hidden = self.input_proj(x)
        
        # 把特征展开为序列形式以便注意力处理 (batch_size, seq_len=1, hidden_dim)
        hidden = hidden.unsqueeze(1)
        
        # 自注意力
        attn_output, _ = self.self_attention(hidden, hidden, hidden)
        hidden = self.layer_norm1(hidden + attn_output)
        
        # 前馈网络
        ff_output = self.feed_forward(hidden)
        hidden = self.layer_norm2(hidden + ff_output)
        
        # 挤压序列维度 (batch_size, hidden_dim)
        hidden = hidden.squeeze(1)
        
        # 输出层
        action_logits = self.action_head(hidden)
        
        # 应用稳定的softmax
        action_probs = F.softmax(action_logits, dim=-1)
        
        # 确保概率之和为1，处理数值精度问题
        probs_sum = torch.sum(action_probs, dim=-1, keepdim=True)
        action_probs = action_probs / probs_sum
        
        # 状态值估计
        state_values = self.value_head(hidden)
        
        return action_probs, state_values


class GPT2TextEncoder:
    """
    使用预训练的GPT-2模型将文本转换为特征表示，并提取注意力信息
    """
    
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.device = next(model.parameters()).device
    
    def encode_text(self, text, max_length=1024):
        """将文本编码为模型输入"""
        # 使用tokenizer将文本转换为token IDs
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=max_length, 
            truncation=True
        )
        
        # 将输入移至相同设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def get_embeddings(self, inputs=None, text=None, pooling="mean", output_attentions=False):
        """
        获取文本的嵌入表示和注意力信息
        
        参数:
        - inputs: 已经通过encode_text处理的输入
        - text: 原始文本，如果inputs为None则使用
        - pooling: 池化方法，可选"mean"、"last"或"cls"
        - output_attentions: 是否输出注意力权重
        
        返回:
        - 文本的嵌入向量
        - 注意力权重（如果output_attentions=True）
        """
        if inputs is None and text is not None:
            inputs = self.encode_text(text)
        
        if inputs is None:
            raise ValueError("必须提供inputs或text参数之一")
        
        # 获取GPT-2的隐藏状态和注意力
        with torch.no_grad():
            outputs = self.model(
                **inputs, 
                output_hidden_states=True,
                output_attentions=output_attentions
            )
        
        # 获取最后一层的隐藏状态
        hidden_states = outputs.hidden_states[-1]
        
        # 根据池化方法选择不同的表示方式
        if pooling == "mean":
            # 平均池化，忽略padding tokens
            mask = inputs["attention_mask"].unsqueeze(-1)
            embeddings = torch.sum(hidden_states * mask, dim=1) / torch.sum(mask, dim=1)
        elif pooling == "last":
            # 使用每个序列的最后一个非padding token
            seq_lengths = torch.sum(inputs["attention_mask"], dim=1) - 1
            batch_size = hidden_states.shape[0]
            embeddings = hidden_states[torch.arange(batch_size), seq_lengths]
        elif pooling == "cls":
            # 使用第一个token
            embeddings = hidden_states[:, 0, :]
        else:
            raise ValueError(f"不支持的池化方法: {pooling}")
        
        # 如果请求了注意力权重，则返回
        if output_attentions:
            return embeddings, outputs.attentions
        
        return embeddings


class TextEncoder:
    """
    将文本转换为模型可处理的特征表示
    
    简化版本：使用字符级别的one-hot编码
    实际应用中可以替换为预训练语言模型的嵌入
    """
    
    def __init__(self, vocab_size=128, embedding_dim=64, device=None):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.char_to_index = {chr(i): i for i in range(vocab_size)}
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = self.embedding.to(self.device)
    
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
        
        # 转换为张量并移到设备上
        return torch.tensor(indices, dtype=torch.long).to(self.device)
    
    def get_embeddings(self, text_indices):
        """获取文本的嵌入表示"""
        if text_indices.device != self.device:
            text_indices = text_indices.to(self.device)
        return self.embedding(text_indices)


class PPOAgent:
    """使用PPO算法的文本编辑代理，集成注意力机制"""
    
    def __init__(self, env, text_embedding_dim=64, hidden_dim=128, gamma=0.99, 
                 clip_ratio=0.2, lr=3e-4, use_gpt2=True, gpt2_path=None, num_heads=4):
        self.env = env
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 状态和动作空间
        self.num_actions = env.action_space.n
        
        # 确定是否使用GPT-2作为文本编码器
        self.use_gpt2 = use_gpt2 and gpt2_path is not None
        
        if self.use_gpt2:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(gpt2_path)
            
            # 设置pad token为eos token（GPT-2默认没有pad token）
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_path)
            self.gpt2_model.to(self.device)
            self.gpt2_model.eval()
            self.text_encoder = GPT2TextEncoder(self.tokenizer, self.gpt2_model)
            print(f"GPT-2模型成功加载: {gpt2_path}")
        else:
            self.text_encoder = TextEncoder(vocab_size=128, embedding_dim=text_embedding_dim, device=self.device)
            text_embedding_dim = self.text_encoder.embedding_dim
        
        # 策略网络：文本嵌入 + 光标位置 + 当前任务 -> 动作分布
        input_dim = text_embedding_dim + 2 + 3  # 文本嵌入 + 光标(x,y) + 任务类型(one-hot)
        
        # 使用新的基于注意力的CursorControlHead
        self.policy = CursorControlHead(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            num_actions=self.num_actions,
            num_heads=num_heads
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # 经验回放缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.action_probs = []
        self.values = []
        self.dones = []
        self.attention_maps = []  # 存储注意力图
    
    def _process_observation(self, obs):
        """处理观察，转换为模型输入，现在包括注意力信息"""
        # 处理文本
        if isinstance(obs, dict) and "text" in obs:
            text = obs["text"]
        else:
            text = str(obs)
            
        # 处理光标位置
        if isinstance(obs, dict):
            if "cursor_x" in obs and "cursor_y" in obs:
                cursor_x = obs["cursor_x"]
                cursor_y = obs["cursor_y"]
            elif "cursor_pos" in obs:
                cursor_pos = obs["cursor_pos"]
                if isinstance(cursor_pos, (list, tuple)) and len(cursor_pos) >= 2:
                    cursor_x, cursor_y = cursor_pos[0], cursor_pos[1]
                else:
                    cursor_x, cursor_y = 0, 0
            elif "cursor" in obs:
                cursor = obs["cursor"]
                if isinstance(cursor, (list, tuple)) and len(cursor) >= 2:
                    cursor_x, cursor_y = cursor[0], cursor[1]
                else:
                    cursor_x, cursor_y = 0, 0
            else:
                cursor_x, cursor_y = 0, 0
        else:
            cursor_x, cursor_y = 0, 0
            
        # 任务类型
        if isinstance(obs, dict):
            task_type = obs.get("task_type", [0,0,0])
        else:
            task_type = [0,0,0]
        
        # 文本编码 - 现在获取注意力信息
        if self.use_gpt2:
            with torch.no_grad():
                if hasattr(self, 'collect_attention_maps') and self.collect_attention_maps:
                    text_embedding, attention_maps = self.text_encoder.get_embeddings(
                        text=text, output_attentions=True
                    )
                    # 存储注意力图以供可视化或分析
                    self.attention_maps.append(attention_maps)
                else:
                    text_embedding = self.text_encoder.get_embeddings(text=text)
        else:
            text_indices = self.text_encoder.encode_text(text)
            text_embedding = self.text_encoder.get_embeddings(text_indices)
            text_embedding = torch.mean(text_embedding, dim=0, keepdim=True)
        
        # 合并特征
        cursor_tensor = torch.tensor([cursor_x, cursor_y], dtype=torch.float32).to(self.device)
        task_tensor = torch.tensor(task_type, dtype=torch.float32).to(self.device)
        
        # 确保所有张量形状正确，并在同一设备上
        if len(text_embedding.shape) == 1:
            text_embedding = text_embedding.unsqueeze(0)
        
        cursor_tensor = cursor_tensor.unsqueeze(0)
        task_tensor = task_tensor.unsqueeze(0)
        
        state = torch.cat([text_embedding, cursor_tensor, task_tensor], dim=1)
        return state
    
    def select_action(self, obs, training=True):
        """根据当前策略选择动作"""
        # 处理观察
        try:
            state = self._process_observation(obs)
            
            # 检查状态是否有NaN
            if torch.isnan(state).any():
                print("警告: 状态包含NaN值! 替换为零。")
                state = torch.where(torch.isnan(state), torch.zeros_like(state), state)
                
            # 获取动作分布和状态值
            action_probs, state_value = self.policy(state)
            
            # 检查概率分布
            if torch.isnan(action_probs).any() or torch.sum(action_probs) == 0:
                print("警告: 动作概率有问题，使用均匀分布。")
                action_probs = torch.ones_like(action_probs) / self.num_actions
            
            # 确保状态值不是NaN
            if torch.isnan(state_value).any():
                print("警告: 状态值是NaN! 替换为零。")
                state_value = torch.zeros_like(state_value)
            
            # 采样动作
            if training:
                try:
                    action_dist = torch.distributions.Categorical(action_probs)
                    action = action_dist.sample()
                    action_log_prob = action_dist.log_prob(action)
                    
                    # 存储经验
                    self.states.append(state.cpu().detach().numpy())
                    self.actions.append(action.item())
                    self.action_probs.append(action_log_prob.item())
                    self.values.append(state_value.item())
                except Exception as e:
                    print(f"采样动作时出错: {e}")
                    # 回退到随机动作
                    action = torch.tensor([random.randint(0, self.num_actions-1)], device=self.device)
                    self.states.append(state.cpu().detach().numpy())
                    self.actions.append(action.item())
                    self.action_probs.append(0.0)  # 假设为均匀分布
                    self.values.append(0.0)
            else:
                # 在评估模式下，直接选择最高概率的动作
                action = torch.argmax(action_probs, dim=1)
            
            return action.item()
            
        except Exception as e:
            print(f"选择动作时出错: {e}")
            # 如果出现任何错误，返回随机动作
            return random.randint(0, self.num_actions-1)
    
    def update(self):
        """更新策略网络"""
        try:
            # 检查是否有足够的数据
            if len(self.states) == 0 or len(self.rewards) == 0:
                print("警告: 没有足够的数据进行更新")
                return
            
            # 计算回报和优势
            returns = self._compute_returns()
            
            # 转换为张量，并确保在正确的设备上
            device = self.device
            
            try:
                # 尝试堆叠状态
                states_batch = torch.tensor(np.vstack(self.states), dtype=torch.float32).to(device)
                # 检查NaN
                if torch.isnan(states_batch).any():
                    print("警告: 状态批次包含NaN，将替换为零")
                    states_batch = torch.where(torch.isnan(states_batch), torch.zeros_like(states_batch), states_batch)
            except Exception as e:
                print(f"处理状态批次错误: {e}")
                # 回退到随机初始化
                print("使用随机状态批次")
                states_batch = torch.rand((len(self.states), self.states[0].shape[0]), device=device)
            
            actions_batch = torch.tensor(self.actions, dtype=torch.long).to(device)
            old_probs_batch = torch.tensor(self.action_probs, dtype=torch.float32).to(device)
            returns_batch = torch.tensor(returns, dtype=torch.float32).to(device)
            values_batch = torch.tensor(self.values, dtype=torch.float32).to(device)
            
            # 检查NaN
            if torch.isnan(old_probs_batch).any():
                print("警告: 动作概率批次包含NaN，将替换为小的常数")
                old_probs_batch = torch.where(torch.isnan(old_probs_batch), 
                                            torch.ones_like(old_probs_batch) * -10.0, 
                                            old_probs_batch)
            
            if torch.isnan(returns_batch).any() or torch.isnan(values_batch).any():
                print("警告: 回报或值批次包含NaN，将替换为零")
                returns_batch = torch.where(torch.isnan(returns_batch), torch.zeros_like(returns_batch), returns_batch)
                values_batch = torch.where(torch.isnan(values_batch), torch.zeros_like(values_batch), values_batch)
            
            # 计算优势
            advantages = returns_batch - values_batch
            
            # 规范化优势（可选）
            if not torch.isnan(advantages).any() and advantages.std() > 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            else:
                print("警告: 优势包含NaN或标准差为零")
                advantages = torch.zeros_like(advantages)
            
            # 进行多次策略更新
            for epoch in range(5):  # 典型的PPO实现会进行多个epoch
                try:
                    # 前向传播
                    action_probs, state_values = self.policy(states_batch)
                    
                    # 检查概率分布
                    if torch.isnan(action_probs).any():
                        print(f"警告: epoch {epoch}，动作概率有NaN，跳过此次更新")
                        continue
                    
                    # 获取新的动作概率
                    dist = torch.distributions.Categorical(action_probs)
                    new_probs = dist.log_prob(actions_batch)
                    
                    # 计算策略比率
                    ratio = torch.exp(new_probs - old_probs_batch)
                    
                    # 检查比率是否包含NaN或无穷大
                    if torch.isnan(ratio).any() or torch.isinf(ratio).any():
                        print(f"警告: epoch {epoch}，比率包含NaN或无穷大，将被裁剪")
                        ratio = torch.clamp(ratio, 0.1, 10.0)  # 更严格的裁剪
                    
                    # 计算PPO目标函数
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # 计算值函数损失
                    value_loss = F.mse_loss(state_values.squeeze(-1), returns_batch)
                    
                    # 计算熵奖励（鼓励探索）
                    entropy = dist.entropy().mean()
                    
                    # 总损失
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                    
                    # 检查损失是否为NaN
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"警告: epoch {epoch}，损失为NaN或无穷大，跳过此次更新")
                        continue
                    
                    # 反向传播和优化
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # 检查梯度是否为NaN
                    has_nan_grad = False
                    for param in self.policy.parameters():
                        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            has_nan_grad = True
                            break
                    
                    if has_nan_grad:
                        print(f"警告: epoch {epoch}，梯度包含NaN或无穷大，跳过此次更新")
                        continue
                    
                    # 梯度裁剪
                    nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                    self.optimizer.step()
                    
                except Exception as e:
                    print(f"更新过程中出错 (epoch {epoch}): {e}")
                    continue  # 跳过这次更新，尝试下一次
            
            # 清空缓冲区
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.action_probs.clear()
            self.values.clear()
            self.dones.clear()
            
        except Exception as e:
            print(f"整个更新过程出错: {e}")
            # 出现严重错误时，清空缓冲区防止累积错误数据
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.action_probs.clear()
            self.values.clear()
            self.dones.clear()
    
    def _compute_returns(self):
        """计算每个时间步的回报值"""
        returns = []
        
        # 从后向前计算回报
        next_return = 0
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                next_return = 0
            next_return = r + self.gamma * next_return
            returns.insert(0, next_return)
        
        return returns
    
    def train(self, num_episodes=1000, max_steps=100, update_frequency=20, render=False):
        """训练代理"""
        rewards_history = []
        
        for episode in tqdm(range(num_episodes)):
            # 初始化环境
            obs = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            self.dones.append(False)  # 初始状态未结束
            
            while not done and step < max_steps:
                # 选择动作
                action = self.select_action(obs)
                
                # 执行动作
                next_obs, reward, done, _ = self.env.step(action)
                
                # 存储奖励和结束状态
                self.rewards.append(reward)
                self.dones.append(done)
                
                # 更新状态和累积奖励
                obs = next_obs
                episode_reward += reward
                step += 1
                
                # 渲染环境（如果需要）
                if render:
                    self.env.render()
            
            # 记录本回合奖励
            rewards_history.append(episode_reward)
            
            # 定期更新策略
            if (episode + 1) % update_frequency == 0:
                self.update()
                # 打印当前进度
                avg_reward = np.mean(rewards_history[-update_frequency:])
                print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")
        
        return rewards_history
    
    def test(self, num_episodes=10, max_steps=100, render=False):
        """测试代理性能"""
        rewards = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < max_steps:
                # 选择动作（不训练）
                action = self.select_action(obs, training=False)
                
                # 执行动作
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
                step += 1
                
                if render:
                    self.env.render()
                    time.sleep(0.1)  # 放慢速度以便观察
            
            rewards.append(episode_reward)
            print(f"测试回合 {episode+1}/{num_episodes}, 奖励: {episode_reward:.2f}")
        
        return rewards

class GRPOAgent:
    """使用GRPO算法的文本编辑代理"""
    
    def __init__(self, env, text_embedding_dim=64, hidden_dim=128, gamma=0.99, 
                 clip_ratio=0.2, lr=3e-4, group_size=16, kl_coef=0.01,
                 use_gpt2=True, gpt2_path=None, ref_policy=None):
        self.env = env
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.kl_coef = kl_coef  # KL散度系数
        self.group_size = group_size  # 每组的样本数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 状态和动作空间
        self.num_actions = env.action_space.n
        
        # 确定是否使用GPT-2作为文本编码器
        self.use_gpt2 = use_gpt2 and gpt2_path is not None
        
        if self.use_gpt2:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(gpt2_path)
            
            # 设置pad token为eos token
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_path)
            self.gpt2_model.to(self.device)
            self.gpt2_model.eval()
            self.text_encoder = GPT2TextEncoder(self.tokenizer, self.gpt2_model)
            print(f"GPT-2模型成功加载: {gpt2_path}")
        else:
            self.text_encoder = TextEncoder(vocab_size=128, embedding_dim=text_embedding_dim, device=self.device)
            text_embedding_dim = self.text_encoder.embedding_dim
        
        # 策略网络：文本嵌入 + 光标位置 + 当前任务 -> 动作分布
        input_dim = text_embedding_dim + 2 + 3  # 文本嵌入 + 光标(x,y) + 任务类型(one-hot)
        
        # 使用之前设计的注意力机制CursorControlHead
        self.policy = CursorControlHead(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            num_actions=self.num_actions,
            num_heads=4  # 使用多头注意力机制
        ).to(self.device)
        
        # 参考策略（用于KL散度）
        self.ref_policy = ref_policy if ref_policy is not None else self.policy
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # 经验回放缓冲区 - 使用组结构
        self.groups = []  # 每个元素是一个包含组内样本的字典
    
    def _process_observation(self, obs):
        """处理观察，转换为模型输入"""
        # 处理文本
        if isinstance(obs, dict) and "text" in obs:
            text = obs["text"]
        else:
            text = str(obs)
            
        # 处理光标位置
        if isinstance(obs, dict):
            if "cursor_x" in obs and "cursor_y" in obs:
                cursor_x = obs["cursor_x"]
                cursor_y = obs["cursor_y"]
            elif "cursor_pos" in obs:
                cursor_pos = obs["cursor_pos"]
                if isinstance(cursor_pos, (list, tuple)) and len(cursor_pos) >= 2:
                    cursor_x, cursor_y = cursor_pos[0], cursor_pos[1]
                else:
                    cursor_x, cursor_y = 0, 0
            elif "cursor" in obs:
                cursor = obs["cursor"]
                if isinstance(cursor, (list, tuple)) and len(cursor) >= 2:
                    cursor_x, cursor_y = cursor[0], cursor[1]
                else:
                    cursor_x, cursor_y = 0, 0
            else:
                cursor_x, cursor_y = 0, 0
        else:
            cursor_x, cursor_y = 0, 0
            
        # 任务类型
        if isinstance(obs, dict):
            task_type = obs.get("task_type", [0,0,0])
        else:
            task_type = [0,0,0]
        
        # 文本编码
        if self.use_gpt2:
            with torch.no_grad():
                text_embedding = self.text_encoder.get_embeddings(text=text)
        else:
            text_indices = self.text_encoder.encode_text(text)
            text_embedding = self.text_encoder.get_embeddings(text_indices)
            text_embedding = torch.mean(text_embedding, dim=0, keepdim=True)
        
        # 合并特征
        cursor_tensor = torch.tensor([cursor_x, cursor_y], dtype=torch.float32).to(self.device)
        task_tensor = torch.tensor(task_type, dtype=torch.float32).to(self.device)
        
        # 确保所有张量形状正确，并在同一设备上
        if len(text_embedding.shape) == 1:
            text_embedding = text_embedding.unsqueeze(0)
        
        cursor_tensor = cursor_tensor.unsqueeze(0)
        task_tensor = task_tensor.unsqueeze(0)
        
        state = torch.cat([text_embedding, cursor_tensor, task_tensor], dim=1)
        return state
    
    def select_action(self, obs, training=True):
        """根据当前策略选择动作"""
        state = self._process_observation(obs)
        
        if torch.isnan(state).any():
            print("警告: 状态包含NaN值! 替换为零。")
            state = torch.where(torch.isnan(state), torch.zeros_like(state), state)
            
        action_probs, state_value = self.policy(state)
        
        if torch.isnan(action_probs).any() or torch.sum(action_probs) == 0:
            print("警告: 动作概率有问题，使用均匀分布。")
            action_probs = torch.ones_like(action_probs) / self.num_actions
        
        if training:
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)
            
            # 存储信息 - GRPO中我们需要保存更多信息以进行组比较
            if len(self.groups) == 0 or len(self.groups[-1]["states"]) >= self.group_size:
                # 创建新组
                self.groups.append({
                    "states": [state.cpu().detach().numpy()],
                    "actions": [action.item()],
                    "action_log_probs": [action_log_prob.item()],
                    "rewards": [],
                    "dones": []
                })
            else:
                # 添加到当前组
                self.groups[-1]["states"].append(state.cpu().detach().numpy())
                self.groups[-1]["actions"].append(action.item())
                self.groups[-1]["action_log_probs"].append(action_log_prob.item())
        else:
            # 在评估模式下，直接选择最高概率的动作
            action = torch.argmax(action_probs, dim=1)
        
        return action.item()
    
    def store_reward(self, reward, done):
        """存储奖励和完成状态"""
        if len(self.groups) > 0 and len(self.groups[-1]["rewards"]) < len(self.groups[-1]["actions"]):
            self.groups[-1]["rewards"].append(reward)
            self.groups[-1]["dones"].append(done)
    
    def update(self):
        """更新策略网络 - GRPO实现"""
        # 检查是否有足够的数据
        if len(self.groups) == 0:
            print("警告: 没有足够的数据进行更新")
            return
        
        # 过滤出完整的组(状态、动作、奖励数量一致的组)
        complete_groups = []
        for group in self.groups:
            if (len(group["states"]) == len(group["actions"]) == 
                len(group["rewards"]) == len(group["action_log_probs"])):
                complete_groups.append(group)
        
        if len(complete_groups) == 0:
            print("警告: 没有完整的组数据")
            return
            
        device = self.device
        
        # 对每个完整的组进行GRPO更新
        for group in complete_groups:
            # 转换为张量
            states_batch = torch.tensor(np.vstack(group["states"]), dtype=torch.float32).to(device)
            if torch.isnan(states_batch).any():
                states_batch = torch.where(torch.isnan(states_batch), torch.zeros_like(states_batch), states_batch)
                    
            actions_batch = torch.tensor(group["actions"], dtype=torch.long).to(device)
            old_log_probs_batch = torch.tensor(group["action_log_probs"], dtype=torch.float32).to(device)
            rewards_batch = torch.tensor(group["rewards"], dtype=torch.float32).to(device)
            
            # 计算组内优势 - GRPO的核心
            rewards_mean = torch.mean(rewards_batch)
            rewards_std = torch.std(rewards_batch) + 1e-8  # 避免除以零
                    
            # 按照论文中的公式计算优势
            advantages = (rewards_batch - rewards_mean) / rewards_std
            
            # 前向传播获取新策略
            action_probs, _ = self.policy(states_batch)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions_batch)
            
            # 计算策略比率 (π_θ/π_θ_old)
            ratio = torch.exp(new_log_probs - old_log_probs_batch)
            
            # 计算GRPO目标函数的第一部分 - 裁剪目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 计算KL散度（对参考策略）- GRPO特有
            # 使用更稳定的方法计算KL散度
            if self.ref_policy is not self.policy:
                with torch.no_grad():
                    ref_action_probs, _ = self.ref_policy(states_batch)
                
                # 使用F.kl_div计算KL散度，更加稳定
                kl_div = F.kl_div(
                    F.log_softmax(action_probs, dim=1),
                    F.softmax(ref_action_probs, dim=1),
                    reduction='batchmean',
                    log_target=False
                )
            else:
                kl_div = torch.tensor(0.0, device=device)
            
            # 总损失 = 策略损失 + KL散度惩罚
            loss = policy_loss + self.kl_coef * kl_div
            
            # 梯度更新
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
        
        # 清空缓冲区
        self.groups = []
    
    def train(self, num_episodes=1000, max_steps=100, update_frequency=20, render=False):
        """训练代理"""
        rewards_history = []
        
        for episode in tqdm(range(num_episodes)):
            # 初始化环境
            obs = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < max_steps:
                # 选择动作
                action = self.select_action(obs)
                
                # 执行动作
                next_obs, reward, done, _ = self.env.step(action)
                
                # 存储奖励和结束状态
                self.store_reward(reward, done)
                
                # 更新状态和累积奖励
                obs = next_obs
                episode_reward += reward
                step += 1
                
                # 渲染环境（如果需要）
                if render:
                    self.env.render()
            
            # 记录本回合奖励
            rewards_history.append(episode_reward)
            
            # 定期更新策略
            if (episode + 1) % update_frequency == 0 or len(self.groups) >= 10:
                self.update()
                # 打印当前进度
                avg_reward = np.mean(rewards_history[-update_frequency:])
                print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")
        
        return rewards_history
    
    def test(self, num_episodes=10, max_steps=100, render=False):
        """测试代理性能"""
        rewards = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < max_steps:
                # 选择动作（不训练）
                action = self.select_action(obs, training=False)
                
                # 执行动作
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
                step += 1
                
                if render:
                    self.env.render()
                    time.sleep(0.1)  # 放慢速度以便观察
            
            rewards.append(episode_reward)
            print(f"测试回合 {episode+1}/{num_episodes}, 奖励: {episode_reward:.2f}")
        
        return rewards