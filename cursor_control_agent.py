import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import deque
import random
import time

class CursorControlHead(nn.Module):
    """
    光标控制头部网络，输出动作分布
    """
    def __init__(self, input_dim, hidden_dim, num_actions=12):
        super(CursorControlHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # 检查输入是否包含NaN
        if torch.isnan(x).any():
            print("警告: 输入包含NaN值!")
            # 替换NaN为0
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # 确保没有过大的值导致数值不稳定
        logits = self.action_head(x)
        # 应用更稳定的softmax，先减去最大值
        logits_max = torch.max(logits, dim=-1, keepdim=True)[0]
        logits_stable = logits - logits_max
        action_probs = F.softmax(logits_stable, dim=-1)
        
        # 确保概率之和为1
        # 有时数值误差会导致加起来不完全是1
        probs_sum = torch.sum(action_probs, dim=-1, keepdim=True)
        action_probs = action_probs / probs_sum
        
        # 检查概率是否包含NaN
        if torch.isnan(action_probs).any():
            print("警告: action_probs包含NaN值!")
            # 回退到均匀分布
            action_probs = torch.ones_like(action_probs) / self.action_head.out_features
        
        state_values = self.value_head(x)
        
        return action_probs, state_values


class GPT2TextEncoder:
    """
    使用预训练的GPT-2模型将文本转换为特征表示
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
    
    def get_embeddings(self, inputs=None, text=None, pooling="mean"):
        """
        获取文本的嵌入表示
        
        参数:
        - inputs: 已经通过encode_text处理的输入
        - text: 原始文本，如果inputs为None则使用
        - pooling: 池化方法，可选"mean"、"last"或"cls"
        
        返回:
        - 文本的嵌入向量
        """
        if inputs is None and text is not None:
            inputs = self.encode_text(text)
        
        if inputs is None:
            raise ValueError("必须提供inputs或text参数之一")
        
        # 获取GPT-2的隐藏状态
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
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
            # 使用第一个token (类似BERT的[CLS]，但GPT-2没有专门的CLS token)
            embeddings = hidden_states[:, 0, :]
        else:
            raise ValueError(f"不支持的池化方法: {pooling}")
        
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
    """使用PPO算法的文本编辑代理"""
    
    def __init__(self, env, text_embedding_dim=64, hidden_dim=128, gamma=0.99, 
                 clip_ratio=0.2, lr=3e-4, use_gpt2=True, gpt2_path=None):
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
            try:
                device = self.device
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(gpt2_path)
                
                # 设置pad token为eos token（GPT-2默认没有pad token）
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_path)
                self.gpt2_model.to(device)
                self.gpt2_model.eval()
                self.text_encoder = GPT2TextEncoder(self.tokenizer, self.gpt2_model)
                print(f"GPT-2模型成功加载: {gpt2_path}")
            except Exception as e:
                print(f"加载GPT-2模型失败: {e}")
                print("将使用基本文本编码器")
                self.use_gpt2 = False
        
        if not self.use_gpt2:
            self.text_encoder = TextEncoder(vocab_size=128, embedding_dim=text_embedding_dim, device=self.device)
            text_embedding_dim = self.text_encoder.embedding_dim
        
        # 策略网络：文本嵌入 + 光标位置 + 当前任务 -> 动作分布
        input_dim = text_embedding_dim + 2 + 3  # 文本嵌入 + 光标(x,y) + 任务类型(one-hot)
        self.policy = nn.Sequential(
            CursorControlHead(input_dim, hidden_dim, self.num_actions)
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 经验回放缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.action_probs = []
        self.values = []
        self.dones = []
    
    def _process_observation(self, obs):
        """处理观察，转换为模型输入"""
        # 打印观察值的键，帮助调试
        # print(f"观察值键: {obs.keys()}")
        
        # 处理文本 - 确保text键存在
        if isinstance(obs, dict) and "text" in obs:
            text = obs["text"]
        else:
            # 如果没有text键，尝试将整个obs当作文本
            text = str(obs)
            
        # 处理光标位置 - 适应不同的键名或提供默认值
        if isinstance(obs, dict):
            # 尝试不同可能的键名
            if "cursor_x" in obs and "cursor_y" in obs:
                cursor_x = obs["cursor_x"]
                cursor_y = obs["cursor_y"]
            elif "cursor_pos" in obs:
                # 如果是元组或列表形式
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
                # 默认值
                cursor_x, cursor_y = 0, 0
        else:
            # 默认值
            cursor_x, cursor_y = 0, 0
            
        # 任务类型 - 提供默认值
        if isinstance(obs, dict):
            task_type = obs.get("task_type", [0,0,0])  # 默认为all-zero向量
        else:
            task_type = [0,0,0]
        
        # 文本编码
        if self.use_gpt2:
            with torch.no_grad():
                text_embedding = self.text_encoder.get_embeddings(text=text)
        else:
            text_indices = self.text_encoder.encode_text(text)
            text_embedding = self.text_encoder.get_embeddings(text_indices)
            text_embedding = torch.mean(text_embedding, dim=0, keepdim=True)  # 平均池化
        
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