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
    光标控制头部网络，可自动检测和适应输入维度
    """
    def __init__(self, input_dim=None, hidden_dim=256, num_actions=12, num_heads=4):
        super(CursorControlHead, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.num_heads = num_heads
        self.input_dim = input_dim  # 可能为None，稍后初始化
        
        # 延迟初始化，等待第一次前向传播确定输入维度
        self.initialized = False
        
    def initialize(self, input_dim):
        """根据输入维度动态初始化网络"""
        self.input_dim = input_dim
        print(f"初始化CursorControlHead，输入维度: {input_dim}，隐藏维度: {self.hidden_dim}")
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        nn.init.xavier_normal_(self.input_proj.weight, gain=0.01)
        
        # 多头自注意力层
        self.attention = nn.MultiheadAttention(
            self.hidden_dim, 
            num_heads=self.num_heads, 
            batch_first=True
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)
        
        # 输出头
        self.action_head = nn.Linear(self.hidden_dim, self.num_actions)
        nn.init.xavier_normal_(self.action_head.weight, gain=0.01)
        
        self.value_head = nn.Linear(self.hidden_dim, 1)
        nn.init.xavier_normal_(self.value_head.weight, gain=0.01)
        
        self.initialized = True
        
    def forward(self, x):
        # 检查是否初始化，如果没有，使用输入的维度初始化模型
        if not self.initialized:
            self.initialize(x.size(-1))
            # 将网络移动到与输入相同的设备
            device = x.device
            self.to(device)
        
        # 检查输入是否包含NaN
        if torch.isnan(x).any():
            print("警告: CursorControlHead输入包含NaN，将替换为零")
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        # 投影到隐藏维度
        hidden = self.input_proj(x)
        
        # 把特征展开为序列形式以便注意力处理
        if len(hidden.shape) == 2:
            hidden = hidden.unsqueeze(1)  # 添加序列维度
        
        # 自注意力 - 捕获潜在的错误
        try:
            attn_output, _ = self.attention(hidden, hidden, hidden)
            hidden = self.layer_norm1(hidden + attn_output)
        except Exception as e:
            print(f"注意力计算出错: {e}")
            # 跳过注意力计算
            hidden = self.layer_norm1(hidden)
        
        # 前馈网络
        ff_output = self.feed_forward(hidden)
        hidden = self.layer_norm2(hidden + ff_output)
        
        # 挤压序列维度
        if hidden.size(1) == 1:
            hidden = hidden.squeeze(1)
        
        # 输出层
        action_logits = self.action_head(hidden)
        
        # 更稳定的softmax - 首先减去最大值
        max_logits = torch.max(action_logits, dim=-1, keepdim=True)[0]
        action_logits_stable = action_logits - max_logits
        
        # 使用更稳健的softmax方法
        exp_logits = torch.exp(action_logits_stable)
        exp_sum = torch.sum(exp_logits, dim=-1, keepdim=True)
        exp_sum = torch.clamp(exp_sum, min=1e-10)  # 避免除以零
        action_probs = exp_logits / exp_sum
        
        # 再次检查和清理NaN
        if torch.isnan(action_probs).any():
            print("警告: 动作概率包含NaN，将使用均匀分布")
            action_probs = torch.ones_like(action_probs) / self.num_actions
        
        # 状态值估计
        state_values = self.value_head(hidden)
        
        # 检查状态值是否为NaN
        if torch.isnan(state_values).any():
            print("警告: 状态值包含NaN，将替换为零")
            state_values = torch.zeros_like(state_values)
        
        return action_probs, state_values

class ModularCursorControlHead(nn.Module):
    """
    模块化光标控制头部网络，功能分离的多头注意力机制
    """
    def __init__(self, input_dim=None, hidden_dim=256, num_actions=12):
        super(ModularCursorControlHead, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.input_dim = input_dim  # 可能为None，稍后初始化
        
        # 延迟初始化，等待第一次前向传播确定输入维度
        self.initialized = False
        
    def initialize(self, input_dim):
        """根据输入维度动态初始化网络"""
        self.input_dim = input_dim
        print(f"初始化ModularCursorControlHead，输入维度: {input_dim}，隐藏维度: {self.hidden_dim}")
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        nn.init.xavier_normal_(self.input_proj.weight, gain=0.01)
        
        # 1. 文本理解模块 - 理解文本内容和结构 (8头)
        self.understanding_attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads=8, batch_first=True
        )
        
        # 2. 空间感知模块 - 处理光标位置和文本布局 (8头)
        self.spatial_attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads=8, batch_first=True
        )
        
        # 3. 操作规划模块 - 计划复杂操作如复制粘贴 (8头)
        self.planning_attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads=8, batch_first=True
        )
        
        # 4. 执行控制模块 - 最终决定具体动作 (8头)
        self.control_attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads=8, batch_first=True
        )
        
        # 模块间转换层
        self.understanding_to_spatial = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.xavier_normal_(self.understanding_to_spatial.weight, gain=0.01)
        
        self.spatial_to_planning = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.xavier_normal_(self.spatial_to_planning.weight, gain=0.01)
        
        self.planning_to_control = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.xavier_normal_(self.planning_to_control.weight, gain=0.01)
        
        # 每个模块的前馈网络
        self.understanding_ff = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        self.spatial_ff = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        self.planning_ff = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        self.control_ff = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 层归一化
        self.norm_u1 = nn.LayerNorm(self.hidden_dim)
        self.norm_u2 = nn.LayerNorm(self.hidden_dim)
        self.norm_s1 = nn.LayerNorm(self.hidden_dim)
        self.norm_s2 = nn.LayerNorm(self.hidden_dim)
        self.norm_p1 = nn.LayerNorm(self.hidden_dim)
        self.norm_p2 = nn.LayerNorm(self.hidden_dim)
        self.norm_c1 = nn.LayerNorm(self.hidden_dim)
        self.norm_c2 = nn.LayerNorm(self.hidden_dim)
        
        # 输出头
        self.action_head = nn.Linear(self.hidden_dim, self.num_actions)
        nn.init.xavier_normal_(self.action_head.weight, gain=0.01)
        
        self.value_head = nn.Linear(self.hidden_dim, 1)
        nn.init.xavier_normal_(self.value_head.weight, gain=0.01)
        
        self.initialized = True
        
    def forward(self, x):
        # 检查是否初始化，如果没有，使用输入的维度初始化模型
        if not self.initialized:
            self.initialize(x.size(-1))
            # 将网络移动到与输入相同的设备
            device = x.device
            self.to(device)
        
        # 检查输入是否包含NaN
        if torch.isnan(x).any():
            print("警告: ModularCursorControlHead输入包含NaN，将替换为零")
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        # 投影到隐藏维度
        hidden = self.input_proj(x)
        
        # 把特征展开为序列形式以便注意力处理
        if len(hidden.shape) == 2:
            hidden = hidden.unsqueeze(1)  # 添加序列维度
        
        # 1. 文本理解模块处理
        try:
            understanding_attn, _ = self.understanding_attention(hidden, hidden, hidden)
            understanding = self.norm_u1(hidden + understanding_attn)
            understanding_ff = self.understanding_ff(understanding)
            understanding = self.norm_u2(understanding + understanding_ff)
        except Exception as e:
            print(f"文本理解模块出错: {e}")
            understanding = self.norm_u2(hidden)  # 跳过出错的模块
            
        # 2. 空间感知模块处理
        try:
            spatial_in = self.understanding_to_spatial(understanding)
            spatial_attn, _ = self.spatial_attention(spatial_in, spatial_in, spatial_in)
            spatial = self.norm_s1(spatial_in + spatial_attn)
            spatial_ff = self.spatial_ff(spatial)
            spatial = self.norm_s2(spatial + spatial_ff)
        except Exception as e:
            print(f"空间感知模块出错: {e}")
            spatial = self.norm_s2(understanding)  # 使用前一模块输出
            
        # 3. 操作规划模块处理
        try:
            planning_in = self.spatial_to_planning(spatial)
            planning_attn, _ = self.planning_attention(planning_in, planning_in, planning_in)
            planning = self.norm_p1(planning_in + planning_attn)
            planning_ff = self.planning_ff(planning)
            planning = self.norm_p2(planning + planning_ff)
        except Exception as e:
            print(f"操作规划模块出错: {e}")
            planning = self.norm_p2(spatial)  # 使用前一模块输出
            
        # 4. 执行控制模块处理
        try:
            control_in = self.planning_to_control(planning)
            control_attn, _ = self.control_attention(control_in, control_in, control_in)
            control = self.norm_c1(control_in + control_attn)
            control_ff = self.control_ff(control)
            control = self.norm_c2(control + control_ff)
        except Exception as e:
            print(f"执行控制模块出错: {e}")
            control = self.norm_c2(planning)  # 使用前一模块输出
            
        # 挤压序列维度
        if control.size(1) == 1:
            control = control.squeeze(1)
        
        # 输出层 - 动作概率
        action_logits = self.action_head(control)
        
        # 更稳定的softmax - 首先减去最大值
        max_logits = torch.max(action_logits, dim=-1, keepdim=True)[0]
        action_logits_stable = action_logits - max_logits
        
        # 使用更稳健的softmax方法
        exp_logits = torch.exp(action_logits_stable)
        exp_sum = torch.sum(exp_logits, dim=-1, keepdim=True)
        exp_sum = torch.clamp(exp_sum, min=1e-10)  # 避免除以零
        action_probs = exp_logits / exp_sum
        
        # 再次检查和清理NaN
        if torch.isnan(action_probs).any():
            print("警告: 动作概率包含NaN，将使用均匀分布")
            action_probs = torch.ones_like(action_probs) / self.num_actions
        
        # 状态值估计
        state_values = self.value_head(control)
        
        # 检查状态值是否为NaN
        if torch.isnan(state_values).any():
            print("警告: 状态值包含NaN，将替换为零")
            state_values = torch.zeros_like(state_values)
        
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
        # self.policy = CursorControlHead(
        self.policy = ModularCursorControlHead(
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
    
    def __init__(self, env, text_embedding_dim=None, hidden_dim=256, gamma=0.99, 
                 clip_ratio=0.2, lr=1e-4, group_size=16, kl_coef=0.01,
                 use_gpt2=False, gpt2_path=None, llama_path=None, ref_policy=None):
        self.env = env
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.kl_coef = kl_coef  # KL散度系数
        self.group_size = group_size  # 每组的样本数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 状态和动作空间
        self.num_actions = env.action_space.n
        
        # 根据提供的路径决定使用哪个模型
        self.use_gpt2 = gpt2_path is not None
        self.use_llama = llama_path is not None
        
        # 确保不同时使用两种模型
        if self.use_gpt2 and self.use_llama:
            print("警告: 同时指定了GPT-2和Llama路径，将使用Llama")
            self.use_gpt2 = False
        
        # 初始化文本编码器
        if self.use_gpt2:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(gpt2_path)
            
            # 设置pad token为eos token
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.lm_model = AutoModelForCausalLM.from_pretrained(gpt2_path)
            self.lm_model.to(self.device)
            self.lm_model.eval()
            self.text_encoder = GPT2TextEncoder(self.tokenizer, self.lm_model)
            print(f"GPT-2模型成功加载: {gpt2_path}")
        elif self.use_llama:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(llama_path)
            
            # 设置pad token如果需要
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.lm_model = AutoModelForCausalLM.from_pretrained(llama_path)
            self.lm_model.to(self.device)
            self.lm_model.eval()
            self.text_encoder = LLMTextEncoder(self.tokenizer, self.lm_model)
            print(f"Llama模型成功加载: {llama_path}")
        else:
            # 如果既没有GPT-2也没有Llama，使用基本文本编码器
            self.text_encoder = TextEncoder(vocab_size=128, embedding_dim=text_embedding_dim or 64, device=self.device)
            text_embedding_dim = self.text_encoder.embedding_dim
            print("使用基本文本编码器")
        
        # 初始化策略网络
        # 不再在初始化时指定input_dim，让模型动态调整
        # self.policy = CursorControlHead(
        self.policy = ModularCursorControlHead(
            input_dim=None,  # 稍后根据第一个样本动态确定
            hidden_dim=hidden_dim,
            num_actions=self.num_actions
        ).to(self.device)
        
        # 参考策略（用于KL散度）
        self.ref_policy = ref_policy if ref_policy is not None else self.policy
        
        # 优化器 - 等待模型完全初始化后再创建
        self.optimizer = None
        
        # 经验回放缓冲区 - 使用组结构
        self.groups = []  # 每个元素是一个包含组内样本的字典
    
    def _initialize_optimizer(self):
        """初始化优化器，确保网络已完全准备好"""
        if self.optimizer is None and hasattr(self.policy, 'initialized') and self.policy.initialized:
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
            print("优化器已初始化")
    
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
            
        # 如果策略网络已经初始化但优化器尚未创建，现在创建它
        if hasattr(self.policy, 'initialized') and self.policy.initialized and self.optimizer is None:
            self._initialize_optimizer()
            
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
        """更新策略网络 - GRPO实现，增强数值稳定性"""
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
            try:
                # 转换为张量
                states_batch = torch.tensor(np.vstack(group["states"]), dtype=torch.float32).to(device)
                
                # 检查和处理NaN状态
                if torch.isnan(states_batch).any():
                    print("警告: 状态批次包含NaN，将替换为零")
                    states_batch = torch.where(torch.isnan(states_batch), torch.zeros_like(states_batch), states_batch)
                    
                actions_batch = torch.tensor(group["actions"], dtype=torch.long).to(device)
                old_log_probs_batch = torch.tensor(group["action_log_probs"], dtype=torch.float32).to(device)
                rewards_batch = torch.tensor(group["rewards"], dtype=torch.float32).to(device)
                
                # 检查和处理NaN
                if torch.isnan(old_log_probs_batch).any():
                    print("警告: 动作概率批次包含NaN，将替换为小的常数")
                    old_log_probs_batch = torch.where(torch.isnan(old_log_probs_batch), 
                                                torch.ones_like(old_log_probs_batch) * -10.0, 
                                                old_log_probs_batch)
                
                if torch.isnan(rewards_batch).any():
                    print("警告: 奖励批次包含NaN，将替换为零")
                    rewards_batch = torch.where(torch.isnan(rewards_batch), 
                                            torch.zeros_like(rewards_batch), 
                                            rewards_batch)
                
                # 计算组内优势 - GRPO的核心
                rewards_mean = torch.mean(rewards_batch)
                rewards_std = torch.std(rewards_batch) + 1e-8  # 避免除以零
                        
                # 按照论文中的公式计算优势
                advantages = (rewards_batch - rewards_mean) / rewards_std
                
                # 前向传播获取新策略，使用try-except捕获潜在错误
                try:
                    # 检查模型中是否有NaN权重
                    for name, param in self.policy.named_parameters():
                        if torch.isnan(param).any():
                            print(f"警告: 参数 {name} 包含NaN值，将跳过此更新")
                            raise ValueError("模型参数包含NaN")
                    
                    action_probs, _ = self.policy(states_batch)
                    
                    # 处理action_probs中的NaN和异常值
                    if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                        print("警告: 动作概率包含NaN或Inf，将替换为均匀分布")
                        action_probs = torch.ones_like(action_probs) / self.num_actions
                    
                    # 确保概率和为1
                    action_probs_sum = action_probs.sum(dim=1, keepdim=True)
                    # 避免除以零或接近零的值
                    action_probs_sum = torch.where(action_probs_sum < 1e-8, 
                                                torch.ones_like(action_probs_sum), 
                                                action_probs_sum)
                    action_probs = action_probs / action_probs_sum
                    
                    # 创建分布前再次检查
                    if torch.isnan(action_probs).any() or (action_probs < 0).any() or (action_probs > 1).any():
                        print("警告: 动作概率仍然无效，跳过此次更新")
                        continue
                    
                    # 创建分布
                    dist = torch.distributions.Categorical(action_probs)
                    
                    new_log_probs = dist.log_prob(actions_batch)
                    
                    # 计算策略比率 (π_θ/π_θ_old)
                    ratio = torch.exp(torch.clamp(new_log_probs - old_log_probs_batch, -20, 20))  # 限制比率范围避免数值不稳定
                    
                    # 计算GRPO目标函数的第一部分 - 裁剪目标
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # 计算KL散度（对参考策略）- GRPO特有
                    if self.ref_policy is not self.policy:
                        with torch.no_grad():
                            ref_action_probs, _ = self.ref_policy(states_batch)
                            # 确保参考概率有效
                            if torch.isnan(ref_action_probs).any() or torch.isinf(ref_action_probs).any():
                                ref_action_probs = torch.ones_like(ref_action_probs) / self.num_actions
                            # 归一化
                            ref_action_probs = ref_action_probs / ref_action_probs.sum(dim=1, keepdim=True)
                        
                        # 使用稳定的KL散度计算
                        kl_div = F.kl_div(
                            F.log_softmax(action_probs, dim=1),
                            ref_action_probs,
                            reduction='batchmean',
                            log_target=False
                        )
                    else:
                        kl_div = torch.tensor(0.0, device=device)
                    
                    # 检查loss是否有效
                    if torch.isnan(policy_loss) or torch.isinf(policy_loss):
                        print("警告: 策略损失是NaN或Inf，跳过此次更新")
                        continue
                    
                    if torch.isnan(kl_div) or torch.isinf(kl_div):
                        print("警告: KL散度是NaN或Inf，设为0")
                        kl_div = torch.tensor(0.0, device=device)
                    
                    # 总损失 = 策略损失 + KL散度惩罚
                    loss = policy_loss + self.kl_coef * kl_div
                    
                    # 梯度更新
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # 检查梯度是否有NaN
                    grads_ok = True
                    for name, param in self.policy.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print(f"警告: 参数 {name} 的梯度包含NaN或Inf，跳过此次更新")
                                grads_ok = False
                                break
                    
                    if not grads_ok:
                        continue
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                except Exception as e:
                    print(f"前向传播或优化过程中出错: {e}")
                    continue
                
            except Exception as e:
                print(f"处理组数据时出错: {e}")
                continue
        
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

class LLMTextEncoder:
    """
    通用LLM文本编码器，适用于各种模型如Llama
    """
    
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.device = next(model.parameters()).device
        
        # 探测模型的隐藏维度
        self.hidden_dim = None
        for name, param in model.named_parameters():
            if 'embed' in name and len(param.shape) == 2:
                self.hidden_dim = param.shape[1]
                break
        
        if self.hidden_dim is None:
            # 如果无法探测，使用默认值
            self.hidden_dim = 4096  # Llama的常见隐藏维度
            print(f"警告: 无法探测模型隐藏维度，使用默认值: {self.hidden_dim}")
        else:
            print(f"探测到模型隐藏维度: {self.hidden_dim}")
    
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
        """
        if inputs is None and text is not None:
            inputs = self.encode_text(text)
        
        if inputs is None:
            raise ValueError("必须提供inputs或text参数之一")
        
        # 获取LLM的隐藏状态和注意力
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