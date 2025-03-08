import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import numpy as np
from tqdm import tqdm
import random
import argparse
import matplotlib.pyplot as plt

# 导入我们已有的模块
from cursor_control_agent import PPOAgent, GRPOAgent
from text_editing_environment import TextEditingEnv
from task_generator_rewards import TaskGenerator, TextEditingEnvWithTasks

def load_cold_start_data(data_path):
    """加载冷启动数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_coldstart_dataset(cold_start_data):
    """将冷启动数据转换为训练数据集"""
    states = []
    actions = []
    
    for example in cold_start_data:
        initial_text = example['initial_text']
        action_sequence = example['action_sequence']
        
        # 创建环境实例
        env = TextEditingEnv(initial_text=initial_text)
        obs = env.reset()
        
        # 记录每个状态和对应的动作
        for action in action_sequence:
            # 将状态添加到数据集
            states.append(obs)
            actions.append(action)
            
            # 执行动作
            obs, _, done, _ = env.step(action)
            if done:
                break
    
    return states, actions

def train_supervised(agent, states, actions, batch_size=32, epochs=40, learning_rate=1e-4):
    """使用冷启动数据进行监督学习训练"""
    # 使用交叉熵损失
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(agent.policy.parameters(), lr=learning_rate)
    
    # 将数据转换为PyTorch张量
    dataset_size = len(states)
    losses = []
    
    for epoch in range(epochs):
        # 打乱数据集
        indices = list(range(dataset_size))
        random.shuffle(indices)
        
        total_loss = 0
        num_batches = 0
        
        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]
            
            batch_states = [states[i] for i in batch_indices]
            batch_actions = [actions[i] for i in batch_indices]
            
            # 将状态转换为模型输入
            processed_states = []
            for state in batch_states:
                processed_state = agent._process_observation(state)
                processed_states.append(processed_state)
            
            try:
                # 合并处理后的状态
                state_tensor = torch.cat(processed_states, dim=0)
            except:
                # 如果状态维度不一致，单独处理每个状态
                continue
                
            action_tensor = torch.tensor(batch_actions, dtype=torch.long).to(agent.device)
            
            # 前向传播
            action_probs, _ = agent.policy(state_tensor)
            
            # 计算损失
            loss = criterion(action_probs, action_tensor)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return losses

def cold_start_agent(args):
    """执行冷启动训练并返回初始化好的代理"""
    print("初始化环境和任务生成器...")
    task_generator = TaskGenerator()
    env = TextEditingEnvWithTasks(task_generator, max_steps=args.max_steps)
    
    print("初始化代理...")
    agent = GRPOAgent(
        env=env,
        text_embedding_dim=768,  # GPT-2隐藏维度
        hidden_dim=256,
        gamma=args.gamma,
        clip_ratio=args.clip_ratio,
        lr=args.learning_rate,
        gpt2_path=args.gpt2_path
    )
    
    # 加载冷启动数据
    print(f"加载冷启动数据: {args.cold_start_data}")
    cold_start_data = load_cold_start_data(args.cold_start_data)
    
    print(f"创建冷启动数据集...")
    states, actions = create_coldstart_dataset(cold_start_data)
    
    print(f"开始监督学习训练...")
    losses = train_supervised(
        agent=agent,
        states=states,
        actions=actions,
        batch_size=args.batch_size,
        epochs=args.coldstart_epochs,
        learning_rate=args.coldstart_lr
    )
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('监督学习损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('coldstart_loss.png')
    
    # 保存冷启动模型
    print(f"保存冷启动模型到: {args.coldstart_model}")
    torch.save(agent.policy.state_dict(), args.coldstart_model)
    
    return agent

def main():
    parser = argparse.ArgumentParser(description='冷启动训练后进行强化学习')
    parser.add_argument('--cold-start-data', type=str, required=True, help='冷启动数据路径')
    parser.add_argument('--coldstart-epochs', type=int, default=5, help='冷启动训练的epoch数')
    parser.add_argument('--coldstart-lr', type=float, default=1e-4, help='冷启动学习率')
    parser.add_argument('--coldstart-model', type=str, default='coldstart_model.pt', help='冷启动模型保存路径')
    
    # 强化学习参数
    parser.add_argument('--mode', type=str, default='coldstart+rl', 
                        choices=['coldstart', 'rl', 'coldstart+rl'],
                        help='运行模式: coldstart, rl, or coldstart+rl')
    parser.add_argument('--task', type=str, default=None, 
                        choices=[None, 'copy_paste', 'find_replace', 'cursor_navigation'],
                        help='任务类型，为None则随机选择')
    parser.add_argument('--num-episodes', type=int, default=5000, help='RL训练的回合数')
    parser.add_argument('--max-steps', type=int, default=100, help='每个回合的最大步数')
    parser.add_argument('--batch-size', type=int, default=64, help='训练的批次大小')
    parser.add_argument('--update-frequency', type=int, default=20, help='策略更新频率(回合)')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--clip-ratio', type=float, default=0.2, help='PPO裁剪比例')
    parser.add_argument('--load-model', type=str, default=None, help='要加载的模型路径')
    parser.add_argument('--save-model', type=str, default='cursor_control_agent.pt', help='模型保存路径')
    parser.add_argument('--render', action='store_true', help='是否渲染环境')
    parser.add_argument('--eval-episodes', type=int, default=10, help='评估的回合数')
    parser.add_argument('--gpt2-path', type=str, default='E:\\ML\\example\\nlp\\GPT-2', help='GPT-2模型路径')
    
    args = parser.parse_args()
    
    if args.mode in ['coldstart', 'coldstart+rl']:
        # 执行冷启动
        agent = cold_start_agent(args)
        
        if args.mode == 'coldstart':
            # 仅执行冷启动，结束
            return
    
    if args.mode in ['rl', 'coldstart+rl']:
        # 准备RL训练
        print("初始化环境和任务生成器...")
        task_generator = TaskGenerator()
        env = TextEditingEnvWithTasks(task_generator, max_steps=args.max_steps)
        
        if args.mode == 'rl':
            # 仅RL，初始化新代理
            print("初始化代理...")
            agent = GRPOAgent(
                env=env,
                text_embedding_dim=768,  # GPT-2隐藏维度
                hidden_dim=256,
                gamma=args.gamma,
                clip_ratio=args.clip_ratio,
                lr=args.learning_rate,
                gpt2_path=args.gpt2_path
            )
            
            # 如果指定了模型路径，则加载模型
            if args.load_model and os.path.exists(args.load_model):
                print(f"加载模型: {args.load_model}")
                agent.policy.load_state_dict(torch.load(args.load_model))
        
        # 执行RL训练
        print(f"开始RL训练，共{args.num_episodes}个回合...")
        rewards = agent.train(
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            update_frequency=args.update_frequency,
            render=args.render
        )
        
        # 保存最终模型
        print(f"保存模型到: {args.save_model}")
        torch.save(agent.policy.state_dict(), args.save_model)
        
        # 绘制学习曲线
        plt.figure(figsize=(10, 5))
        plt.plot(np.convolve(rewards, np.ones(args.update_frequency)/args.update_frequency, mode='valid'))
        plt.title('RL学习曲线 (移动平均)')
        plt.xlabel('回合')
        plt.ylabel('奖励')
        plt.savefig('rl_learning_curve.png')
        plt.show()

if __name__ == "__main__":
    main()