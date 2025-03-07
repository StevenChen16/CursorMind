import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import pygame
import time
import sys

# 确保可以导入我们的模块
sys.path.append('.')

# 导入我们的模块
from cursor_control_agent import PPOAgent, GRPOAgent, TextEncoder, CursorControlHead
from task_generator_rewards import TaskGenerator, TextEditingEnvWithTasks

def parse_args():
    parser = argparse.ArgumentParser(description='训练LLM光标控制代理')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'demo'],
                        help='运行模式: train, test, or demo')
    parser.add_argument('--task', type=str, default=None, 
                        choices=[None, 'copy_paste', 'find_replace', 'cursor_navigation'],
                        help='任务类型，为None则随机选择')
    parser.add_argument('--num-episodes', type=int, default=5000, help='训练的回合数')
    parser.add_argument('--max-steps', type=int, default=100, help='每个回合的最大步数')
    parser.add_argument('--batch-size', type=int, default=64, help='PPO更新的批次大小')
    parser.add_argument('--update-frequency', type=int, default=20, help='策略更新频率(回合)')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--clip-ratio', type=float, default=0.2, help='PPO裁剪比例')
    parser.add_argument('--load-model', type=str, default=None, help='要加载的模型路径')
    parser.add_argument('--save-model', type=str, default='cursor_control_agent.pt', help='模型保存路径')
    parser.add_argument('--render', action='store_true', help='是否渲染环境')
    parser.add_argument('--eval-episodes', type=int, default=10, help='评估的回合数')
    parser.add_argument('--gpt2-path', type=str, default='E:\\ML\\example\\nlp\\GPT-2', help='GPT-2模型路径')
    parser.add_argument('--training-phase', type=int, default=1, choices=[1, 2], help='训练阶段: 1=基础光标移动, 2=完整功能')
    parser.add_argument('--phase-switch-episode', type=int, default=2000, help='切换到阶段2的回合数(仅当starting-phase=1时有效)')
    return parser.parse_args()
    return parser.parse_args()

def train_agent(args):
    """训练光标控制代理"""
    print("初始化环境和任务生成器...")
    task_generator = TaskGenerator()
    env = TextEditingEnvWithTasks(task_generator, max_steps=args.max_steps)
    
    # 设置初始训练阶段
    task_generator.set_training_phase(args.training_phase)
    
    print("初始化代理...")
    # agent = PPOAgent(
    agent = GRPOAgent(
        env=env,
        text_embedding_dim=768,
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
    
    print(f"开始训练，共{args.num_episodes}个回合...")
    rewards = []
    
    for episode in tqdm(range(args.num_episodes)):
        # 检查是否需要切换训练阶段
        if args.training_phase == 1 and episode == args.phase_switch_episode:
            print(f"\n切换到训练阶段2（完整功能）...")
            task_generator.set_training_phase(2)
            
            # 可选：保存阶段1完成时的检查点
            phase1_model_path = args.save_model.replace('.pt', '_phase1.pt')
            print(f"保存阶段1模型到: {phase1_model_path}")
            torch.save(agent.policy.state_dict(), phase1_model_path)
        
        # 现有的训练代码...
        obs = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < args.max_steps:
            action = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            
            # 存储奖励和结束状态
            if isinstance(agent, GRPOAgent):  # 如果使用GRPO代理
                agent.store_reward(reward, done)
            
            obs = next_obs
            episode_reward += reward
            step += 1
            
            if args.render:
                env.render()
        
        # 记录本回合奖励
        rewards.append(episode_reward)
        
        # 定期更新策略
        if (episode + 1) % args.update_frequency == 0:
            agent.update()
            # 打印当前进度
            avg_reward = np.mean(rewards[-args.update_frequency:])
            print(f"Episode {episode+1}/{args.num_episodes}, Avg Reward: {avg_reward:.2f}")
    
    # 保存模型
    print(f"保存模型到: {args.save_model}")
    torch.save(agent.policy.state_dict(), args.save_model)
    
    # 绘制学习曲线
    plt.figure(figsize=(10, 5))
    plt.plot(np.convolve(rewards, np.ones(args.update_frequency)/args.update_frequency, mode='valid'))
    plt.title('学习曲线 (移动平均)')
    plt.xlabel('回合')
    plt.ylabel('奖励')
    plt.savefig('learning_curve.png')
    plt.show()
    
    return agent


def test_agent(args):
    """测试代理性能"""
    print("初始化环境和任务生成器...")
    task_generator = TaskGenerator()
    env = TextEditingEnvWithTasks(task_generator, max_steps=args.max_steps)
    
    print("初始化代理...")
    # agent = PPOAgent(
    agent = GRPOAgent(
        env=env,
        text_embedding_dim=768,  # GPT-2隐藏维度
        hidden_dim=256,
        gpt2_path=args.gpt2_path
    )
    
    # 加载模型
    if args.load_model and os.path.exists(args.load_model):
        print(f"加载模型: {args.load_model}")
        agent.policy.load_state_dict(torch.load(args.load_model))
    else:
        print("错误: 未找到模型文件，请先训练模型或提供正确的路径")
        return
    
    print(f"开始测试，共{args.eval_episodes}个回合...")
    
    # 按任务类型分组测试
    task_types = ['copy_paste', 'find_replace', 'cursor_navigation']
    
    results = {}
    for task_type in task_types:
        print(f"\n测试任务类型: {task_type}")
        
        # 重置环境，设置特定任务
        env.reset(task_type=task_type)
        
        # 测试代理在此任务上的表现
        task_rewards = agent.test(
            num_episodes=args.eval_episodes,
            max_steps=args.max_steps,
            render=args.render
        )
        
        results[task_type] = {
            'rewards': task_rewards,
            'mean_reward': np.mean(task_rewards),
            'success_rate': sum(r > 0 for r in task_rewards) / len(task_rewards)
        }
    
    # 打印总结果
    print("\n===== 测试结果 =====")
    for task_type, result in results.items():
        print(f"{task_type}:")
        print(f"  平均奖励: {result['mean_reward']:.2f}")
        print(f"  成功率: {result['success_rate'] * 100:.2f}%")
    
    return results

def interactive_demo(args):
    """交互式演示"""
    print("初始化环境和任务生成器...")
    task_generator = TaskGenerator()
    env = TextEditingEnvWithTasks(task_generator, max_steps=args.max_steps)
    
    print("初始化代理...")
    # agent = PPOAgent(
    agent = GRPOAgent(
        env=env,
        text_embedding_dim=768,  # GPT-2隐藏维度
        hidden_dim=256,
        gpt2_path=args.gpt2_path
    )
    
    # 加载模型
    if args.load_model and os.path.exists(args.load_model):
        print(f"加载模型: {args.load_model}")
        agent.policy.load_state_dict(torch.load(args.load_model))
        model_loaded = True
    else:
        print("警告: 未找到模型文件，将使用随机策略")
        model_loaded = False
    
    # 设置任务
    obs = env.reset(task_type=args.task)
    
    print("\n===== 交互式演示 =====")
    print(f"任务类型: {args.task if args.task else '随机'}")
    print(f"初始文本:\n{obs['text']}")
    print(f"目标文本:\n{env.current_task.target_text}")
    print("\n控制模式:")
    print("1. 代理自动控制 (A键)")
    print("2. 手动控制 (方向键, Shift, Ctrl+C/V/X)")
    print("3. 切换任务 (T键)")
    print("4. 退出 (ESC键)")
    
    env.render(mode="human")
    
    # 交互循环
    running = True
    auto_mode = True  # 默认使用代理自动控制
    step = 0
    clock = pygame.time.Clock()
    
    while running and step < args.max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_a:
                    # 切换自动/手动模式
                    auto_mode = not auto_mode
                    print(f"切换到{'自动' if auto_mode else '手动'}模式")
                elif event.key == pygame.K_t:
                    # 切换任务
                    obs = env.reset(task_type=args.task)
                    step = 0
                    print(f"\n新任务:")
                    print(f"初始文本:\n{obs['text']}")
                    print(f"目标文本:\n{env.current_task.target_text}")
                elif not auto_mode:
                    # 手动控制逻辑
                    action_key_map = {
                        pygame.K_UP: 0,       # u
                        pygame.K_DOWN: 1,     # d
                        pygame.K_LEFT: 2,     # l
                        pygame.K_RIGHT: 3,    # r
                        pygame.K_LCTRL: 4,    # ctrl
                        pygame.K_RCTRL: 4,    # ctrl
                        pygame.K_LSHIFT: 5,   # shift
                        pygame.K_RSHIFT: 5,   # shift
                        pygame.K_LALT: 6,     # alt
                        pygame.K_RALT: 6,     # alt
                        pygame.K_c: 9,        # copy (需要先按Ctrl)
                        pygame.K_v: 10,       # paste (需要先按Ctrl)
                        pygame.K_x: 11,       # cut (需要先按Ctrl)
                    }
                    
                    if event.key in action_key_map:
                        action = action_key_map[event.key]
                        
                        # 特殊处理组合键
                        if action in [9, 10, 11]:  # copy, paste, cut
                            if env.key_pressed[0]:  # 检查Ctrl是否被按下
                                obs, reward, done, info = env.step(action)
                                print(f"执行: {env.action_to_name[action]}, 奖励: {reward:.4f}")
                                step += 1
                        else:
                            obs, reward, done, info = env.step(action)
                            print(f"执行: {env.action_to_name[action]}, 奖励: {reward:.4f}")
                            step += 1
                        
                        if done:
                            print(f"任务{'完成' if info.get('task_completed', False) else '失败'}")
                            running = False
        
        # 自动模式：代理控制
        if auto_mode and running:
            if model_loaded:
                # 使用训练好的代理
                action = agent.select_action(obs, training=False)
            else:
                # 随机策略
                action = env.action_space.sample()
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            action_name = env.action_to_name[action]
            print(f"代理执行: {action_name}, 奖励: {reward:.4f}")
            step += 1
            
            if done:
                print(f"任务{'完成' if info.get('task_completed', False) else '失败'}")
                time.sleep(2)  # 暂停查看结果
                # 重置环境，开始新任务
                obs = env.reset(task_type=args.task)
                step = 0
                print(f"\n新任务:")
                print(f"初始文本:\n{obs['text']}")
                print(f"目标文本:\n{env.current_task.target_text}")
        
        # 更新环境渲染
        env.render(mode="human")
        # clock.tick(5)  # 控制速度，每秒5帧
    
    env.close()
    print("演示结束")

def main():
    args = parse_args()
    
    if args.mode == 'train':
        train_agent(args)
    elif args.mode == 'test':
        test_agent(args)
    elif args.mode == 'demo':
        interactive_demo(args)
    else:
        print(f"未知模式: {args.mode}")

if __name__ == "__main__":
    main()