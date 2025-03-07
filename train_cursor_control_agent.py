import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import pygame
import time
import sys
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

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
    parser.add_argument('--log-dir', type=str, default='runs', help='TensorBoard日志目录')
    return parser.parse_args()

def train_agent(args):
    """训练光标控制代理"""
    # 创建TensorBoard日志目录
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(args.log_dir, f"{current_time}_phase{args.training_phase}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # 记录超参数
    hparams = {
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'clip_ratio': args.clip_ratio,
        'update_frequency': args.update_frequency,
        'training_phase': args.training_phase,
        'max_steps': args.max_steps,
        'batch_size': args.batch_size
    }
    writer.add_hparams(hparams, {})
    
    # 初始化环境和任务生成器
    task_generator = TaskGenerator()
    env = TextEditingEnvWithTasks(task_generator, max_steps=args.max_steps)
    
    # 设置初始训练阶段
    task_generator.set_training_phase(args.training_phase)
    
    # 初始化代理
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
        agent.policy.load_state_dict(torch.load(args.load_model))
        writer.add_text('training_info', f'已加载模型: {args.load_model}')
    
    # 跟踪训练统计信息
    rewards = []
    episode_lengths = []
    task_completion_rates = []
    
    # 创建进度条
    pbar = tqdm(total=args.num_episodes, desc="训练进度")
    pbar.set_postfix({'reward': 0.0, 'length': 0, 'success': 0.0})
    
    # 训练循环
    global_step = 0
    update_counter = 0
    
    for episode in range(args.num_episodes):
        # 检查是否需要切换训练阶段
        if args.training_phase == 1 and episode == args.phase_switch_episode:
            task_generator.set_training_phase(2)
            writer.add_text('training_info', f'切换到训练阶段2（完整功能）', global_step)
            
            # 保存阶段1完成时的检查点
            phase1_model_path = args.save_model.replace('.pt', '_phase1.pt')
            torch.save(agent.policy.state_dict(), phase1_model_path)
            writer.add_text('training_info', f'保存阶段1模型到: {phase1_model_path}', global_step)
        
        # 重置环境
        obs = env.reset()
        episode_reward = 0
        done = False
        step = 0
        task_type = env.current_task.name if hasattr(env, 'current_task') else "unknown"
        
        # 记录任务信息
        writer.add_text(f'Episode_{episode}/task', f'任务类型: {task_type}', global_step)
        writer.add_text(f'Episode_{episode}/initial_text', obs['text'], global_step)
        if hasattr(env, 'current_task') and hasattr(env.current_task, 'target_text'):
            writer.add_text(f'Episode_{episode}/target_text', env.current_task.target_text, global_step)
        
        # 回合内循环
        while not done and step < args.max_steps:
            # 选择动作
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            
            # 存储奖励和结束状态
            agent.store_reward(reward, done)
            
            # 更新状态和累积奖励
            obs = next_obs
            episode_reward += reward
            step += 1
            global_step += 1
            
            # 记录每步的奖励到TensorBoard
            writer.add_scalar('step/reward', reward, global_step)
            writer.add_scalar('step/cumulative_reward', episode_reward, global_step)
            
            # 可选渲染
            if args.render:
                env.render()
        
        # 记录回合结果
        rewards.append(episode_reward)
        episode_lengths.append(step)
        task_completed = info.get('task_completed', False)
        task_completion_rates.append(1.0 if task_completed else 0.0)
        
        # 记录回合统计信息到TensorBoard
        writer.add_scalar('episode/reward', episode_reward, episode)
        writer.add_scalar('episode/length', step, episode)
        writer.add_scalar('episode/task_completed', 1.0 if task_completed else 0.0, episode)
        
        # 如果有固定时间窗口的平均，也记录
        window_size = min(50, len(rewards))
        if window_size > 0:
            recent_rewards = rewards[-window_size:]
            recent_lengths = episode_lengths[-window_size:]
            recent_completions = task_completion_rates[-window_size:]
            
            writer.add_scalar('episode/avg_reward_50', np.mean(recent_rewards), episode)
            writer.add_scalar('episode/avg_length_50', np.mean(recent_lengths), episode)
            writer.add_scalar('episode/completion_rate_50', np.mean(recent_completions), episode)
        
        # 定期更新策略
        update_counter += 1
        if update_counter >= args.update_frequency:
            agent.update()
            update_counter = 0
            
            # 记录更新信息
            writer.add_scalar('training/updates', episode // args.update_frequency, episode)
            
            # 对于可能的模型参数，记录其梯度信息和权重分布
            for name, param in agent.policy.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'gradients/{name}', param.grad, episode)
                writer.add_histogram(f'weights/{name}', param, episode)
        
        # 更新进度条
        # 计算最近的成功率和平均奖励
        recent_success_rate = np.mean(task_completion_rates[-window_size:]) if window_size > 0 else 0
        recent_avg_reward = np.mean(recent_rewards) if window_size > 0 else episode_reward
        recent_avg_length = np.mean(recent_lengths) if window_size > 0 else step
        
        # 更新进度条的附加信息
        pbar.set_postfix({
            'reward': f'{recent_avg_reward:.2f}', 
            'length': f'{recent_avg_length:.1f}', 
            'success': f'{recent_success_rate:.2f}'
        })
        pbar.update(1)
    
    # 关闭进度条
    pbar.close()
    
    # 保存最终模型
    torch.save(agent.policy.state_dict(), args.save_model)
    writer.add_text('training_info', f'保存模型到: {args.save_model}', global_step)
    
    # 绘制学习曲线并保存
    plt.figure(figsize=(10, 5))
    plt.plot(np.convolve(rewards, np.ones(args.update_frequency)/args.update_frequency, mode='valid'))
    plt.title('学习曲线 (移动平均)')
    plt.xlabel('回合')
    plt.ylabel('奖励')
    curve_path = os.path.join(log_dir, 'learning_curve.png')
    plt.savefig(curve_path)
    
    # 添加图片到TensorBoard
    from PIL import Image
    import io
    img = Image.open(curve_path)
    img_arr = np.array(img)
    writer.add_image('learning_curve', img_arr, dataformats='HWC')
    
    # 关闭TensorBoard写入器
    writer.close()
    
    return agent


def test_agent(args):
    """测试代理性能"""
    # 创建TensorBoard日志目录
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(args.log_dir, f"test_{current_time}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # 初始化环境和任务生成器
    task_generator = TaskGenerator()
    env = TextEditingEnvWithTasks(task_generator, max_steps=args.max_steps)
    
    # 初始化代理
    agent = GRPOAgent(
        env=env,
        text_embedding_dim=768,
        hidden_dim=256,
        gpt2_path=args.gpt2_path
    )
    
    # 加载模型
    if args.load_model and os.path.exists(args.load_model):
        agent.policy.load_state_dict(torch.load(args.load_model))
        writer.add_text('test_info', f'已加载模型: {args.load_model}')
    else:
        writer.add_text('test_info', '错误: 未找到模型文件，请先训练模型或提供正确的路径')
        writer.close()
        return
    
    # 按任务类型分组测试
    task_types = ['copy_paste', 'find_replace', 'cursor_navigation']
    results = {}
    
    for task_type in task_types:
        writer.add_text('test_info', f'测试任务类型: {task_type}')
        
        # 重置环境，设置特定任务
        env.reset(task_type=task_type)
        
        # 创建进度条
        pbar = tqdm(total=args.eval_episodes, desc=f"测试 {task_type}")
        
        # 测试记录
        task_rewards = []
        task_steps = []
        task_completions = []
        
        for episode in range(args.eval_episodes):
            # 重置环境
            obs = env.reset(task_type=task_type)
            episode_reward = 0
            done = False
            step = 0
            
            # 回合内循环
            while not done and step < args.max_steps:
                action = agent.select_action(obs, training=False)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                step += 1
                
                if args.render:
                    env.render()
                    time.sleep(0.1)
            
            # 记录结果
            task_rewards.append(episode_reward)
            task_steps.append(step)
            task_completions.append(1.0 if info.get('task_completed', False) else 0.0)
            
            # 记录到TensorBoard
            writer.add_scalar(f'test/{task_type}/reward', episode_reward, episode)
            writer.add_scalar(f'test/{task_type}/steps', step, episode)
            writer.add_scalar(f'test/{task_type}/completed', 1.0 if info.get('task_completed', False) else 0.0, episode)
            
            # 更新进度条
            pbar.set_postfix({
                'reward': f'{episode_reward:.2f}',
                'steps': step,
                'completed': info.get('task_completed', False)
            })
            pbar.update(1)
        
        # 关闭进度条
        pbar.close()
        
        # 计算并记录任务统计信息
        mean_reward = np.mean(task_rewards)
        success_rate = np.mean(task_completions)
        
        results[task_type] = {
            'rewards': task_rewards,
            'mean_reward': mean_reward,
            'success_rate': success_rate
        }
        
        # 记录汇总指标到TensorBoard
        writer.add_scalar(f'test_summary/{task_type}/mean_reward', mean_reward, 0)
        writer.add_scalar(f'test_summary/{task_type}/success_rate', success_rate, 0)
    
    # 记录所有任务的平均表现
    all_rewards = [r for task in results.values() for r in task['rewards']]
    all_success = sum(results[t]['success_rate'] * args.eval_episodes for t in task_types) / (len(task_types) * args.eval_episodes)
    writer.add_scalar('test_summary/all/mean_reward', np.mean(all_rewards), 0)
    writer.add_scalar('test_summary/all/success_rate', all_success, 0)
    
    # 关闭TensorBoard写入器
    writer.close()
    
    return results


def interactive_demo(args):
    """交互式演示"""
    # 简单初始化，因为交互式演示不需要TensorBoard
    task_generator = TaskGenerator()
    env = TextEditingEnvWithTasks(task_generator, max_steps=args.max_steps)
    
    agent = GRPOAgent(
        env=env,
        text_embedding_dim=768,
        hidden_dim=256,
        gpt2_path=args.gpt2_path
    )
    
    # 加载模型
    model_loaded = False
    if args.load_model and os.path.exists(args.load_model):
        agent.policy.load_state_dict(torch.load(args.load_model))
        model_loaded = True
    
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
    
    # 添加一个进度条来显示当前步骤和奖励
    pbar = tqdm(total=args.max_steps, desc="演示进度")
    pbar.set_postfix({'mode': 'auto', 'reward': 0.0})
    
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
                    pbar.set_postfix({'mode': 'auto' if auto_mode else 'manual', 'reward': pbar.postfix['reward']})
                elif event.key == pygame.K_t:
                    # 切换任务
                    obs = env.reset(task_type=args.task)
                    step = 0
                    pbar.reset()
                    pbar.set_postfix({'mode': 'auto' if auto_mode else 'manual', 'reward': 0.0})
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
                                pbar.set_postfix({'mode': 'manual', 'reward': f'{reward:.4f}'})
                                step += 1
                                pbar.update(1)
                        else:
                            obs, reward, done, info = env.step(action)
                            pbar.set_postfix({'mode': 'manual', 'reward': f'{reward:.4f}'})
                            step += 1
                            pbar.update(1)
                        
                        if done:
                            pbar.close()
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
            pbar.set_postfix({'mode': 'auto', 'reward': f'{reward:.4f}'})
            step += 1
            pbar.update(1)
            
            if done:
                pbar.close()
                time.sleep(2)  # 暂停查看结果
                # 重置环境，开始新任务
                obs = env.reset(task_type=args.task)
                step = 0
                pbar = tqdm(total=args.max_steps, desc="演示进度")
                pbar.set_postfix({'mode': 'auto', 'reward': 0.0})
        
        # 更新环境渲染
        env.render(mode="human")
        clock.tick(5)  # 控制速度，每秒5帧
    
    # 关闭进度条和环境
    if pbar is not None:
        pbar.close()
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