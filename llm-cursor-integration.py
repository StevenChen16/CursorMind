import torch
import numpy as np
import os
import sys
import argparse
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 确保可以导入我们的模块
sys.path.append('.')

# 导入我们的模块
from text_editing_environment import TextEditingEnv
from cursor_control_agent import PPOAgent
from task_generator_rewards import TaskGenerator, TextEditingEnvWithTasks


class LLMWithCursorControl:
    """
    集成LLM和光标控制代理的类
    
    该类将GPT-2语言模型与训练好的光标控制代理结合，
    允许LLM生成文本并使用光标代理操作和编辑文本。
    """
    
    def __init__(self, 
                 gpt2_path="E:\\ML\\example\\nlp\\GPT-2", 
                 cursor_model_path="cursor_control_agent.pt",
                 max_steps=100):
        """
        初始化LLM和光标控制代理
        
        参数:
            gpt2_path: GPT-2模型和分词器的路径
            cursor_model_path: 训练好的光标控制模型的路径
            max_steps: 每个任务的最大步骤数
        """
        # 初始化GPT-2模型
        print("加载GPT-2模型...")
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)
            self.lm_model = GPT2LMHeadModel.from_pretrained(gpt2_path)
            print("GPT-2模型加载成功")
        except Exception as e:
            print(f"加载GPT-2模型失败: {e}")
            raise
        
        # 初始化环境和任务生成器
        print("初始化文本编辑环境...")
        self.task_generator = TaskGenerator()
        self.env = TextEditingEnvWithTasks(self.task_generator, max_steps=max_steps)
        
        # 初始化光标控制代理
        print("初始化光标控制代理...")
        self.cursor_agent = PPOAgent(
            env=self.env,
            text_embedding_dim=768,  # GPT-2隐藏维度
            hidden_dim=256,
            gpt2_path=gpt2_path
        )
        
        # 加载训练好的光标控制模型
        if os.path.exists(cursor_model_path):
            print(f"加载光标控制模型: {cursor_model_path}")
            self.cursor_agent.policy.load_state_dict(torch.load(cursor_model_path))
        else:
            print(f"警告: 光标控制模型不存在，将使用随机策略: {cursor_model_path}")
    
    def generate_text(self, prompt, max_length=200, num_return_sequences=1):
        """使用GPT-2生成文本"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 生成文本
        outputs = self.lm_model.generate(
            inputs.input_ids, 
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.8
        )
        
        # 解码生成的文本
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def perform_edit_task(self, task_type=None, initial_text=None, target_text=None, render=True):
        """执行编辑任务"""
        # 重置环境或使用给定的文本
        if initial_text is not None and target_text is not None:
            # 使用自定义文本
            obs = self.env.reset(task_type)
            self.env.current_task.initial_text = initial_text
            self.env.current_task.target_text = target_text
            
            # 手动重置环境状态
            self.env.reset(initial_text)
        else:
            # 使用任务生成器的默认任务
            obs = self.env.reset(task_type)
        
        print(f"\n===== 执行任务: {self.env.current_task.name} =====")
        print(f"初始文本:\n{obs['text']}")
        print(f"目标文本:\n{self.env.current_task.target_text}")
        
        # 使用光标代理执行任务
        done = False
        step = 0
        total_reward = 0
        
        while not done and step < self.env.current_task.max_steps:
            # 选择动作
            action = self.cursor_agent.select_action(obs, training=False)
            
            # 执行动作
            next_obs, reward, done, info = self.env.step(action)
            
            # 显示执行的动作
            action_name = self.env.action_to_name[action]
            print(f"步骤 {step+1}: 执行动作 {action_name}, 奖励: {reward:.4f}")
            
            # 更新状态
            obs = next_obs
            total_reward += reward
            step += 1
            
            # 可选渲染
            if render:
                self.env.render(mode="human")
                time.sleep(0.2)  # 减慢速度以便查看
        
        # 显示结果
        print(f"\n任务{'完成' if info.get('task_completed', False) else '未完成'}")
        print(f"总步数: {step}")
        print(f"总奖励: {total_reward:.4f}")
        print(f"最终文本:\n{obs['text']}")
        
        return {
            'completed': info.get('task_completed', False),
            'steps': step,
            'reward': total_reward,
            'final_text': obs['text']
        }
    
    def interactive_session(self):
        """交互式会话，结合LLM生成和光标操作"""
        print("\n===== LLM与光标控制交互式会话 =====")
        print("1. 生成文本 (输入'generate:prompt')")
        print("2. 执行编辑任务 (输入'edit:task_type')")
        print("3. 自定义编辑任务 (输入'custom')")
        print("4. 退出 (输入'exit')")
        
        task_types = ['copy_paste', 'find_replace', 'cursor_navigation']
        
        while True:
            command = input("\n> ")
            
            if command.lower() == 'exit':
                break
            
            elif command.lower().startswith('generate:'):
                prompt = command[9:].strip()
                if not prompt:
                    prompt = input("请输入提示词: ")
                
                print("生成文本中...")
                generated_texts = self.generate_text(prompt, num_return_sequences=3)
                
                print("\n生成的文本:")
                for i, text in enumerate(generated_texts):
                    print(f"\n--- 文本 {i+1} ---\n{text}")
                
                # 询问是否将生成的文本用于编辑任务
                use_for_edit = input("\n是否使用生成的文本进行编辑任务? (y/n): ")
                if use_for_edit.lower() == 'y':
                    text_idx = int(input(f"选择要使用的文本 (1-{len(generated_texts)}): ")) - 1
                    if 0 <= text_idx < len(generated_texts):
                        initial_text = generated_texts[text_idx]
                        
                        # 询问编辑任务类型
                        print("\n选择编辑任务类型:")
                        for i, task in enumerate(task_types):
                            print(f"{i+1}. {task}")
                        
                        task_idx = int(input(f"选择任务类型 (1-{len(task_types)}): ")) - 1
                        if 0 <= task_idx < len(task_types):
                            # 根据任务类型生成目标文本
                            task_type = task_types[task_idx]
                            
                            # 创建一个临时任务
                            task = self.task_generator.tasks[task_type]
                            task.initial_text = initial_text
                            _, target_text = task.generate()
                            
                            # 执行编辑任务
                            self.perform_edit_task(
                                task_type=task_type,
                                initial_text=initial_text,
                                target_text=target_text
                            )
            
            elif command.lower().startswith('edit:'):
                task_type = command[5:].strip()
                if not task_type or task_type not in task_types:
                    print(f"无效的任务类型。可用任务: {', '.join(task_types)}")
                    continue
                
                self.perform_edit_task(task_type=task_type)
            
            elif command.lower() == 'custom':
                print("创建自定义编辑任务")
                
                # 输入初始文本
                print("\n请输入初始文本 (输入空行结束):")
                initial_text_lines = []
                while True:
                    line = input()
                    if line == "":
                        break
                    initial_text_lines.append(line)
                
                initial_text = "\n".join(initial_text_lines)
                
                # 输入目标文本
                print("\n请输入目标文本 (输入空行结束):")
                target_text_lines = []
                while True:
                    line = input()
                    if line == "":
                        break
                    target_text_lines.append(line)
                
                target_text = "\n".join(target_text_lines)
                
                # 选择任务类型
                print("\n选择编辑任务类型:")
                for i, task in enumerate(task_types):
                    print(f"{i+1}. {task}")
                
                task_idx = int(input(f"选择任务类型 (1-{len(task_types)}): ")) - 1
                if 0 <= task_idx < len(task_types):
                    # 执行自定义编辑任务
                    self.perform_edit_task(
                        task_type=task_types[task_idx],
                        initial_text=initial_text,
                        target_text=target_text
                    )
            
            else:
                print("未知命令。可用命令: generate:prompt, edit:task_type, custom, exit")
    
    def close(self):
        """关闭环境和资源"""
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description='LLM与光标控制集成')
    parser.add_argument('--gpt2-path', type=str, default='E:\\ML\\example\\nlp\\GPT-2',
                        help='GPT-2模型和分词器的路径')
    parser.add_argument('--cursor-model', type=str, default='cursor_control_agent.pt',
                        help='训练好的光标控制模型的路径')
    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['interactive', 'demo'],
                        help='运行模式: interactive 或 demo')
    parser.add_argument('--task', type=str, default=None,
                        choices=[None, 'copy_paste', 'find_replace', 'cursor_navigation'],
                        help='在demo模式下使用的任务类型')
    
    args = parser.parse_args()
    
    # 初始化集成系统
    system = LLMWithCursorControl(
        gpt2_path=args.gpt2_path,
        cursor_model_path=args.cursor_model
    )
    
    try:
        if args.mode == 'interactive':
            system.interactive_session()
        elif args.mode == 'demo':
            system.perform_edit_task(task_type=args.task)
    finally:
        system.close()


if __name__ == "__main__":
    main()