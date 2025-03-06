from text_editing_environment import TextEditingEnv
import random
import difflib
import re
import time
import numpy as np
import pygame

class TextEditTask:
    """文本编辑任务的基类，定义了任务的基本接口"""
    
    def __init__(self, name, max_steps=100):
        self.name = name
        self.max_steps = max_steps
        self.initial_text = ""
        self.target_text = ""
    
    def generate(self):
        """生成一个任务实例，返回初始文本和目标文本"""
        raise NotImplementedError("子类必须实现此方法")
    
    def get_reward(self, current_text, target_text, action_taken, cursor_pos, done, clipboard=""):
        """计算当前步骤的奖励"""
        raise NotImplementedError("子类必须实现此方法")
    
    def is_completed(self, current_text, target_text):
        """检查任务是否完成"""
        return current_text == target_text


class CopyPasteTask(TextEditTask):
    """复制粘贴任务：从文档中复制特定文本并粘贴到指定位置"""
    
    def __init__(self, 
                 max_steps=100, 
                 min_length=3, 
                 max_length=10, 
                 min_text_length=20, 
                 max_text_length=100):
        super().__init__("copy_paste", max_steps)
        self.min_length = min_length  # 要复制的最小文本长度
        self.max_length = max_length  # 要复制的最大文本长度
        self.min_text_length = min_text_length  # 初始文本的最小长度
        self.max_text_length = max_text_length  # 初始文本的最大长度
    
    def generate(self):
        """生成一个复制粘贴任务"""
        # 生成随机文本
        text_length = random.randint(self.min_text_length, self.max_text_length)
        words = ['lorem', 'ipsum', 'dolor', 'sit', 'amet', 'consectetur', 
                 'adipiscing', 'elit', 'sed', 'do', 'eiusmod', 'tempor', 
                 'incididunt', 'ut', 'labore', 'et', 'dolore', 'magna', 'aliqua']
        
        text_words = []
        for _ in range(text_length // 5):  # 平均每个单词5个字符
            text_words.append(random.choice(words))
        
        initial_text = ' '.join(text_words)
        
        # 选择要复制的文本片段
        start_idx = random.randint(0, len(initial_text) - self.max_length)
        copy_length = random.randint(self.min_length, min(self.max_length, len(initial_text) - start_idx))
        text_to_copy = initial_text[start_idx:start_idx + copy_length]
        
        # 选择粘贴位置
        paste_idx = random.randint(0, len(initial_text))
        
        # 创建目标文本（完成复制粘贴后的文本）
        target_text = initial_text[:paste_idx] + text_to_copy + initial_text[paste_idx:]
        
        self.initial_text = initial_text
        self.target_text = target_text
        self.text_to_copy = text_to_copy
        self.copy_start_idx = start_idx
        self.copy_end_idx = start_idx + copy_length
        self.paste_idx = paste_idx
        
        return initial_text, target_text
    
    def get_reward(self, current_text, target_text, action_taken, cursor_pos, done, clipboard=""):
        """计算奖励"""
        # 基础奖励
        if done and current_text == target_text:
            return 10.0  # 任务完成奖励
        
        # 离散奖励信号
        action_name = action_taken if isinstance(action_taken, str) else None
        reward = 0.0
        
        # 1. 光标移动到复制区域起始位置附近
        if action_name in ['l', 'r', 'u', 'd']:
            dist_to_copy_start = abs(cursor_pos[0] * 1000 + cursor_pos[1] - self.copy_start_idx)
            if dist_to_copy_start < 5:
                reward += 0.1
        
        # 2. 选中正确的文本
        if action_name == 'shift' and action_name in ['l', 'r', 'u', 'd']:
            # 检查是否正在选择要复制的文本
            selection_length = abs(cursor_pos[1] - self.copy_start_idx)
            if selection_length > 0 and selection_length <= len(self.text_to_copy):
                reward += 0.1 * (selection_length / len(self.text_to_copy))
        
        # 3. 复制操作
        if action_name == 'copy':
            # 检查剪贴板是否包含要复制的文本
            if self.text_to_copy in clipboard:
                reward += 1.0
        
        # 4. 光标移动到粘贴位置
        if action_name in ['l', 'r', 'u', 'd']:
            dist_to_paste = abs(cursor_pos[0] * 1000 + cursor_pos[1] - self.paste_idx)
            if dist_to_paste < 5:
                reward += 0.1
        
        # 5. 粘贴操作
        if action_name == 'paste':
            # 检查粘贴后的文本是否更接近目标
            similarity_before = difflib.SequenceMatcher(None, current_text, target_text).ratio()
            
            # 模拟粘贴操作后的文本
            paste_pos = cursor_pos[0] * 1000 + cursor_pos[1]
            simulated_text = current_text[:paste_pos] + clipboard + current_text[paste_pos:]
            
            similarity_after = difflib.SequenceMatcher(None, simulated_text, target_text).ratio()
            
            if similarity_after > similarity_before:
                reward += 1.0 * (similarity_after - similarity_before)
        
        # 惩罚过多的步骤
        return reward - 0.01  # 每步骤的小惩罚


class FindReplaceTask(TextEditTask):
    """查找替换任务：查找文本中所有出现的特定单词或短语并替换为另一个"""
    
    def __init__(self, max_steps=200):
        super().__init__("find_replace", max_steps)
    
    def generate(self):
        """生成一个查找替换任务"""
        # 生成随机文本，包含多个重复单词
        words = ['lorem', 'ipsum', 'dolor', 'sit', 'amet', 'consectetur', 
                 'adipiscing', 'elit', 'sed', 'do', 'eiusmod', 'tempor', 
                 'incididunt', 'ut', 'labore', 'et', 'dolore', 'magna', 'aliqua']
        
        # 选择一个要重复的单词
        word_to_replace = random.choice(words)
        replacement_word = "REPLACED"
        
        # 生成包含多个待替换单词的文本
        text_words = []
        for _ in range(40):  # 生成约40个单词
            if random.random() < 0.2:  # 20%的概率使用待替换单词
                text_words.append(word_to_replace)
            else:
                text_words.append(random.choice(words))
        
        initial_text = ' '.join(text_words)
        
        # 创建目标文本（所有出现的单词都被替换）
        target_text = initial_text.replace(word_to_replace, replacement_word)
        
        self.initial_text = initial_text
        self.target_text = target_text
        self.word_to_replace = word_to_replace
        self.replacement_word = replacement_word
        
        return initial_text, target_text
    
    def get_reward(self, current_text, target_text, action_taken, cursor_pos, done, clipboard=""):
        """计算奖励"""
        # 任务完成奖励
        if done and current_text == target_text:
            return 10.0
        
        # 进度奖励：基于已替换的单词数量
        initial_occurrences = self.initial_text.count(self.word_to_replace)
        current_occurrences = current_text.count(self.word_to_replace)
        
        if initial_occurrences > current_occurrences:
            # 有单词被替换了
            replace_progress = (initial_occurrences - current_occurrences) / initial_occurrences
            return replace_progress * 2.0 - 0.01  # 进度奖励减去小惩罚
        
        return -0.01  # 每步骤的小惩罚


class CursorNavigationTask(TextEditTask):
    """光标导航任务：将光标移动到文本中的特定位置"""
    
    def __init__(self, max_steps=50):
        super().__init__("cursor_navigation", max_steps)
    
    def generate(self):
        """生成一个光标导航任务"""
        # 生成随机文本
        paragraphs = []
        for _ in range(3):  # 生成3个段落
            words = []
            for _ in range(random.randint(20, 40)):  # 每段20-40个单词
                word_length = random.randint(3, 10)
                word = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(word_length))
                words.append(word)
            paragraphs.append(' '.join(words))
        
        initial_text = '\n\n'.join(paragraphs)
        
        # 选择目标位置
        lines = initial_text.split('\n')
        
        # 确保选择非空行
        non_empty_lines = [(i, line) for i, line in enumerate(lines) if len(line) > 0]
        if not non_empty_lines:
            # 极端情况：如果所有行都是空的，添加一个非空行
            lines[0] = "placeholder text"
            target_line = 0
            target_col = 0
        else:
            # 从非空行中随机选择一行
            line_idx, selected_line = random.choice(non_empty_lines)
            target_line = line_idx
            target_col = random.randint(0, len(selected_line) - 1)
        
        # 在目标位置插入一个特殊标记（方便任务评估）
        special_mark = "<<<TARGET>>>"
        modified_lines = lines.copy()
        modified_lines[target_line] = modified_lines[target_line][:target_col] + special_mark + modified_lines[target_line][target_col:]
        
        # 创建目标文本（包含特殊标记）
        target_text = '\n'.join(modified_lines)
        
        self.initial_text = initial_text
        self.target_text = target_text
        self.target_position = (target_line, target_col)
        
        return initial_text, target_text
    
    def get_reward(self, current_text, target_text, action_taken, cursor_pos, done, clipboard=""):
        """计算奖励"""
        # 目标完成奖励（光标到达目标位置）
        target_line, target_col = self.target_position
        current_line, current_col = cursor_pos
        
        if current_line == target_line and current_col == target_col:
            return 5.0  # 到达目标位置的奖励
        
        # 计算与目标位置的距离
        distance = abs(current_line - target_line) + abs(current_col - target_col)
        
        # 距离奖励：越接近目标位置，奖励越高
        distance_reward = 1.0 / (1.0 + distance)
        
        # 朝正确方向移动的奖励
        action_name = action_taken if isinstance(action_taken, str) else None
        direction_reward = 0.0
        
        if action_name == 'u' and target_line < current_line:
            direction_reward = 0.1
        elif action_name == 'd' and target_line > current_line:
            direction_reward = 0.1
        elif action_name == 'l' and target_col < current_col:
            direction_reward = 0.1
        elif action_name == 'r' and target_col > current_col:
            direction_reward = 0.1
        
        return distance_reward + direction_reward - 0.01


class TaskGenerator:
    """任务生成器：负责创建各种文本编辑任务"""
    
    def __init__(self):
        self.tasks = {
            'copy_paste': CopyPasteTask(),
            'find_replace': FindReplaceTask(),
            'cursor_navigation': CursorNavigationTask()
        }
    
    def generate_task(self, task_type=None):
        """生成指定类型的任务，如果未指定则随机选择"""
        if task_type is None:
            task_type = random.choice(list(self.tasks.keys()))
        
        task = self.tasks[task_type]
        initial_text, target_text = task.generate()
        
        return {
            'task_type': task_type,
            'task_object': task,
            'initial_text': initial_text,
            'target_text': target_text
        }


class TextEditingEnvWithTasks(TextEditingEnv):
    """扩展文本编辑环境，包含任务和奖励机制"""
    
    def __init__(self, task_generator, max_steps=100):
        self.task_generator = task_generator
        self.current_task = None
        self.steps_taken = 0
        self.action_history = []
        super().__init__(initial_text="", max_episode_steps=max_steps)
    
    def reset(self, task_type=None, initial_text=""):
        """重置环境并设置新任务"""
        # 检查是否是初始化阶段的调用
        if not hasattr(self, 'task_generator') or self.task_generator is None:
            # 如果task_generator不存在，这肯定是初始化阶段
            return super().reset(initial_text or "")
        
        # 初始化阶段的另一种情况：task_type为空字符串
        if task_type == "":
            # 当task_type是空字符串时，也认为是初始化阶段
            return super().reset(initial_text or "")
        
        # 正常的任务生成流程
        task_info = self.task_generator.generate_task(task_type)
        self.current_task = task_info['task_object']
        task_initial_text = task_info['initial_text']
        
        # 重置环境状态
        obs = super().reset(task_initial_text)
        self.steps_taken = 0
        self.action_history = []
        
        return obs
    
    def step(self, action):
        """执行一个步骤并计算任务相关的奖励"""
        # 执行动作
        action_name = self.action_to_name[action]
        self.action_history.append(action_name)
        
        obs, _, done, info = super().step(action)
        self.steps_taken += 1
        
        # 检查任务是否完成
        current_text = obs['text']
        task_completed = self.current_task.is_completed(current_text, self.current_task.target_text)
        
        # 计算任务相关的奖励
        reward = self.current_task.get_reward(
            current_text, 
            self.current_task.target_text,
            action_name,
            self.cursor,
            done or task_completed,
            self.clipboard  # Pass the clipboard content
        )
        
        # 任务完成或达到最大步数则结束
        done = done or task_completed or self.steps_taken >= self.current_task.max_steps
        
        # 添加任务相关信息
        info['task_completed'] = task_completed
        info['target_text'] = self.current_task.target_text
        
        return obs, reward, done, info


# 示例用法
def demo_text_editing_with_tasks():
    """演示带任务的文本编辑环境"""
    try:
        pygame.init()
    except Exception as e:
        print(f"Pygame初始化失败: {e}")
        return
        
    task_generator = TaskGenerator()
    env = TextEditingEnvWithTasks(task_generator, max_steps=50)
    
    # 尝试不同类型的任务
    for task_type in ['copy_paste', 'find_replace', 'cursor_navigation']:
        print(f"\n=== 演示任务: {task_type} ===")
        obs = env.reset(task_type)
        
        print(f"初始文本:\n{obs['text']}")
        print(f"目标文本:\n{env.current_task.target_text}")
        
        # 随机执行一些动作
        total_reward = 0
        for _ in range(10):
            action = env.action_space.sample()
            action_name = env.action_to_name[action]
            obs, reward, done, info = env.step(action)
            
            print(f"执行动作: {action_name}, 奖励: {reward:.4f}")
            total_reward += reward
            
            if done:
                break
        
        print(f"任务{'已完成' if info['task_completed'] else '未完成'}")
        print(f"总奖励: {total_reward:.4f}")
        print(f"当前文本:\n{obs['text']}")
    
    env.close()
    pygame.quit()


if __name__ == "__main__":
    try:
        # 运行演示
        demo_text_editing_with_tasks()
        
        # 也可以尝试手动控制
        print("\n=== 手动控制演示 ===")
        pygame.init()
        task_generator = TaskGenerator()
        env = TextEditingEnvWithTasks(task_generator, max_steps=50)
        
        obs = env.reset('copy_paste')
        print(f"初始文本:\n{obs['text']}")
        print(f"目标文本:\n{env.current_task.target_text}")
        print("请按以下键控制：")
        print("方向键：移动光标")
        print("Shift+方向键：选择文本")
        print("Ctrl+C：复制")
        print("Ctrl+V：粘贴")
        print("Ctrl+X：剪切")
        print("ESC：退出")
        
        try:
            env.render(mode="human")
            
            running = True
            clock = pygame.time.Clock()
            
            while running:
                pygame.event.pump()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                
                clock.tick(30)
                
        except Exception as e:
            print(f"渲染错误: {e}")
        finally:
            env.close()
            pygame.quit()
    except Exception as e:
        print(f"程序运行错误: {e}")
        pygame.quit()