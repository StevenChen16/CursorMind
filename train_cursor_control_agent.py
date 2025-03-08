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
import json

# 确保可以导入我们的模块
sys.path.append('.')

# 导入我们的模块
from cursor_control_agent import PPOAgent, GRPOAgent
from task_generator_rewards import TaskGenerator, TextEditingEnvWithTasks, TextEditTask

# 定义课程学习中各个阶段的任务类
class Stage1SingleDirectionTask(TextEditTask):
    """阶段1: 单向光标移动 - 只学习向一个方向移动（如只向右）"""
    
    def __init__(self, max_steps=20, direction='right'):
        super().__init__("stage1_single_direction", max_steps)
        self.direction = direction  # 'up', 'down', 'left', 'right'
        self.direction_to_action = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        
    def generate(self):
        """生成一个简单的任务，要求光标只向一个方向移动"""
        text = "简单的测试文本，用于学习单向移动。"
        self.initial_text = text
        
        # 目标是向指定方向移动几步
        self.target_steps = np.random.randint(3, 8)  # 移动3-7步
        
        # 对于不同方向，设置不同的起始点和目标点
        if self.direction == 'right':
            self.start_position = (0, 0)  # 左上角
            self.target_position = (0, self.target_steps)  # 向右移动
        elif self.direction == 'down':
            self.start_position = (0, 0)  # 左上角
            self.target_position = (self.target_steps, 0)  # 向下移动
        elif self.direction == 'left':
            self.start_position = (0, self.target_steps)  # 右侧开始
            self.target_position = (0, 0)  # 向左移动到左边
        elif self.direction == 'up':
            self.start_position = (self.target_steps, 0)  # 下方开始
            self.target_position = (0, 0)  # 向上移动到顶部
            
        # 目标文本只是为了任务接口保持一致，实际不使用
        self.target_text = text
        
        return self.initial_text, self.target_text
        
    def get_reward(self, current_text, target_text, action_taken, cursor_pos, done, clipboard=""):
        """简单的方向奖励"""
        # 初始化状态跟踪
        if not hasattr(self, 'prev_cursor_pos'):
            self.prev_cursor_pos = cursor_pos
            
        reward = -0.001  # 微小的时间惩罚
        
        # 检查动作是否为正确的方向键
        expected_action = self.direction_to_action[self.direction]
        if action_taken == expected_action:
            reward += 0.1  # 正确方向的奖励
            
            # 如果光标朝目标方向移动了，给予更多奖励
            if self.direction == 'right' and cursor_pos[1] > self.prev_cursor_pos[1]:
                reward += 0.2
            elif self.direction == 'left' and cursor_pos[1] < self.prev_cursor_pos[1]:
                reward += 0.2
            elif self.direction == 'down' and cursor_pos[0] > self.prev_cursor_pos[0]:
                reward += 0.2
            elif self.direction == 'up' and cursor_pos[0] < self.prev_cursor_pos[0]:
                reward += 0.2
                
        # 如果按了错误方向键，给予惩罚
        elif action_taken in [0, 1, 2, 3] and action_taken != expected_action:
            reward -= 0.1
            
        # 检查是否到达目标位置
        target_row, target_col = self.target_position
        if cursor_pos[0] == target_row and cursor_pos[1] == target_col:
            reward += 1.0  # 到达目标位置的大奖励
            
        # 更新状态
        self.prev_cursor_pos = cursor_pos
        
        return reward
        
    def is_completed(self, current_text, target_text):
        """检查任务是否完成 - 当光标到达目标位置时完成"""
        cursor_pos = self.prev_cursor_pos if hasattr(self, 'prev_cursor_pos') else (0, 0)
        target_row, target_col = self.target_position
        return cursor_pos[0] == target_row and cursor_pos[1] == target_col


class Stage2BasicNavTask(TextEditTask):
    """阶段2: 基础光标导航 - 学习基本的上下左右移动"""
    
    def __init__(self, max_steps=30):
        super().__init__("stage2_basic_nav", max_steps)
        
    def generate(self):
        """生成一个基础导航任务，随机目标位置"""
        # 创建一个简单的多行文本
        text = "这是第一行。\n这是第二行。\n这是第三行。\n这是第四行。"
        self.initial_text = text
        
        # 文本行数和每行长度
        lines = text.split('\n')
        self.num_rows = len(lines)
        self.max_col = max(len(line) for line in lines)
        
        # 随机选择起始和目标位置
        self.start_position = (np.random.randint(0, self.num_rows), 
                               np.random.randint(0, self.max_col))
        
        # 确保目标位置与起始位置不同，并且距离适中
        while True:
            self.target_position = (np.random.randint(0, self.num_rows), 
                                    np.random.randint(0, self.max_col))
            # 计算曼哈顿距离
            distance = abs(self.target_position[0] - self.start_position[0]) + \
                      abs(self.target_position[1] - self.start_position[1])
            if distance >= 3 and distance <= 6:  # 适中的距离
                break
                
        self.target_text = text  # 目标文本相同，我们只关心光标位置
        
        return self.initial_text, self.target_text
        
    def get_reward(self, current_text, target_text, action_taken, cursor_pos, done, clipboard=""):
        """基础导航奖励，基于到目标的距离"""
        # 初始化状态跟踪
        if not hasattr(self, 'prev_cursor_pos'):
            self.prev_cursor_pos = cursor_pos
        if not hasattr(self, 'prev_distance'):
            # 计算到目标的曼哈顿距离
            self.prev_distance = abs(cursor_pos[0] - self.target_position[0]) + \
                                abs(cursor_pos[1] - self.target_position[1])
            
        reward = -0.001  # 微小的时间惩罚
        
        # 计算当前到目标的距离
        current_distance = abs(cursor_pos[0] - self.target_position[0]) + \
                          abs(cursor_pos[1] - self.target_position[1])
        
        # 如果距离减少，给予奖励
        if current_distance < self.prev_distance:
            reward += 0.2 * (self.prev_distance - current_distance)  # 距离减少的比例奖励
        elif current_distance > self.prev_distance:
            reward -= 0.1  # 轻微惩罚距离增加
            
        # 检查是否到达目标位置
        if current_distance == 0:
            reward += 2.0  # 到达目标位置的大奖励
            
        # 鼓励使用正确的方向键
        if action_taken in [0, 1, 2, 3]:  # 方向键
            # 判断按键是否往正确方向移动
            correct_direction = False
            
            if action_taken == 0 and self.target_position[0] < cursor_pos[0]:  # 向上且目标在上方
                correct_direction = True
            elif action_taken == 1 and self.target_position[0] > cursor_pos[0]:  # 向下且目标在下方
                correct_direction = True
            elif action_taken == 2 and self.target_position[1] < cursor_pos[1]:  # 向左且目标在左边
                correct_direction = True
            elif action_taken == 3 and self.target_position[1] > cursor_pos[1]:  # 向右且目标在右边
                correct_direction = True
                
            if correct_direction:
                reward += 0.1
                
        # 更新状态
        self.prev_cursor_pos = cursor_pos
        self.prev_distance = current_distance
        
        return reward
        
    def is_completed(self, current_text, target_text):
        """当光标到达目标位置时任务完成"""
        cursor_pos = self.prev_cursor_pos if hasattr(self, 'prev_cursor_pos') else (0, 0)
        target_row, target_col = self.target_position
        return cursor_pos[0] == target_row and cursor_pos[1] == target_col


# 更多阶段的任务类实现...（阶段3-10）
class Stage3TargetNavTask(TextEditTask):
    """阶段3: 目标导航 - 将光标移动到特定位置（短距离）"""
    
    def __init__(self, max_steps=40):
        super().__init__("stage3_target_nav", max_steps)
        
    def generate(self):
        """生成一个任务，要求光标移动到特定位置"""
        # 创建一个包含明显目标标记的文本
        text = "普通文本开始\n这里包含 [目标位置] 需要导航到此处\n普通文本结束。"
        self.initial_text = text
        
        # 找到目标标记的位置
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if '[目标位置]' in line:
                target_row = i
                target_col = line.find('[目标位置]')
                break
        
        self.target_position = (target_row, target_col)
        self.target_text = text  # 目标文本相同
        
        return self.initial_text, self.target_text
        
    def get_reward(self, current_text, target_text, action_taken, cursor_pos, done, clipboard=""):
        """基于到标记位置距离的奖励"""
        # 初始化状态跟踪
        if not hasattr(self, 'prev_cursor_pos'):
            self.prev_cursor_pos = cursor_pos
        if not hasattr(self, 'prev_distance'):
            self.prev_distance = abs(cursor_pos[0] - self.target_position[0]) + \
                                abs(cursor_pos[1] - self.target_position[1])
            
        reward = -0.001  # 微小的时间惩罚
        
        # 计算当前到目标的距离
        current_distance = abs(cursor_pos[0] - self.target_position[0]) + \
                          abs(cursor_pos[1] - self.target_position[1])
        
        # 如果距离减少，给予奖励
        if current_distance < self.prev_distance:
            progress = (self.prev_distance - current_distance) / self.prev_distance
            reward += 0.3 * progress  # 距离减少的比例奖励
        elif current_distance > self.prev_distance:
            reward -= 0.05  # 轻微惩罚距离增加
            
        # 接近目标的奖励阶梯
        if current_distance <= 5 and self.prev_distance > 5:
            reward += 0.2  # 接近目标
        if current_distance <= 2 and self.prev_distance > 2:
            reward += 0.3  # 非常接近目标
            
        # 到达目标的奖励
        if current_distance == 0:
            reward += 2.0  # 到达目标位置的大奖励
            
        # 更新状态
        self.prev_cursor_pos = cursor_pos
        self.prev_distance = current_distance
        
        return reward
        
    def is_completed(self, current_text, target_text):
        """当光标到达目标位置时任务完成"""
        cursor_pos = self.prev_cursor_pos if hasattr(self, 'prev_cursor_pos') else (0, 0)
        return cursor_pos[0] == self.target_position[0] and cursor_pos[1] == self.target_position[1]


class Stage4ComplexNavTask(TextEditTask):
    """阶段4: 复杂导航 - 处理更远距离、更复杂路径的导航"""
    
    def __init__(self, max_steps=60):
        super().__init__("stage4_complex_nav", max_steps)
        
    def generate(self):
        """生成一个复杂导航任务，需要穿过多行文本"""
        # 创建一个更长的文本，有多个段落
        paragraphs = [
            "这是第一段文本，包含了一些内容。",
            "这是第二段文本，有不同的长度。这行更长一些。",
            "第三段比较短。",
            "第四段又是一个比较长的段落，包含了更多的内容和文字。",
            "最后一段包含 [目标] 位置标记。"
        ]
        text = "\n\n".join(paragraphs)
        self.initial_text = text
        
        # 找到目标标记的位置
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if '[目标]' in line:
                target_row = i
                target_col = line.find('[目标]')
                break
        
        self.target_position = (target_row, target_col)
        self.start_position = (0, 0)  # 始终从文本开头开始
        self.target_text = text  # 目标文本相同
        
        return self.initial_text, self.target_text
        
    def get_reward(self, current_text, target_text, action_taken, cursor_pos, done, clipboard=""):
        """复杂导航的奖励函数"""
        # 初始化状态跟踪
        if not hasattr(self, 'prev_cursor_pos'):
            self.prev_cursor_pos = cursor_pos
        if not hasattr(self, 'prev_distance'):
            self.prev_distance = abs(cursor_pos[0] - self.target_position[0]) + \
                                abs(cursor_pos[1] - self.target_position[1])
        if not hasattr(self, 'initial_distance'):
            self.initial_distance = self.prev_distance
            
        reward = -0.001  # 微小的时间惩罚
        
        # 计算当前到目标的距离
        current_distance = abs(cursor_pos[0] - self.target_position[0]) + \
                          abs(cursor_pos[1] - self.target_position[1])
        
        # 距离减少奖励（非线性，距离越小奖励越大）
        if current_distance < self.prev_distance:
            # 计算进度占总距离的百分比
            progress = (self.prev_distance - current_distance) / self.initial_distance
            # 使用非线性函数增加接近目标时的奖励
            reward += 0.4 * progress
        elif current_distance > self.prev_distance:
            # 距离增加惩罚，但不要太严厉，允许探索
            reward -= 0.03
            
        # 里程碑奖励
        distance_ratio = current_distance / self.initial_distance
        if distance_ratio < 0.75 and hasattr(self, 'milestone_75') and not self.milestone_75:
            reward += 0.2
            self.milestone_75 = True
        elif distance_ratio < 0.5 and hasattr(self, 'milestone_50') and not self.milestone_50:
            reward += 0.3
            self.milestone_50 = True
        elif distance_ratio < 0.25 and hasattr(self, 'milestone_25') and not self.milestone_25:
            reward += 0.5
            self.milestone_25 = True
            
        # 初始化里程碑标记
        if not hasattr(self, 'milestone_75'):
            self.milestone_75 = False
        if not hasattr(self, 'milestone_50'):
            self.milestone_50 = False
        if not hasattr(self, 'milestone_25'):
            self.milestone_25 = False
            
        # 到达目标奖励
        if current_distance == 0:
            reward += 3.0  # 更大的奖励，因为任务更复杂
            
        # 更新状态
        self.prev_cursor_pos = cursor_pos
        self.prev_distance = current_distance
        
        return reward
        
    def is_completed(self, current_text, target_text):
        """当光标到达目标位置时任务完成"""
        cursor_pos = self.prev_cursor_pos if hasattr(self, 'prev_cursor_pos') else (0, 0)
        return cursor_pos[0] == self.target_position[0] and cursor_pos[1] == self.target_position[1]


class Stage5TextSelectionTask(TextEditTask):
    """阶段5: 基础文本选择 - 学习使用Shift键选择文本"""
    
    def __init__(self, max_steps=50):
        super().__init__("stage5_text_selection", max_steps)
        
    def generate(self):
        """生成一个文本选择任务"""
        # 创建一个包含明显目标文本的内容
        text = "这是普通文本。[目标文本在这里]。这是后续内容。"
        self.initial_text = text
        
        # 找到目标文本的起始和结束位置
        start_index = text.find('[')
        end_index = text.find(']') + 1  # 包括结束括号
        
        # 假设文本都在一行，计算列位置
        self.selection_start = (0, start_index)
        self.selection_end = (0, end_index)
        
        self.target_text = text  # 目标文本相同
        
        return self.initial_text, self.target_text
        
    def get_reward(self, current_text, target_text, action_taken, cursor_pos, done, clipboard=""):
        """文本选择的奖励函数"""
        # 初始化状态跟踪
        if not hasattr(self, 'prev_cursor_pos'):
            self.prev_cursor_pos = cursor_pos
        if not hasattr(self, 'shift_pressed'):
            self.shift_pressed = False
        if not hasattr(self, 'selection_started'):
            self.selection_started = False
            
        reward = -0.001  # 微小的时间惩罚
        
        # 首先需要移动到选择起始位置
        if not self.selection_started:
            # 计算到选择起始位置的距离
            distance_to_start = abs(cursor_pos[0] - self.selection_start[0]) + \
                              abs(cursor_pos[1] - self.selection_start[1])
            
            if distance_to_start == 0:
                # 到达起始位置
                reward += 0.5
                self.selection_started = True
            else:
                # 鼓励移动到起始位置
                prev_distance = abs(self.prev_cursor_pos[0] - self.selection_start[0]) + \
                              abs(self.prev_cursor_pos[1] - self.selection_start[1])
                if distance_to_start < prev_distance:
                    reward += 0.1
        else:
            # 已经在起始位置，现在需要按Shift并移动到结束位置
            if action_taken == 5:  # Shift键
                reward += 0.3  # 奖励按下Shift
                self.shift_pressed = True
            elif self.shift_pressed and action_taken in [0, 1, 2, 3]:  # 方向键
                # 这里应该检查是否有选择区域
                if hasattr(self, 'selection') and self.selection is not None:
                    # 有选择区域，检查是否接近目标选择
                    sel_start = min(self.selection[1], self.selection[3])
                    sel_end = max(self.selection[1], self.selection[3])
                    
                    # 计算选择与目标的重叠度
                    target_start = self.selection_start[1]
                    target_end = self.selection_end[1]
                    
                    # 简单的重叠度计算 (可以改进为更准确的版本)
                    overlap = max(0, min(sel_end, target_end) - max(sel_start, target_start))
                    total_len = max(sel_end, target_end) - min(sel_start, target_start)
                    if total_len > 0:
                        overlap_ratio = overlap / total_len
                        reward += 0.2 * overlap_ratio
                        
                        # 如果完全匹配目标选择
                        if sel_start == target_start and sel_end == target_end:
                            reward += 1.0
            elif not self.shift_pressed and action_taken in [0, 1, 2, 3]:
                # 没按Shift就移动，适当惩罚
                reward -= 0.05
            
        # 更新状态
        self.prev_cursor_pos = cursor_pos
        
        return reward
        
    def is_completed(self, current_text, target_text):
        """当正确选择了目标文本时，任务完成"""
        if hasattr(self, 'selection') and self.selection is not None:
            sel_start = min(self.selection[1], self.selection[3])
            sel_end = max(self.selection[1], self.selection[3])
            target_start = self.selection_start[1]
            target_end = self.selection_end[1]
            
            # 是否完全匹配目标选择
            return sel_start == target_start and sel_end == target_end
        return False


# 课程学习任务生成器
class CurriculumTaskGenerator(TaskGenerator):
    """
    课程学习任务生成器，根据当前阶段生成适当难度的任务
    """
    
    def __init__(self):
        super().__init__()
        # 定义各个阶段的任务
        self.curriculum_tasks = {
            1: Stage1SingleDirectionTask(),  # 阶段1: 单向光标移动
            2: Stage2BasicNavTask(),        # 阶段2: 基础光标导航
            3: Stage3TargetNavTask(),       # 阶段3: 目标导航
            4: Stage4ComplexNavTask(),      # 阶段4: 复杂导航
            5: Stage5TextSelectionTask(),   # 阶段5: 基础文本选择
            # 更多阶段...
        }
        
        # 使用已有的任务作为高级阶段
        self.tasks['copy_paste'] = self.tasks['copy_paste']      # 阶段8: 简单复制粘贴
        self.tasks['find_replace'] = self.tasks['find_replace']  # 阶段10: 查找替换
        
        # 当前阶段
        self.current_stage = 1
        
        # 每个阶段的成功率跟踪
        self.stage_success_rates = {stage: 0.0 for stage in self.curriculum_tasks.keys()}
        self.stage_attempts = {stage: 0 for stage in self.curriculum_tasks.keys()}
        self.stage_successes = {stage: 0 for stage in self.curriculum_tasks.keys()}
        
        # 阶段晋升阈值（成功率达到此值才晋升）
        self.promotion_threshold = 0.7
        
        # 文件路径，用于保存和加载进度
        self.progress_file = "curriculum_progress.json"
        
        # 尝试加载之前的进度
        self.load_progress()
    
    def set_training_stage(self, stage):
        """设置当前训练阶段"""
        if stage in self.curriculum_tasks or stage in self.tasks:
            self.current_stage = stage
            print(f"设置训练阶段为: {stage}")
            return True
        else:
            print(f"警告: 无效的训练阶段 {stage}")
            return False
    
    def generate_task(self, task_type=None):
        """根据当前阶段生成任务"""
        # 如果指定了特定任务类型，使用该类型
        if task_type is not None:
            if task_type in self.tasks:
                task = self.tasks[task_type]
                initial_text, target_text = task.generate()
                return {
                    'task_type': task_type,
                    'task_object': task,
                    'initial_text': initial_text,
                    'target_text': target_text
                }
            else:
                print(f"警告: 未知任务类型 {task_type}")
        
        # 否则，使用当前阶段的任务
        if self.current_stage in self.curriculum_tasks:
            task = self.curriculum_tasks[self.current_stage]
        elif self.current_stage == 8:
            task = self.tasks['copy_paste']
        elif self.current_stage == 10:
            task = self.tasks['find_replace']
        else:
            # 默认使用光标导航任务
            task = self.tasks['cursor_navigation']
            
        initial_text, target_text = task.generate()
        
        return {
            'task_type': f"stage{self.current_stage}",
            'task_object': task,
            'initial_text': initial_text,
            'target_text': target_text
        }
    
    def update_success_rate(self, stage, success):
        """更新指定阶段的成功率"""
        if stage in self.stage_attempts:
            self.stage_attempts[stage] += 1
            if success:
                self.stage_successes[stage] += 1
                
            # 更新成功率
            self.stage_success_rates[stage] = self.stage_successes[stage] / self.stage_attempts[stage]
            
            # 检查是否应该晋升到下一阶段
            if (self.stage_success_rates[stage] >= self.promotion_threshold and 
                self.stage_attempts[stage] >= 20 and  # 确保有足够的尝试次数
                stage == self.current_stage and 
                stage + 1 in self.curriculum_tasks):
                self.current_stage += 1
                print(f"\n晋升到阶段 {self.current_stage}! 前一阶段成功率: {self.stage_success_rates[stage]:.2f}")
                
                # 保存进度
                self.save_progress()
    
    def save_progress(self):
        """保存当前进度"""
        progress = {
            'current_stage': self.current_stage,
            'stage_success_rates': self.stage_success_rates,
            'stage_attempts': self.stage_attempts,
            'stage_successes': self.stage_successes
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=4)
            
        print(f"进度已保存到 {self.progress_file}")
    
    def load_progress(self):
        """加载之前的进度"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    
                self.current_stage = progress.get('current_stage', 1)
                self.stage_success_rates = progress.get('stage_success_rates', {})
                self.stage_attempts = progress.get('stage_attempts', {})
                self.stage_successes = progress.get('stage_successes', {})
                
                # 确保所有阶段都有条目
                for stage in self.curriculum_tasks.keys():
                    if str(stage) not in self.stage_success_rates:
                        self.stage_success_rates[str(stage)] = 0.0
                    if str(stage) not in self.stage_attempts:
                        self.stage_attempts[str(stage)] = 0
                    if str(stage) not in self.stage_successes:
                        self.stage_successes[str(stage)] = 0
                        
                print(f"已加载训练进度，当前阶段: {self.current_stage}")
            except Exception as e:
                print(f"加载进度时出错: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description='训练LLM光标控制代理')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'demo'],
                        help='运行模式: train, test, or demo')
    parser.add_argument('--task', type=str, default=None, 
                        choices=[None, 'copy_paste', 'find_replace', 'cursor_navigation'],
                        help='任务类型，为None则使用课程学习当前阶段')
    parser.add_argument('--stage', type=int, default=None,
                        help='指定训练阶段，覆盖保存的进度')
    parser.add_argument('--num-episodes', type=int, default=10000, help='训练的回合数')
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
    parser.add_argument('--log-dir', type=str, default='runs', help='TensorBoard日志目录')
    parser.add_argument('--checkpoint-freq', type=int, default=500, help='保存检查点的频率（回合）')
    return parser.parse_args()


def train_agent(args):
    """训练光标控制代理，使用课程学习方法"""
    # 创建TensorBoard日志目录
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(args.log_dir, f"curriculum_{current_time}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # 记录超参数
    hparams = {
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'clip_ratio': args.clip_ratio,
        'update_frequency': args.update_frequency,
        'max_steps': args.max_steps,
        'batch_size': args.batch_size
    }
    writer.add_hparams(hparams, {})
    
    # 初始化课程学习任务生成器
    task_generator = CurriculumTaskGenerator()
    
    # 如果指定了特定阶段，则设置该阶段
    if args.stage is not None:
        task_generator.set_training_stage(args.stage)
    
    # 初始化环境
    env = TextEditingEnvWithTasks(task_generator, max_steps=args.max_steps)
    
    # 初始化代理
    agent = GRPOAgent(
        env=env,
        text_embedding_dim=768,
        hidden_dim=256,
        gamma=args.gamma,
        clip_ratio=args.clip_ratio,
        lr=args.learning_rate,
        gpt2_path=args.gpt2_path,
        group_size=16  # GRPO的组大小参数
    )
    
    # 如果指定了模型路径，则加载模型
    if args.load_model and os.path.exists(args.load_model):
        agent.policy.load_state_dict(torch.load(args.load_model))
        writer.add_text('training_info', f'已加载模型: {args.load_model}')
    
    # 创建检查点目录
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 创建进度条
    pbar = tqdm(total=args.num_episodes, desc=f"阶段{task_generator.current_stage}训练")
    pbar.set_postfix({
        'reward': 0.0, 
        'success': 0.0, 
        'stage': task_generator.current_stage
    })
    
    # 跟踪训练统计信息
    rewards = []
    successes = []
    stage_history = []
    
    # 训练循环
    global_step = 0
    update_counter = 0
    
    for episode in range(args.num_episodes):
        # 记录当前阶段
        current_stage = task_generator.current_stage
        stage_history.append(current_stage)
        
        # 重置环境
        obs = env.reset(task_type=args.task)  # 如果指定了任务，使用该任务，否则使用课程学习的任务
        episode_reward = 0
        done = False
        step = 0
        
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
        
        # 记录本回合的成功情况
        task_completed = info.get('task_completed', False)
        successes.append(1.0 if task_completed else 0.0)
        rewards.append(episode_reward)
        
        # 更新任务生成器的成功率统计
        task_generator.update_success_rate(current_stage, task_completed)
        
        # 记录回合统计信息到TensorBoard
        writer.add_scalar('episode/reward', episode_reward, episode)
        writer.add_scalar('episode/success', 1.0 if task_completed else 0.0, episode)
        writer.add_scalar('episode/stage', current_stage, episode)
        
        # 记录每个阶段的成功率
        for stage, rate in task_generator.stage_success_rates.items():
            writer.add_scalar(f'stage/success_rate_{stage}', rate, episode)
        
        # 计算最近50回合的平均值
        window_size = min(50, len(rewards))
        if window_size > 0:
            recent_rewards = rewards[-window_size:]
            recent_successes = successes[-window_size:]
            
            writer.add_scalar('episode/avg_reward_50', np.mean(recent_rewards), episode)
            writer.add_scalar('episode/success_rate_50', np.mean(recent_successes), episode)
        
        # 定期更新策略
        update_counter += 1
        if update_counter >= args.update_frequency:
            agent.update()
            update_counter = 0
            
            # 记录更新信息
            writer.add_scalar('training/updates', episode // args.update_frequency, episode)
        
        # 定期保存检查点
        if (episode + 1) % args.checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode+1}.pt")
            torch.save(agent.policy.state_dict(), checkpoint_path)
            writer.add_text('checkpoint', f'保存检查点: {checkpoint_path}', episode)
        
        # 更新进度条
        # 计算最近的成功率和平均奖励
        recent_success_rate = np.mean(recent_successes) if window_size > 0 else 0
        recent_avg_reward = np.mean(recent_rewards) if window_size > 0 else episode_reward
        
        # 如果阶段变化，重置进度条描述
        if current_stage != task_generator.current_stage:
            pbar.set_description(f"阶段{task_generator.current_stage}训练")
        
        # 更新进度条的附加信息
        pbar.set_postfix({
            'reward': f'{recent_avg_reward:.2f}', 
            'success': f'{recent_success_rate:.2f}', 
            'stage': task_generator.current_stage
        })
        pbar.update(1)
    
    # 关闭进度条
    pbar.close()
    
    # 保存最终模型
    torch.save(agent.policy.state_dict(), args.save_model)
    writer.add_text('training_info', f'保存最终模型到: {args.save_model}', global_step)
    
    # 保存课程学习进度
    task_generator.save_progress()
    
    # 绘制学习曲线
    plt.figure(figsize=(15, 10))
    
    # 1. 奖励曲线
    plt.subplot(3, 1, 1)
    plt.plot(rewards)
    plt.title('任务奖励')
    plt.xlabel('回合')
    plt.ylabel('累积奖励')
    
    # 2. 成功率曲线
    plt.subplot(3, 1, 2)
    window_success = np.convolve(successes, np.ones(50)/50, mode='valid')
    plt.plot(window_success)
    plt.title('成功率 (50回合移动平均)')
    plt.xlabel('回合')
    plt.ylabel('成功率')
    plt.ylim(0, 1)
    
    # 3. 阶段进展
    plt.subplot(3, 1, 3)
    plt.plot(stage_history)
    plt.title('课程学习阶段进展')
    plt.xlabel('回合')
    plt.ylabel('阶段')
    
    # 保存图表
    curve_path = os.path.join(log_dir, 'learning_curves.png')
    plt.tight_layout()
    plt.savefig(curve_path)
    
    # 关闭TensorBoard写入器
    writer.close()
    
    return agent


def main():
    args = parse_args()
    
    if args.mode == 'train':
        train_agent(args)
    elif args.mode == 'test':
        # test_agent 函数需要适配课程学习，可以稍后实现
        print("测试模式暂不支持课程学习")
    elif args.mode == 'demo':
        # demo 函数需要适配课程学习，可以稍后实现
        print("演示模式暂不支持课程学习")
    else:
        print(f"未知模式: {args.mode}")

if __name__ == "__main__":
    main()