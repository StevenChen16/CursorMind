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
        """计算复制粘贴任务的奖励"""
        # 记录上一步的信息用于比较
        if not hasattr(self, 'prev_cursor_pos'):
            self.prev_cursor_pos = cursor_pos
        if not hasattr(self, 'prev_text'):
            self.prev_text = current_text
        if not hasattr(self, 'prev_clipboard'):
            self.prev_clipboard = clipboard
        if not hasattr(self, 'action_history'):
            self.action_history = []
        if not hasattr(self, 'stage'):
            self.stage = 'locate_source'  # 初始阶段：定位源
        
        # 更新行动历史
        self.action_history.append(action_taken)
        
        # 初始化奖励
        reward = -0.01  # 基础时间惩罚
        
        # 获取动作名称
        action_name = action_taken if isinstance(action_taken, str) else None
        
        # 计算关键位置
        copy_start_pos = self.copy_start_idx
        copy_end_pos = self.copy_end_idx
        paste_pos = self.paste_idx
        
        # 计算当前距离
        cursor_abs_pos = cursor_pos[0] * 1000 + cursor_pos[1]
        dist_to_copy_start = abs(cursor_abs_pos - copy_start_pos)
        dist_to_copy_end = abs(cursor_abs_pos - copy_end_pos)
        dist_to_paste = abs(cursor_abs_pos - paste_pos)
        
        # 计算上一步距离
        prev_cursor_abs_pos = self.prev_cursor_pos[0] * 1000 + self.prev_cursor_pos[1]
        prev_dist_to_copy_start = abs(prev_cursor_abs_pos - copy_start_pos)
        prev_dist_to_copy_end = abs(prev_cursor_abs_pos - copy_end_pos)
        prev_dist_to_paste = abs(prev_cursor_abs_pos - paste_pos)
        
        # 1. 阶段性奖励计算 - 降低奖励值以避免过度奖励
        if self.stage == 'locate_source':
            # 定位源文本阶段
            if action_name in ['l', 'r', 'u', 'd']:
                # 移动朝向复制起点的奖励 - 降低倍数
                if dist_to_copy_start < prev_dist_to_copy_start:
                    reward += 0.1 * (prev_dist_to_copy_start - dist_to_copy_start) / max(prev_dist_to_copy_start, 1)
                elif dist_to_copy_start > prev_dist_to_copy_start:
                    reward -= 0.02 * (dist_to_copy_start - prev_dist_to_copy_start) / max(prev_dist_to_copy_start, 1)
                
                # 接近复制区域起点的里程碑奖励 - 降低奖励值
                if dist_to_copy_start < 10 and prev_dist_to_copy_start >= 10:
                    reward += 0.05
                if dist_to_copy_start < 5 and prev_dist_to_copy_start >= 5:
                    reward += 0.08
                if dist_to_copy_start == 0 and prev_dist_to_copy_start > 0:
                    reward += 0.1
                    self.stage = 'selection'  # 进入选择阶段
            
            # 不适当的动作惩罚 - 增加惩罚
            if action_name in ['copy', 'paste', 'cut']:
                reward -= 0.05
        
        elif self.stage == 'selection':
            # 选择文本阶段
            if action_name == 'shift' and 'shift' not in self.action_history[:-1]:
                reward += 0.05  # 正确开始选择
            
            # 判断是否有文本被选中
            selection_active = False
            if hasattr(self, 'selection') and self.selection is not None:
                selection_active = True
                sel_start = min(self.selection[1], self.selection[3])
                sel_end = max(self.selection[1], self.selection[3])
                sel_len = sel_end - sel_start
                
                # 计算选择覆盖率
                correct_sel_start = min(copy_start_pos % 1000, copy_end_pos % 1000)
                correct_sel_end = max(copy_start_pos % 1000, copy_end_pos % 1000)
                correct_sel_len = correct_sel_end - correct_sel_start
                
                # 选择精度奖励 - 降低奖励，增加惩罚
                if correct_sel_len > 0:  # 避免除以零
                    if sel_len <= correct_sel_len:
                        # 选择部分正确但不足
                        correct_ratio = sel_len / correct_sel_len
                        reward += correct_ratio * 0.2  # 降低正确选择的奖励
                    else:
                        # 选择过多
                        over_selection = sel_len - correct_sel_len
                        reward -= 0.03 * over_selection / correct_sel_len  # 增加过度选择惩罚
                
                # 选择方向引导
                if action_name in ['l', 'r', 'u', 'd'] and selection_active:
                    if dist_to_copy_end < prev_dist_to_copy_end:
                        reward += 0.02  # 朝正确方向扩展选择
                    else:
                        reward -= 0.01  # 朝错误方向扩展选择
            
            # 不适当的动作惩罚
            if not selection_active and action_name in ['copy', 'paste', 'cut']:
                reward -= 0.05
            
            # 检测是否完成选择并进入复制阶段
            if selection_active and action_name in ['copy', 'cut']:
                self.stage = 'copy'  # 进入复制阶段
        
        elif self.stage == 'copy':
            # 复制阶段
            if action_name == 'copy':
                # 检查剪贴板是否包含目标文本 - 更细致的奖励
                if clipboard == self.text_to_copy:
                    reward += 0.3  # 完全匹配降低奖励
                    self.stage = 'locate_paste'  # 进入定位粘贴位置阶段
                elif self.text_to_copy in clipboard:
                    # 部分包含但有额外内容
                    reward += 0.15
                    self.stage = 'locate_paste'
                elif clipboard:
                    # 部分匹配奖励
                    similarity = difflib.SequenceMatcher(None, clipboard, self.text_to_copy).ratio()
                    if similarity > 0.5:
                        reward += similarity * 0.1
                    else:
                        reward -= 0.03  # 复制了错误的内容
                    
                    # 仍然前进到下一阶段，但给予较少奖励
                    if similarity > 0.3:
                        self.stage = 'locate_paste'
            
            # 不适当动作惩罚
            if action_name == 'paste':
                reward -= 0.05
        
        elif self.stage == 'locate_paste':
            # 定位粘贴位置阶段
            if action_name in ['l', 'r', 'u', 'd']:
                # 移动朝向粘贴点的奖励 - 降低倍数
                if dist_to_paste < prev_dist_to_paste:
                    reward += 0.01 * (prev_dist_to_paste - dist_to_paste) / max(prev_dist_to_paste, 1)
                elif dist_to_paste > prev_dist_to_paste:
                    reward -= 0.02 * (dist_to_paste - prev_dist_to_paste) / max(prev_dist_to_paste, 1)
                
                # 接近粘贴位置的里程碑奖励 - 降低奖励值
                if dist_to_paste < 10 and prev_dist_to_paste >= 10:
                    reward += 0.05
                if dist_to_paste < 5 and prev_dist_to_paste >= 5:
                    reward += 0.08
                if dist_to_paste == 0 and prev_dist_to_paste > 0:
                    reward += 0.1
                    self.stage = 'paste'  # 进入粘贴阶段
            
            # 不适当的动作惩罚
            if action_name in ['copy', 'cut']:
                reward -= 0.05
        
        elif self.stage == 'paste':
            # 粘贴阶段
            if action_name == 'paste':
                # 检查粘贴后的文本是否更接近目标 - 更注重相似度提升
                similarity_before = difflib.SequenceMatcher(None, current_text, target_text).ratio()
                
                # 模拟粘贴操作后的文本
                paste_pos_val = cursor_pos[0] * 1000 + cursor_pos[1]
                simulated_text = current_text[:paste_pos_val] + clipboard + current_text[paste_pos_val:]
                
                similarity_after = difflib.SequenceMatcher(None, simulated_text, target_text).ratio()
                
                if similarity_after > similarity_before:
                    # 相似度提升奖励 - 更合理的范围
                    improvement = similarity_after - similarity_before
                    reward += 0.3 * improvement  # 降低基础奖励
                    
                    # 相似度里程碑奖励
                    if similarity_after >= 0.95:
                        reward += 0.4  # 几乎完美
                    elif similarity_after >= 0.8:
                        reward += 0.2  # 非常接近
                else:
                    # 粘贴降低了相似度
                    reward -= 0.1  # 轻微惩罚错误粘贴
        
        # 最终任务完成奖励 - 强化对最终结果的重视
        if done:
            final_similarity = difflib.SequenceMatcher(None, current_text, target_text).ratio()
            if final_similarity == 1.0:  # 完全匹配
                reward += 2.0  # 给予显著但不过高的完成奖励
            elif final_similarity >= 0.9:  # 非常接近
                reward += 1.0 * final_similarity
            elif final_similarity >= 0.7:  # 基本接近
                reward += 0.5 * final_similarity
            else:  # 不够接近
                reward -= (1.0 - final_similarity) * 0.3  # 根据差距给予惩罚
        
        # 通用组合键奖励 - 修复之前的索引错误问题并降低奖励
        if action_name in ['copy', 'paste', 'cut'] and len(self.action_history) >= 2:
            if action_name == 'copy' and self.action_history[-2] == 'ctrl':
                reward += 0.05  # 降低组合键奖励
            elif action_name == 'paste' and self.action_history[-2] == 'ctrl':
                reward += 0.05
            elif action_name == 'cut' and self.action_history[-2] == 'ctrl':
                reward += 0.05
        
        # 更新状态
        self.prev_cursor_pos = cursor_pos
        self.prev_text = current_text
        self.prev_clipboard = clipboard
        
        return reward


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
        """计算查找替换任务的奖励"""
        # 记录上一步的信息用于比较
        if not hasattr(self, 'prev_cursor_pos'):
            self.prev_cursor_pos = cursor_pos
        if not hasattr(self, 'prev_text'):
            self.prev_text = current_text
        if not hasattr(self, 'found_words'):
            self.found_words = []
        if not hasattr(self, 'replaced_words'):
            self.replaced_words = 0
        if not hasattr(self, 'target_word_positions'):
            # 找出所有需要替换的词的位置
            self.target_word_positions = []
            index = 0
            while True:
                index = self.initial_text.find(self.word_to_replace, index)
                if index == -1:
                    break
                self.target_word_positions.append(index)
                index += len(self.word_to_replace)
            self.total_words = len(self.target_word_positions)
        
        # 初始化奖励
        reward = -0.01  # 基础时间惩罚
        
        # 获取动作名称
        action_name = action_taken if isinstance(action_taken, str) else None
        
        # 计算当前位置到最近目标词的距离
        current_pos = cursor_pos[0] * 1000 + cursor_pos[1]
        prev_pos = self.prev_cursor_pos[0] * 1000 + self.prev_cursor_pos[1]
        
        # 分析文本变化情况
        initial_occurrences = self.initial_text.count(self.word_to_replace)
        current_occurrences = current_text.count(self.word_to_replace)
        new_replaced = initial_occurrences - current_occurrences - self.replaced_words
        
        # 提供更积极的引导奖励
        if self.target_word_positions:
            # 找出当前最近的目标词
            distances = [abs(current_pos - pos) for pos in self.target_word_positions]
            min_distance = min(distances)
            nearest_target_index = distances.index(min_distance)
            nearest_target_pos = self.target_word_positions[nearest_target_index]
            
            # 上一步到当前目标词的距离
            prev_distance = abs(prev_pos - nearest_target_pos)
            
            # 1. 移动朝向目标词的奖励 - 更明显的引导
            if action_name in ['l', 'r', 'u', 'd']:
                if min_distance < prev_distance:
                    # 朝向目标词移动 - 提供更积极的引导
                    distance_improvement = prev_distance - min_distance
                    reward += 0.02 * min(distance_improvement / max(prev_distance, 1), 0.5)  # 设置合理上限
                elif min_distance > prev_distance:
                    # 远离目标词移动 - 轻微惩罚
                    reward -= 0.01
                
                # 接近目标词的里程碑奖励
                target_word_length = len(self.word_to_replace)
                if min_distance < 2 * target_word_length and prev_distance >= 2 * target_word_length:
                    reward += 0.05
                if min_distance < target_word_length and prev_distance >= target_word_length:
                    reward += 0.08
                if min_distance == 0 and prev_distance > 0:
                    reward += 0.1  # 精确到达目标词起始位置
                    
                    # 检查是否发现新目标词
                    if nearest_target_pos not in self.found_words:
                        self.found_words.append(nearest_target_pos)
                        reward += 0.1  # 发现新目标词奖励
        
        # 2. 替换操作奖励 - 更合理的奖励
        if new_replaced > 0:
            # 成功替换词奖励
            self.replaced_words += new_replaced
            reward += 0.2 * new_replaced  # 每替换一个词的基础奖励
            
            # 替换比例奖励 - 更合理的范围
            if initial_occurrences > 0:  # 避免除以零
                replace_ratio = self.replaced_words / initial_occurrences
                reward += replace_ratio * 0.3  # 降低替换比例奖励
            
            # 检查是否全部替换完成
            if current_occurrences == 0:
                reward += 0.5  # 全部替换完成额外奖励
        
        # 3. 查找相关奖励 - 更细致的引导
        # 检查是否正在使用查找操作（如Ctrl+F，但简化环境中可能没有此操作）
        if action_name in ['ctrl', 'shift'] and self.replaced_words < initial_occurrences:
            # 鼓励使用快捷键
            reward += 0.01
        
        # 4. 任务进度奖励 - 更平滑的奖励
        if initial_occurrences > 0:
            progress = self.replaced_words / initial_occurrences
            # 仅在首次达到里程碑时给予奖励
            if not hasattr(self, 'progress_milestones'):
                self.progress_milestones = {0.25: False, 0.5: False, 0.75: False, 1.0: False}
                
            for milestone, achieved in self.progress_milestones.items():
                if not achieved and progress >= milestone:
                    self.progress_milestones[milestone] = True
                    reward += milestone * 0.3  # 与里程碑成比例的奖励
        
        # 5. 最终任务完成评估
        if done:
            final_similarity = difflib.SequenceMatcher(None, current_text, target_text).ratio()
            if final_similarity == 1.0:  # 完全匹配
                reward += 1.0
            elif final_similarity >= 0.9:  # 非常接近
                reward += 0.8
            elif final_similarity >= 0.7:  # 基本接近
                reward += 0.4
            else:  # 不够接近
                reward -= (1.0 - final_similarity) * 0.2  # 根据差距给予惩罚
        
        # 更新状态
        self.prev_cursor_pos = cursor_pos
        self.prev_text = current_text
        
        return reward


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
        """计算光标导航任务的奖励"""
        # 目标位置
        target_line, target_col = self.target_position
        current_line, current_col = cursor_pos
        
        # 记录上一步的信息用于比较
        if not hasattr(self, 'prev_cursor_pos'):
            self.prev_cursor_pos = cursor_pos
        if not hasattr(self, 'initial_distance'):
            # 计算初始距离
            self.initial_distance = abs(current_line - target_line) + abs(current_col - target_col)
            self.max_line_dist = max(1, target_line)  # 避免除以零
            self.max_col_dist = max(1, target_col)  # 避免除以零
        if not hasattr(self, 'min_distance_reached'):
            self.min_distance_reached = self.initial_distance
        if not hasattr(self, 'position_history'):
            self.position_history = [cursor_pos]
        if not hasattr(self, 'achieved_milestones'):
            self.achieved_milestones = {
                'halfway': False,  # 距离减半
                'almost': False,   # 距离减少90%
                'close': False     # 距离为1
            }
        
        # 初始化奖励
        reward = -0.01  # 基础时间惩罚
        
        # 获取动作名称
        action_name = action_taken if isinstance(action_taken, str) else None
        
        # 计算距离
        current_distance = abs(current_line - target_line) + abs(current_col - target_col)
        prev_line, prev_col = self.prev_cursor_pos
        prev_distance = abs(prev_line - target_line) + abs(prev_col - target_col)
        
        # 1. 完全达到目标位置奖励 - 保持较高但合理
        if current_line == target_line and current_col == target_col:
            reward += 1.0  # 精确达到目标的奖励
        # 2. 接近但不精确奖励 - 更细致的引导
        elif current_distance == 1:
            reward += 0.5  # 仅差1个位置
        elif current_distance <= 3:
            reward += 0.2  # 差2-3个位置
        elif current_distance <= 5:
            reward += 0.1  # 差4-5个位置
        
        # 3. 朝正确方向移动的强化奖励
        if action_name in ['u', 'd', 'l', 'r']:
            # 垂直方向正确
            vert_correct = (action_name == 'u' and target_line < current_line) or \
                        (action_name == 'd' and target_line > current_line)
            # 水平方向正确
            horz_correct = (action_name == 'l' and target_col < current_col) or \
                        (action_name == 'r' and target_col > current_col)
            
            if vert_correct:
                # 垂直方向正确
                vert_dist_improve = abs(prev_line - target_line) - abs(current_line - target_line)
                if vert_dist_improve > 0:
                    # 实际减少了垂直距离
                    reward += 0.04 * (vert_dist_improve / self.max_line_dist)
            elif action_name in ['u', 'd'] and not vert_correct:
                # 垂直方向错误
                reward -= 0.03
                
            if horz_correct:
                # 水平方向正确
                horz_dist_improve = abs(prev_col - target_col) - abs(current_col - target_col)
                if horz_dist_improve > 0:
                    # 实际减少了水平距离
                    reward += 0.04 * (horz_dist_improve / self.max_col_dist)
            elif action_name in ['l', 'r'] and not horz_correct:
                # 水平方向错误
                reward -= 0.03
                
            # 同时两个方向都正确的额外奖励
            if vert_correct and horz_correct:
                reward += 0.02
        
        # 4. 距离里程碑奖励 - 仅在首次达到时
        if current_distance < self.min_distance_reached:
            # 更新最小距离
            self.min_distance_reached = current_distance
            
            # 计算达成的里程碑
            if not self.achieved_milestones['halfway'] and current_distance <= self.initial_distance / 2:
                self.achieved_milestones['halfway'] = True
                reward += 0.1  # 距离减半
                
            if not self.achieved_milestones['almost'] and current_distance <= self.initial_distance / 10:
                self.achieved_milestones['almost'] = True
                reward += 0.2  # 距离减少90%
                
            if not self.achieved_milestones['close'] and current_distance <= 1:
                self.achieved_milestones['close'] = True
                reward += 0.3  # 非常接近
        
        # 5. 徘徊惩罚 - 避免原地打转
        self.position_history.append(cursor_pos)
        if len(self.position_history) > 5:
            self.position_history.pop(0)
            
            # 检查最近5步是否有实质性移动
            unique_positions = set((p[0], p[1]) for p in self.position_history)
            if len(unique_positions) <= 2:  # 只有1-2个不同位置
                reward -= 0.02  # 轻微徘徊惩罚
        
        # 6. 不适当操作惩罚 - 减轻但保留
        if action_name in ['copy', 'paste', 'cut', 'shift', 'ctrl', 'alt']:
            reward -= 0.03  # 导航任务中使用不必要的操作
        
        # 更新状态
        self.prev_cursor_pos = cursor_pos
        
        return reward


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
    # try:
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
    # except Exception as e:
    #     print(f"程序运行错误: {e}")
    #     pygame.quit()