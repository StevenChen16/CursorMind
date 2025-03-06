import numpy as np
import gym
from gym import spaces
import pygame
import time
import sys
import traceback
from typing import List, Dict, Tuple, Optional, Union

class TextEditingEnv(gym.Env):
    """
    一个文本编辑环境，模拟Vim的一些基本功能，允许AI操作光标，选择、复制和粘贴文本。
    
    动作空间:
    - 光标移动: u(上), d(下), l(左), r(右)
    - 特殊键: ctrl, shift, alt
    - 点击操作: left_click, right_click
    - 编辑操作: copy, paste, cut, select_all
    
    观察空间:
    - 当前文本内容
    - 光标位置
    - 是否有选中的文本
    - 选中文本的范围
    - 剪贴板内容
    """
    
    def __init__(self, initial_text="", max_text_length=1000, max_episode_steps=100):
        super().__init__()
        
        # 环境参数
        self.max_text_length = max_text_length
        self.max_episode_steps = max_episode_steps
        
        # 定义动作空间
        self.action_space = spaces.Discrete(12)  # 12种基本动作
        self.action_to_name = {
            0: "u",        # 上
            1: "d",        # 下
            2: "l",        # 左
            3: "r",        # 右
            4: "ctrl",     # Ctrl 键
            5: "shift",    # Shift 键
            6: "alt",      # Alt 键
            7: "left_click",  # 左键点击
            8: "right_click", # 右键点击
            9: "copy",     # 复制 (Ctrl+C)
            10: "paste",    # 粘贴 (Ctrl+V)
            11: "cut",      # 剪切 (Ctrl+X)
        }
        
        # 定义观察空间
        # 简化为文本内容、光标位置、选中文本区域等
        self.observation_space = spaces.Dict({
            "text": spaces.Text(max_text_length),
            "cursor_position": spaces.Box(low=0, high=max_text_length, shape=(2,), dtype=np.int32),
            "selection": spaces.Box(low=0, high=max_text_length, shape=(2,), dtype=np.int32),
            "clipboard": spaces.Text(max_text_length),
            "key_pressed": spaces.MultiDiscrete([2, 2, 2]),  # [ctrl, shift, alt]
        })
        
        # 初始化环境状态
        self.reset(initial_text)
        
        # 可视化参数
        self.visualization_enabled = False
        self.screen = None
        self.font = None
        self.text_color = (255, 255, 255)
        self.bg_color = (30, 30, 30)
        self.cursor_color = (255, 165, 0)
        self.selection_color = (100, 100, 200, 100)
        
        # 添加 pygame 初始化状态标志
        self.pygame_initialized = False
    
    def reset(self, initial_text=""):
        """重置环境为初始状态"""
        # 文本状态
        self.text = initial_text
        self.text_lines = initial_text.split("\n")
        if not self.text_lines:
            self.text_lines = [""]
        
        # 将文本表示为一个二维字符数组方便编辑
        self.char_grid = []
        for line in self.text_lines:
            self.char_grid.append(list(line))
        
        # 光标位置 [行, 列]
        self.cursor = [0, 0]
        
        # 选中区域 [开始行,开始列,结束行,结束列]，None表示没有选中
        self.selection = None
        
        # 按键状态 [ctrl, shift, alt]
        self.key_pressed = [0, 0, 0]
        
        # 剪贴板
        self.clipboard = ""
        
        # 步骤计数
        self.steps = 0
        
        # 返回观察
        return self._get_observation()
    
    def _get_observation(self):
        """构建当前环境状态的观察"""
        text = "\n".join("".join(line) for line in self.char_grid)
        
        cursor_pos = np.array(self.cursor, dtype=np.int32)
        
        if self.selection is None:
            selection = np.array([0, 0], dtype=np.int32)
        else:
            selection = np.array([
                self.selection[0] * 1000 + self.selection[1],  # 开始位置
                self.selection[2] * 1000 + self.selection[3]   # 结束位置
            ], dtype=np.int32)
        
        return {
            "text": text,
            "cursor_position": cursor_pos,
            "selection": selection,
            "clipboard": self.clipboard,
            "key_pressed": np.array(self.key_pressed)
        }
    
    def step(self, action):
        """执行一个动作并返回新的状态、奖励、完成标志和额外信息"""
        self.steps += 1
        done = self.steps >= self.max_episode_steps
        
        action_name = self.action_to_name[action]
        
        # 执行动作
        if action_name in ["u", "d", "l", "r"]:
            self._move_cursor(action_name)
        elif action_name in ["ctrl", "shift", "alt"]:
            self._toggle_key(action_name)
        elif action_name == "left_click":
            # 简化：左键点击当前位置
            self.selection = None
        elif action_name == "right_click":
            # 简化：右键显示上下文菜单（这里不实现具体功能）
            pass
        elif action_name == "copy":
            self._copy()
        elif action_name == "paste":
            self._paste()
        elif action_name == "cut":
            self._cut()
        
        # 计算奖励（这里简单返回0，实际应用中可以基于任务完成情况设计奖励）
        reward = 0
        
        return self._get_observation(), reward, done, {}
    
    def _move_cursor(self, direction):
        """移动光标"""
        row, col = self.cursor
        
        if direction == "u" and row > 0:
            row -= 1
            # 确保光标不会超出当前行的长度
            col = min(col, len(self.char_grid[row]))
        
        elif direction == "d" and row < len(self.char_grid) - 1:
            row += 1
            # 确保光标不会超出当前行的长度
            col = min(col, len(self.char_grid[row]))
        
        elif direction == "l" and col > 0:
            col -= 1
        
        elif direction == "r" and col < len(self.char_grid[row]):
            col += 1
        
        self.cursor = [row, col]
        
        # 如果按住shift，则更新选择区域
        if self.key_pressed[1] == 1:  # shift被按下
            if self.selection is None:
                # 开始一个新的选择
                self.selection = [row, col, row, col]
            else:
                # 更新现有选择的结束点
                self.selection[2] = row
                self.selection[3] = col
        else:
            # 如果没有按shift，则清除选择
            self.selection = None
    
    def _toggle_key(self, key):
        """切换按键状态"""
        if key == "ctrl":
            self.key_pressed[0] = 1 - self.key_pressed[0]
        elif key == "shift":
            self.key_pressed[1] = 1 - self.key_pressed[1]
        elif key == "alt":
            self.key_pressed[2] = 1 - self.key_pressed[2]
    
    def _copy(self):
        """复制选中的文本到剪贴板"""
        if self.selection is not None:
            start_row, start_col, end_row, end_col = self.selection
            
            # 确保开始位置在结束位置之前
            if (start_row > end_row) or (start_row == end_row and start_col > end_col):
                start_row, start_col, end_row, end_col = end_row, end_col, start_row, start_col
            
            # 提取选中的文本
            selected_text = []
            for r in range(start_row, end_row + 1):
                if r == start_row and r == end_row:
                    # 选择在同一行
                    line_text = "".join(self.char_grid[r][start_col:end_col])
                elif r == start_row:
                    # 选择的第一行
                    line_text = "".join(self.char_grid[r][start_col:])
                elif r == end_row:
                    # 选择的最后一行
                    line_text = "".join(self.char_grid[r][:end_col])
                else:
                    # 中间行
                    line_text = "".join(self.char_grid[r])
                
                selected_text.append(line_text)
            
            self.clipboard = "\n".join(selected_text)
    
    def _paste(self):
        """粘贴剪贴板内容到当前光标位置"""
        if not self.clipboard:
            return
        
        # 如果有选择，先删除选中内容
        if self.selection is not None:
            self._delete_selected_text()
        
        # 插入剪贴板内容
        row, col = self.cursor
        clipboard_lines = self.clipboard.split("\n")
        
        # 插入第一行
        self.char_grid[row] = self.char_grid[row][:col] + list(clipboard_lines[0]) + self.char_grid[row][col:]
        
        # 如果剪贴板有多行，插入剩余行
        if len(clipboard_lines) > 1:
            remaining_line = self.char_grid[row][col + len(clipboard_lines[0]):]
            self.char_grid[row] = self.char_grid[row][:col + len(clipboard_lines[0])]
            
            # 插入中间行
            for i in range(1, len(clipboard_lines) - 1):
                self.char_grid.insert(row + i, list(clipboard_lines[i]))
            
            # 插入最后一行和原行的剩余部分
            self.char_grid.insert(row + len(clipboard_lines) - 1, 
                                list(clipboard_lines[-1]) + remaining_line)
        
        # 更新光标位置到粘贴内容的末尾
        if len(clipboard_lines) == 1:
            self.cursor = [row, col + len(clipboard_lines[0])]
        else:
            self.cursor = [row + len(clipboard_lines) - 1, len(clipboard_lines[-1])]
        
        # 更新text_lines
        self.text_lines = ["".join(line) for line in self.char_grid]
        self.text = "\n".join(self.text_lines)
    
    def _cut(self):
        """剪切选中的文本"""
        self._copy()
        self._delete_selected_text()
    
    def _delete_selected_text(self):
        """删除选中的文本"""
        if self.selection is None:
            return
        
        start_row, start_col, end_row, end_col = self.selection
        
        # 确保开始位置在结束位置之前
        if (start_row > end_row) or (start_row == end_row and start_col > end_col):
            start_row, start_col, end_row, end_col = end_row, end_col, start_row, start_col
        
        # 处理删除逻辑
        if start_row == end_row:
            # 单行删除
            self.char_grid[start_row] = self.char_grid[start_row][:start_col] + self.char_grid[start_row][end_col:]
        else:
            # 多行删除
            # 保存第一行的开始部分和最后一行的结束部分
            first_line_start = self.char_grid[start_row][:start_col]
            last_line_end = self.char_grid[end_row][end_col:]
            
            # 删除中间所有行
            self.char_grid = self.char_grid[:start_row] + self.char_grid[end_row+1:]
            
            # 组合第一行的开始和最后一行的结束
            if start_row < len(self.char_grid):
                self.char_grid[start_row] = first_line_start + last_line_end
            else:
                self.char_grid.append(list(first_line_start + last_line_end))
        
        # 更新光标位置到选择的开始位置
        self.cursor = [start_row, start_col]
        
        # 清除选择
        self.selection = None
        
        # 更新text_lines
        self.text_lines = ["".join(line) for line in self.char_grid]
        self.text = "\n".join(self.text_lines)
    
    def render(self, mode="human"):
        """渲染当前环境状态"""
        if mode == "human":
            try:
                if not self.visualization_enabled:
                    self._init_visualization()
                
                self._update_visualization()
                return None
            except Exception as e:
                print(f"渲染错误: {e}")
                traceback.print_exc()
                self.close()
                return None
        else:
            # 返回文本表示
            output = []
            for i, line in enumerate(self.char_grid):
                # 渲染行
                rendered_line = ""
                for j, char in enumerate(line):
                    if self.cursor == [i, j]:
                        rendered_line += f"[{char}]"  # 光标位置
                    elif (self.selection is not None and 
                          i >= min(self.selection[0], self.selection[2]) and 
                          i <= max(self.selection[0], self.selection[2]) and
                          j >= min(self.selection[1], self.selection[3]) and 
                          j <= max(self.selection[1], self.selection[3])):
                        rendered_line += f"|{char}|"  # 选中文本
                    else:
                        rendered_line += char
                
                # 处理行末光标
                if self.cursor == [i, len(line)]:
                    rendered_line += "[]"
                
                output.append(rendered_line)
            
            # 添加状态信息
            output.append(f"\nCursor: {self.cursor}")
            output.append(f"Selection: {self.selection}")
            output.append(f"Keys: Ctrl={bool(self.key_pressed[0])}, Shift={bool(self.key_pressed[1])}, Alt={bool(self.key_pressed[2])}")
            output.append(f"Clipboard: '{self.clipboard}'")
            
            return "\n".join(output)
    
    def _init_visualization(self):
        """初始化可视化"""
        try:
            # 检查pygame是否已初始化
            if not self.pygame_initialized:
                pygame.init()
                self.pygame_initialized = True
                
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("Text Editing Environment")
            
            # 尝试加载字体，如果失败则使用默认字体
            try:
                self.font = pygame.font.SysFont('monospace', 16)
            except:
                print("无法加载指定字体，使用默认字体")
                self.font = pygame.font.Font(None, 16)
                
            self.visualization_enabled = True
            print("可视化初始化成功")
        except Exception as e:
            print(f"可视化初始化失败: {e}")
            traceback.print_exc()
            self.visualization_enabled = False
            raise
    
    def _update_visualization(self):
        """更新可视化显示"""
        try:
            if not self.visualization_enabled or self.screen is None:
                return
                
            self.screen.fill(self.bg_color)
            
            # 渲染文本
            y_offset = 10
            for i, line in enumerate(self.char_grid):
                x_offset = 10
                line_text = "".join(line)
                
                # 计算选中区域
                selection_rects = []
                if self.selection is not None:
                    start_row, start_col, end_row, end_col = self.selection
                    if start_row > end_row or (start_row == end_row and start_col > end_col):
                        start_row, start_col, end_row, end_col = end_row, end_col, start_row, start_col
                    
                    if i >= start_row and i <= end_row:
                        if i == start_row and i == end_row:
                            # 选择在同一行
                            selection_start = start_col
                            selection_end = end_col
                        elif i == start_row:
                            # 选择的第一行
                            selection_start = start_col
                            selection_end = len(line)
                        elif i == end_row:
                            # 选择的最后一行
                            selection_start = 0
                            selection_end = end_col
                        else:
                            # 中间行，全选
                            selection_start = 0
                            selection_end = len(line)
                        
                        # 计算选中区域的矩形
                        if selection_start < len(line):
                            char_width = self.font.size("A")[0]  # 估计字符宽度
                            selection_width = char_width * (selection_end - selection_start)
                            selection_rect = pygame.Rect(
                                x_offset + selection_start * char_width, 
                                y_offset, 
                                selection_width, 
                                self.font.get_height()
                            )
                            selection_rects.append(selection_rect)
                
                # 绘制选中区域
                for rect in selection_rects:
                    pygame.draw.rect(self.screen, self.selection_color, rect)
                
                # 绘制文本
                text_surface = self.font.render(line_text, True, self.text_color)
                self.screen.blit(text_surface, (x_offset, y_offset))
                
                # 绘制光标
                if i == self.cursor[0]:
                    char_width = self.font.size("A")[0]  # 估计字符宽度
                    cursor_x = x_offset + self.cursor[1] * char_width
                    pygame.draw.line(
                        self.screen, 
                        self.cursor_color, 
                        (cursor_x, y_offset), 
                        (cursor_x, y_offset + self.font.get_height())
                    )
                
                y_offset += self.font.get_height() + 2
            
            # 显示状态信息
            status_line = f"Cursor: {self.cursor} | "
            status_line += f"Selection: {self.selection} | "
            status_line += f"Keys: Ctrl={bool(self.key_pressed[0])}, Shift={bool(self.key_pressed[1])}, Alt={bool(self.key_pressed[2])} | "
            status_line += f"Clipboard: '{self.clipboard}'"
            
            status_surface = self.font.render(status_line, True, (200, 200, 200))
            self.screen.blit(status_surface, (10, 550))
            
            pygame.display.flip()
            
            # 处理任何挂起的事件以防止界面冻结
            pygame.event.pump()
            
        except Exception as e:
            print(f"更新可视化时出错: {e}")
            traceback.print_exc()
            self.visualization_enabled = False
    
    def close(self):
        """关闭环境"""
        try:
            if self.visualization_enabled:
                pygame.quit()
            self.visualization_enabled = False
            self.pygame_initialized = False
        except Exception as e:
            print(f"关闭环境时出错: {e}")
            traceback.print_exc()


# 一些使用示例代码
def manual_control_demo():
    """手动控制演示 - 允许用户通过键盘控制环境"""
    print("正在初始化Pygame...")
    try:
        pygame.init()
    except Exception as e:
        print(f"Pygame初始化失败: {e}")
        return
        
    print("正在创建环境...")
    env = TextEditingEnv(initial_text="Hello, world!\nThis is a text editing environment.\nYou can move the cursor, select text, and edit it.")
    
    print("重置环境...")
    obs = env.reset()
    
    print("渲染环境...")
    try:
        env.render(mode="human")
    except Exception as e:
        print(f"渲染错误: {e}")
        env.close()
        pygame.quit()
        return
    
    action_keys = {
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
    
    running = True
    clock = pygame.time.Clock()
    
    print("进入主循环...")
    try:
        while running:
            # 确保事件处理
            pygame.event.pump()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in action_keys:
                        action = action_keys[event.key]
                        
                        # 特殊处理组合键
                        if action in [9, 10, 11]:  # copy, paste, cut
                            if env.key_pressed[0]:  # 检查Ctrl是否被按下
                                obs, reward, done, info = env.step(action)
                                print(f"Action: {env.action_to_name[action]}, Reward: {reward}")
                        else:
                            obs, reward, done, info = env.step(action)
                            print(f"Action: {env.action_to_name[action]}, Reward: {reward}")
                            
                            # 特殊处理组合键的情况
                            if action == 4:  # ctrl
                                print("Ctrl pressed")
                    
                    try:
                        env.render(mode="human")
                    except Exception as e:
                        print(f"渲染错误: {e}")
                        running = False
            
            # 限制帧率，防止CPU占用过高
            clock.tick(30)
    except Exception as e:
        print(f"主循环异常: {e}")
        traceback.print_exc()
    finally:
        print("关闭环境...")
        env.close()
        print("退出Pygame...")
        pygame.quit()


def agent_demo():
    """简单的智能体演示 - 随机动作"""
    print("初始化环境...")
    try:
        pygame.init()
    except Exception as e:
        print(f"Pygame初始化失败: {e}")
        return
        
    env = TextEditingEnv(initial_text="Hello, world!\nThis is a text editing environment.\nLet's see if an agent can learn to edit text.")
    obs = env.reset()
    
    total_reward = 0
    done = False
    
    # 使用可视化渲染
    try:
        env.render(mode="human")
    except Exception as e:
        print(f"渲染错误: {e}")
        env.close()
        pygame.quit()
        return
        
    print("等待1秒...")
    time.sleep(1)  # 短暂暂停以便查看初始状态
    
    print("开始随机动作...")
    try:
        step_count = 0
        max_steps = 50  # 限制最大步数以防止无限循环
        
        while not done and step_count < max_steps:
            # 随机选择动作
            action = env.action_space.sample()
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # 显示环境状态
            env.render(mode="human")
            
            print(f"Step {step_count}: Action: {env.action_to_name[action]}, Reward: {reward}")
            
            # 处理事件以保持响应性
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            time.sleep(0.2)  # 减慢速度以便观察
    except Exception as e:
        print(f"演示过程中出错: {e}")
        traceback.print_exc()
    finally:
        print(f"总步数: {step_count}")
        print(f"总奖励: {total_reward}")
        print("关闭环境...")
        env.close()
        print("退出Pygame...")
        pygame.quit()


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--agent":
            agent_demo()
        else:
            manual_control_demo()
    except Exception as e:
        print(f"程序运行时出错: {e}")
        traceback.print_exc()
        # 确保pygame已退出
        pygame.quit()