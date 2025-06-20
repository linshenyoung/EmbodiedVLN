import os
import cv2
import re
import anthropic
import numpy as np
from PIL import Image
from vln.agent import AirsimAgent
import base64
from openai import OpenAI
from prompts.prompt2 import build_prompt
from utils import encode_image, LM_VLN

class BaseAgent(object):

    def __init__(self, root_dir):
        self.root_dir = root_dir

    def load_navi_task(self):
        navi_data = []

        task_file = os.path.join(self.root_dir, 'start_loc.txt')
        gt_traj_dir = os.path.join(self.root_dir, 'label')
        if not os.path.isfile(task_file):
            raise ValueError(f"Task file not found in {task_file}")

        with open(task_file, 'r') as f:
            task_data = f.readlines()

        traj_files = os.listdir(gt_traj_dir)
        traj_files = sorted(traj_files, key=lambda x: int(x.split('.')[0]))

        assert len(task_data) == len(traj_files)

        for i in range(len(task_data)):
            task_line = task_data[i]
            traj_file = os.path.join(self.root_dir, 'label', traj_files[i])

            init_pos, init_rot, task_desc = self.parse_task_line(task_line) # 返回Datasets/vln/start_loc.txt中的坐标(init_pos), 旋转(init_rot), 文字描述(task_desc)
            gt_traj = self.parse_traj_file(traj_file) # 返回Datasets/vln/label/x.csv中的各行的坐标xyz
            target_pos = init_pos + gt_traj[len(gt_traj)-1] # 获得目标坐标

            gt_traj_len = 0.0
            last_pos = np.zeros(3)
            for j in range(len(gt_traj)): # 计算csv文件中的所有步数路径长度之和
                step_len = np.linalg.norm(gt_traj[j] - last_pos)
                gt_traj_len += step_len
                last_pos = gt_traj[j]

            navi_data.append({
                "start_pos": init_pos, # 起始点坐标
                "target_pos": target_pos, # 终点坐标
                "start_rot": init_rot, # 起始旋转
                "gt_traj": gt_traj, # csv文件中每步移动的坐标
                "gt_traj_len": gt_traj_len, # 总步长
                "task_desc": task_desc # 任务的文字描述
            })

        return navi_data
    

    def parse_task_line(self, task_line: str):
        task_line = task_line.strip('\n')
        items = task_line.split(';')

        pos_corp = items[0].strip(' ') # xyz坐标
        rot_corp = items[1].strip(' ') # 旋转角度
        desc = items[2].strip(' ') # 文本描述

        pos_str_items = pos_corp.split(' ')[1:]
        rot_str_items = rot_corp.split(':')[1].split(', ')

        for i in range(len(pos_str_items)):
            pos_str_items[i] = pos_str_items[i].strip(',')

        for i in range(len(rot_str_items)):
            rot_str_items[i] = rot_str_items[i].strip(' ')

        pos = list(map(float, pos_str_items)) # 获得处理之后的pos
        rot = list(map(float, rot_str_items)) # 获得处理之后的rot

        pos = np.array(pos) / 100   # cm to m
        rot = np.array(rot)

        return pos, rot, desc
    

    def parse_traj_file(self, traj_file: str):
            if not os.path.isfile(traj_file):
                raise ValueError(f"Trajectory file is not found in {traj_file}")
            with open(traj_file, 'r') as f:
                traj_lines = f.readlines()

            traj = []
            traj_lines = traj_lines[1:]
            for i in range(len(traj_lines)):
                traj_line = traj_lines[i].strip('\n')

                pos_str_items = traj_line.split(',')[1:] # 获得坐标
                pos = list(map(float, pos_str_items))
                traj.append(pos)

            return np.array(traj)
    
    def parse_llm_action(self, llm_output: str):

        matches = re.findall(r':(.*)', llm_output)
        thinking_str = matches[0]
        thinking_str = thinking_str.strip(" ")
        command_str = matches[1]
        command_str = command_str.strip(" ")
        command_str = command_str.lower()

        print("matches: ", matches)

        act_enum = -1
        if 'stop' in command_str:
            return 0, thinking_str, command_str
        elif 'forth' in command_str:
            return 1, thinking_str, command_str
        elif 'left' in command_str:
            return 2, thinking_str, command_str
        elif 'right' in command_str:
            return 3, thinking_str, command_str
        elif 'up' in command_str:
            return 4, thinking_str, command_str
        elif 'down' in command_str:
            return 5, thinking_str, command_str
        else:
            return -1, thinking_str, command_str