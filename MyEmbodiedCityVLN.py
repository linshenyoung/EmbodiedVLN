import os
import cv2
import anthropic
import numpy as np
from PIL import Image
from vln.agent import AirsimAgent
import base64
from openai import OpenAI
from prompts.prompt2 import build_prompt
from utils import encode_image, LM_VLN
from EmbodiedNavAgent import EmbodiedNavAgent


class MyEmbodiedCityVLN:

    def __init__(self, root_dir, eval_model, api_key):
        self.root_dir = root_dir
        self.eval_model = eval_model
        self.agent = AirsimAgent(None, None, None)
        self.agent_nav = EmbodiedNavAgent(eval_model, api_key, root_dir)
        self.navi_tasks = self.agent_nav.load_navi_task()
        self.history = []

    def run(self):

        navi_data = self.navi_tasks
        SR_count = 0.0
        SPL = 0.0
        traj_len = 0.0
        ne_count = 0.0
        SR_short_idx = []
        SR_long_idx = []
        for idx, navi_task in enumerate(navi_data):
            # if idx > 50:
            #     break

            traj_len = 0.0

            start_pos = navi_task["start_pos"]         # 起始点坐标
            start_rot = navi_task["start_rot"]         # 起始旋转
            gt_traj = navi_task["gt_traj"]             # csv文件中每步移动的坐标
            target_pos = navi_task["target_pos"]       # 终点坐标
            gt_traj_len = navi_task["gt_traj_len"]     # 总步长
            task_desc = navi_task["task_desc"]         # 任务的文字描述

            start_pos[2] = -start_pos[2]    # unreal coords to airsim coords

            start_pose = np.concatenate((start_pos, start_rot))
            # print(f"start pose: {start_pose}")

            self.agent.setVehiclePose(start_pose)
            # time.sleep(1)
            # self.agent.client.moveToPositionAsync(float(start_pose[0]), float(start_pose[1]), float(start_pose[2]), 1).join()
            pos, rot = self.agent.get_current_state()
            print(f"pos: {pos}, rot: {rot}")

            history = []

            step_size = 0
            while step_size < 30:

                messages = []

                self.agent_nav.get_image(self.agent) # get the front and down view RGB and Depth Images and save them into /imgs folder

                # Getting the base64 string
            
                encoded_imgs = self.agent_nav.encode_images() # encode these four images into base64 string for LLM to inference

                obs_desc = self.agent_nav.parse_images(encoded_imgs) # understand these four images and give some descriptions on them
            
                obs_desc_summarized = self.agent_nav.summarizer(obs_desc) # to prevent the prompt from large scale, we summarize the observation descriptions into one sentence

                if step_size == 0:
                    history = ["Init Observation: Navigation start, no actions taken yet.\nCurrent scene from the view is " + obs_desc_summarized] # init the history

                fillin = [task_desc, history, obs_desc_summarized]
            
                # history = self.agent_nav.get_prompt("HISTORY_PROMPT", fillin)

                msg = self.agent_nav.prompt_manager(fillin, encoded_imgs, messages) # get the final prompt

                answer = self.agent_nav.query(msg) # forward once
                # print(answer)

                act, thinking, raw_act = self.agent_nav.parse_llm_action(answer) # parse the action & thought
                print("action: ", act)
                if act == 0:
                    break

                self.agent.makeAction(act) # move

                # Update history
                if step_size == 0:
                    history[-1] = history[-1] + "\n" + "Thought: " + thinking + "\n" + "Action: " + raw_act + "\n"
                else:
                    history.append("Observation: " + obs_desc_summarized + "\n" + "Thought: " + thinking + "\n" + "Action: " + raw_act + "\n")

                cur_pos, cur_rot = self.agent.get_current_state()

                if act in [1, 4, 5]:
                    traj_len += 10.0

                step_size += 1

                dist = np.linalg.norm(cur_pos - target_pos)
                print(f"Task idx: {idx}, current step size: {step_size}, current dist: {dist}")

                if dist < 20:
                    break
                elif dist > 300:
                    break

            print(f"Max step size reached or target reached. step size: {step_size}")
            final_pos, final_rot = self.agent.get_current_state()
            dist = np.linalg.norm(final_pos - target_pos)
            if dist < 20: # 找到了
                if gt_traj_len > 100: # 判断是long还是short类型
                    SR_long_idx.append(idx)
                else:
                    SR_short_idx.append(idx)
                SR_count += 1
                SPL_count = gt_traj_len / max(gt_traj_len, traj_len)
                SPL += SPL_count

            ne_count += dist
            print(f"####### SR count: {SR_count}, SPL: {SPL}, NE: {ne_count}")
            # time.sleep(10)

        SR = SR_count / len(navi_data)
        NE = ne_count / len(navi_data)
        print(f"SR: {SR}, SPL: {SPL}, NE: {NE}")
        print(SR_short_idx)
        print(SR_long_idx)
        # print(f"SR_count: {SR_count}, NE_count: {ne_count}")


if __name__ == "__main__":

    model = "claude-3-haiku-20240307"  # LM models, for example: "claude-3-haiku-20240307", "gpt-4o"

    api_key = "" # to prevent the api from leakage

    print("Using model :", model)
    print("connecting to LM...")

    vln_eval = MyEmbodiedCityVLN("Datasets/vln", model, api_key)
    navi_data = vln_eval.navi_tasks
    vln_eval.run()
    # print(navi_data[0])