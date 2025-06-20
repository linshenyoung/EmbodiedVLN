
# My EmbodiedCity Visual Language Navigation Agent
# 2025/6

from vln.agent import AirsimAgent
from utils import encode_image, LM_VLN
# from langchain.chains.llm import LLMChain
# from langchain.llms.openai import OpenAI
# from langchain.chat_models import ChatOpenAI
# from langchain.chat_models import ChatAnthropic
from langchain.prompts import PromptTemplate
import anthropic
import gc
import math
import time
import random
import torch
from vln.evaluate import get_metrics_from_results
from vln.agent import Agent
from vln.env import get_gold_nav
from vln.prompt_builder import get_navigation_lines
from tqdm import tqdm
from http import HTTPStatus
import dashscope
import os
import cv2
import numpy as np
from PIL import Image
import base64
from openai import OpenAI
from prompts.prompt2 import build_prompt
from prompts.prompt_SUM import build_prompt_SUM
from prompts.prompt_PAR import build_prompt_PAR
from prompts.prompt_NAV import build_prompt_NAV
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from scipy.stats import bootstrap
import nltk
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from prompts.prompt_template import (
    NAVIGATION_PROMPT,
    HISTORY_PROMPT,
    PARSE_IMAGE_PROMPT,
    SUMMARIZE_OBS_DESC
)
import re
from BaseAgent import BaseAgent


class EmbodiedNavAgent(BaseAgent):

    """
    Large Model Agent for EmbodiedCity VLN task
    """
    def __init__(self, model, api_key, root_dir): # init the agent and get the llm client
        """
        :param model: LM model
        :param api_key: api key corresponding to the LM model
        """
        self.image_path = ["imgs/1.jpg", "imgs/2.jpg", "imgs/3.jpg", "imgs/4.jpg"]
        self.model = model
        self.root_dir = root_dir
        self.api_key = api_key
        self.model_class = model.split('-')[0]
        if self.model_class == 'claude':
            self.llm_client = anthropic.Anthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key=api_key,
                base_url="https://api.gptsapi.net",
            )
        elif self.model_class == 'gpt':
            self.llm_client = OpenAI(
                api_key=api_key,
            )
        else:
            raise ValueError(f"Unknown evaluation model type {self.eval_model}")
        

    def get_prompt(self, template, fillin):

        if template == "NAVIGATION_PROMPT":
            prompt = build_prompt_NAV(fillin)
        #     template_text = NAVIGATION_PROMPT
        #     variable = ["navi_desc", "history", "observation"]
        elif template == "HISTORY_PROMPT":
            pass
        #     template_text = HISTORY_PROMPT
        #     variable = ["history", "previous_action", "observation"]
        elif template == "SUMMARIZE_OBS_DESC":
            prompt = build_prompt_SUM(fillin)
        #     template_text = SUMMARIZE_OBS_DESC
        #     variable = ["Horizontal_RGB", "Horizontal_Depth", "Top_RGB", "Top_Depth"]
        else:
            prompt = build_prompt_PAR()
        #     template_text = PARSE_IMAGE_PROMPT
        #     variable = []

        # # tmp = template.strip(')

        # prompt = PromptTemplate(template=template_text, input_variables=variable)

        # if template == "NAVIGATION_PROMPT":
        #     prompt.format(navi_desc=fillin[0],
        #                  history=fillin[1],
        #                  observation=fillin[2],
        #                  )
        # elif template == "HISTORY_PROMPT":
        #     prompt.format(history=fillin[0],
        #                  previous_action=fillin[1],
        #                  observation=fillin[2],
        #                  )
        # elif template == "SUMMARIZE_OBS_DESC":
        #     prompt.format(Horizontal_RGB=fillin[0],
        #                   Horizontal_Depth=fillin[1],
        #                   Top_RGB=fillin[2],
        #                   Top_Depth=fillin[3],
        #                   )

        # # llm = LLMChain(llm=self.llm, prompt=prompt)

        # # if template == "NAVIGATION_PROMPT":
        # #     output = llm.run(navi_desc=fillin[0],
        # #                  history=fillin[1],
        # #                  observation=fillin[2],
        # #                 )
        # # else:
        # #     output = llm.run(history=fillin[0],
        # #                  previous_action=fillin[1],
        # #                  observation=fillin[2],
        # #                 )
        
        return prompt
        

    def get_image(self, agent_handler):

        img1 = agent_handler.get_xyg_image(image_type=0, cameraID="0")  # 获取前景图
        img1 = Image.fromarray(img1, 'RGB')
        img1.save('imgs/1.jpg', format="jpeg")
        img2 = agent_handler.get_xyg_image(image_type=1, cameraID="0")  # 获取前景深度图
        cv2.imwrite('imgs/2.jpg', img2)
        img1 = agent_handler.get_xyg_image(image_type=0, cameraID="3")  # 获取俯视图
        img1 = Image.fromarray(img1, 'RGB')
        img1.save('imgs/3.jpg', format="jpeg")
        img2 = agent_handler.get_xyg_image(image_type=1, cameraID="3")  # 获取俯视深度图
        cv2.imwrite('imgs/4.jpg', img2)


    def encode_images(self):

        base64_image1 = encode_image(self.image_path[0])
        base64_image2 = encode_image(self.image_path[1])
        base64_image3 = encode_image(self.image_path[2])
        base64_image4 = encode_image(self.image_path[3])

        encoded_imgs = [base64_image1, base64_image2, base64_image3, base64_image4]

        return encoded_imgs
        

    def parse_images(self, encoded_imgs):

        """
            input : images' base64 encoded string
            output : for instance, is a directory of different views and descriptions.
                [
                    'a kitchen with a yellow chair and black tiled floor.',
                    'The chair is closer while the black tiled floor is far away from the view.',
                    'a large window with a white shade.',
                    'The shade is close to the view.'
                ]
        """
        messages = []

        UserContent = self.get_prompt("PARSE_IMAGE_PROMPT", [])

        inputGPT_imgs = [
            {
                "type": "text",
                "text": UserContent
            }
        ]

        image1_media_type = "image/jpg"

        for encoded_img in encoded_imgs:
            inputGPT_imgs += [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image1_media_type,
                        "data": encoded_img
                    }
                }
            ]

        messages.append({"role": "user", "content": inputGPT_imgs})

        try:
            chat_response = self.llm_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                messages=messages,
            )
            answer = chat_response.content[0].text

            print("Parse Images Answer: ", answer)
        except Exception as e:
            print(f'❌ Error: LM response - {type(e).__name__}: {e}') # 如果出现LLM使用错误，返回报错原因
            answer = "Debug_parse_images"

        # matches = re.findall(r'\w+:\s*(.*?)(?=\n\w+:|$)', answer, re.DOTALL)

        # 去除每项首尾空格并组成列表
        # result_list = [m.strip() for m in matches]
        result_list = re.split(r'(?<=\.)\s+', answer.strip())
        while len(result_list) < 4:
            result_list.append("MISSING")
        # print(result_list)
        return result_list


    def summarizer(self, obs_desc):

        messages = []
        sum_prompt = self.get_prompt("SUMMARIZE_OBS_DESC", obs_desc)

        inputGPT = [
            {
                "type": "text",
                "text": sum_prompt
            }
        ]
        # llm = ChatAnthropic(api_key=self.api_key,
        #                     base_url="https://api.gptsapi.net",
        #                     model="claude-3-haiku-20240307")
        # llm_chain = LLMChain(llm=llm, prompt=sum_prompt)
        # summarized = llm_chain.run(Horizontal_RGB=obs_desc[0],
        #                            Horizontal_Depth=obs_desc[1],
        #                            Top_RGB=obs_desc[2],
        #                            Top_Depth=obs_desc[3])

        messages.append({"role": "user", "content": inputGPT})

        try:
            chat_response = self.llm_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                messages=messages,
            )
            answer = chat_response.content[0].text

            print("Summary Answer: ", answer)
        except Exception as e:
            print(f'❌ Error: LM response - {type(e).__name__}: {e}') # 如果出现LLM使用错误，返回报错原因
            answer = "Debug_parse_images"


        return answer # summarized answer


    def prompt_manager(self, fillin, encoded_imgs, messages):

        UserContent = self.get_prompt("NAVIGATION_PROMPT", fillin)
        
        if self.model_class == 'claude':

            inputGPT = [
                {
                    "type": "text",
                    "text": UserContent
                }
            ]
            image1_media_type = "image/jpg"
            for encoded_img in encoded_imgs:
                inputGPT += [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image1_media_type,
                            "data": encoded_img
                        }
                    }
                ]

            messages.append({"role": "user", "content": inputGPT})

        else:
            messages.append({"role": "user", "content": [
                {
                    "type": "text",
                    "text": UserContent
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_imgs[0]}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_imgs[1]}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_imgs[2]}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_imgs[3]}"
                    }
                },
            ]})

        return messages
        

    def query(self, messages):

        # Access according to the official API input format of different models
        if self.model_class == 'claude':

            try:
                chat_response = self.llm_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    messages=messages,
                )
                answer = chat_response.content[0].text

                print("query answer :", answer)
            except Exception as e:
                print(f'❌ Error: LM response - {type(e).__name__}: {e}') # 如果出现LLM使用错误，返回报错原因
                answer = "action: moveForth"

            # answer = chat_response.content[0].text
            # # print(f'ChatGPT: {answer}')
            # messages.append({"role": "assistant", "content": chat_response.content})

        elif self.model_class == 'gpt':

            # else:
            #     UserContent = input("请输入（输入 'quit' 结束）：")
            #     if UserContent == 'quit':
            #         break
            #     UserContent += "Current status:\nObservation of drones: Pan tilt angle  and 90 degrees.\nCommand:"
            # # What’s in this image?
            # What’s the similarity between this image and the previous image?

            try:
                # time.sleep(5)

                # models = self.llm_client.models.list()
                # for m in models:
                #     print(m.id) # 查询api的可用model

                chat_response = self.llm_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=4000,
                )
                answer = chat_response.choices[0].message.content
                # print(f'ChatGPT: {answer}')
                messages.append({"role": "assistant", "content": answer})

                print("query answer :", answer)
            except Exception as e:
                print(f'❌ Error: LM response - {type(e).__name__}: {e}') # 如果出现LLM使用错误，返回报错原因
                answer = "action: moveForth"

        return answer