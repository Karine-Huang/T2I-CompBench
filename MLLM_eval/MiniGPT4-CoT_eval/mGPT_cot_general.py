# This script is based on https://github.com/Vision-CAIR/MiniGPT-4/blob/main/demo.py

import argparse
import os
import random
import json

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

import spacy
import en_core_web_sm
import csv
import re



def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument(
        "--cfg-path",
        required=True,
        default="eval_configs/minigpt4_eval.yaml",
        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="spatial",
        help="spatial, non-spatial, or complex",
    )
    parser.add_argument(
        "--img_file",
        type=str,
        help="load images from this file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="location of output file",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="mGPT_cot_output.csv",
        help="name of output file",
    )
    parser.add_argument(
        "--max_ask",
        type=int,
        default=3,
        help="if chatbot does not respond, maximum number of enquiries to make",
    )
    parser.add_argument(
        "--ask_number",
        type=int,
        default=1,
        help="the number for which a question will be repeatedly asked",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="beam search",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="temperature",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="step to sample images",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="start point of list of images",
    )

    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return chat_state, img_list

def upload_img(img_path):
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(img_path, chat_state, img_list) # chat_state is conv; msg = "Received."; img_list.append(image_emb)
    return llm_message, chat_state, img_list

def ask(user_message, chatbot, chat_state):
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return chatbot, chat_state,


def answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    return llm_message, chatbot, chat_state, img_list

if __name__ == "__main__":
    print('Initializing Chat')
    args = parse_args()
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print('Initialization Finished')

    nlp = spacy.load("en_core_web_sm")
    img_file_path = args.img_file
    output_path = args.output_path
    output_name = args.output_name
    csv_f = open(os.path.join(output_path, output_name), 'w', newline='')
    writer = csv.writer(csv_f)
    header = ['image name', 'caption', 'chat', 'score of image']
    writer.writerow(header)

    score_acc = 0.0
    img_names = os.listdir(img_file_path)
    img_names.sort(key=lambda x: int(x.split("_")[1].split('.')[0]))
    category = args.category
    num_beams = args.num_beams
    temperature = args.temperature
    ask_number = args.ask_number
    max_ask = args.max_ask
    step = args.step
    start = args.start

    for i in range(start,len(img_names),step):

        f = img_names[i].split('_')[0]
        row = [img_names[i], f]
        score = 0.0
        chatbot = []
        img_path = img_file_path + img_names[i]
        if category == "spatial":
            Q1 = "You are my assistant to identify objects and their spatial layout in the image. Briefly describe the image within 50 words."

            text_input = "According to the image and your previous answer, evaluate if the text \"{}\" is correctly portrayed in the image.".format(f) + \
                         " Give a score from 0 to 100, according the criteria:" \
                         "\n 100: correct spatial layout in the image for all objects mentioned in the text" \
                         "\n 80: basically, spatial layout of objects matches the text" \
                         "\n 60: spatial layout not aligned properly with the text" \
                         "\n 40: image not aligned properly with the text" \
                         "\n 20: image almost irrelevant to the text" \
                         "\n Provide your analysis and explanation in JSON format with the following keys: score (e.g., 85), explanation (within 20 words)."

        elif category == "non-spatial":
            Q1 = "You are my assistant to identify the actions, events, objects and their relationships in the image. Briefly describe the image within 50 words."

            text_input = "According to the image and your previous answer, evaluate if the text \"{}\" is correctly portrayed in the image.".format(f) + \
                         " Give a score from 0 to 100, according the criteria:" \
                         "\n 100: the image accurately portrayed the actions, events and relationships between objects described in the text." \
                         "\n 80: the image portrayed most of the actions, events and relationships but with minor discrepancies." \
                         "\n 60: the image depicted some elements, but action relationships between objects are not correct." \
                         "\n 40: the image failed to convey the full scope of the text." \
                         "\n 20: the image did not depict any actions or events that match the text." \
                         "\n Provide your analysis and explanation in JSON format with the following keys: score (e.g., 85), explanation (within 20 words)."

        elif category == "complex":
            Q1 = "You are my assistant to evaluate the correspondence of the image to a given text prompt. Briefly describe the image within 50 words, " \
                 "focus on the objects in the image and their attributes (such as color, shape, texture), spatial layout and action relationships."

            text_input = "According to the image and your previous answer, evaluate how well the image aligns with the text prompt: \"{}\".".format(f) + \
                         " Give a score from 0 to 100, according the criteria:" + \
                         "\n 100: the image perfectly matches the content of the text prompt, with no discrepancies." + \
                         "\n 80: the image portrayed most of the actions, events and relationships but with minor discrepancies." + \
                         "\n 60: the image depicted some elements in the text prompt, but ignored some key parts or details." + \
                         "\n 40: the image did not depict any actions or events that match the text." \
                         "\n 20: the image failed to convey the full scope in the text prompt." \
                         "\n Provide your analysis and explanation in JSON format with the following keys: score (e.g., 85), explanation (within 20 words)."

        else:
            raise ValueError("Please specify a correct category")

        msg, chat_state, img_list = upload_img(img_path)
        chatbot, chat_state = ask(Q1, chatbot, chat_state)  # conv.append_message(conv.roles[0], text)
        output_answer, chatbot, chat_state, img_list = answer(chatbot, chat_state, img_list, num_beams, temperature)

        score_1q_acc = 0
        for k in range(ask_number):
            chatbot, chat_state = ask(text_input, chatbot, chat_state)
            output_answer, chatbot, chat_state, img_list = answer(chatbot, chat_state, img_list, num_beams,temperature)
            output_answer = output_answer.lower()
            output_answer_list = [item for item in re.split("(\W)", output_answer) if len(item) > 0]
            ask_count = 0
            clear_answer = False
            s = None
            for idx in range(len(output_answer_list)):
                if output_answer_list[idx].isnumeric():
                    if 0.0 <= float(output_answer_list[idx]) <= 100.0:
                        s = float(output_answer_list[idx])
                        break

            if s is not None:
                clear_answer = True
                score_1q_acc += s

            else:
                while ask_count < max_ask and clear_answer is False:
                    chatbot, chat_state = ask(text_input, chatbot, chat_state)
                    output_answer, chatbot, chat_state, img_list = answer(chatbot, chat_state, img_list, num_beams,temperature)
                    output_answer = output_answer.lower()
                    output_answer_list = [item for item in re.split("(\W+)", output_answer) if len(item) > 0]
                    ask_count += 1

                    s = None
                    for idx in range(len(output_answer_list)):
                        if output_answer_list[idx].isnumeric():
                            if 0.0 <= float(output_answer_list[idx]) <= 100.0:
                                s = float(output_answer_list[idx])
                                break

                    if s is not None:
                        clear_answer = True
                        score_1q_acc += s

        score = score_1q_acc/ask_number

        row.append(chatbot)
        row.append(score)
        score_acc += score
        writer.writerow(row)

    score_avg = score_acc/len(range(start,len(img_names),step))
    writer.writerow(['avarage score of all images:', score_avg])




