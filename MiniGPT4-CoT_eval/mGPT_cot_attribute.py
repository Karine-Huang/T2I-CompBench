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
        default="color",
        help="color, shape or texture",
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
        default=0.7,
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
    header = ['image name', 'noun phrase', 'scores of phrases', 'chat', 'score of image']
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
    start=args.start

    for i in range(start,len(img_names),step):

        f = img_names[i].split('_')[0]
        doc = nlp(f)
        noun_list = list(doc.noun_chunks)
        row = [img_names[i], noun_list]
        res = []
        score = 0.0
        chatbot = []
        img_path = img_file_path + img_names[i]

        Q1 = "You are my assistant to identify any objects and their {} in the image. Briefly describe what it is in the image within 50 words".format(category)

        msg, chat_state, img_list = upload_img(img_path)
        chatbot, chat_state = ask(Q1, chatbot, chat_state)
        output_answer, chatbot, chat_state, img_list = answer(chatbot, chat_state, img_list, num_beams, temperature)

        for j in range(len(noun_list)):
            np = noun_list[j].text
            doc_np = nlp(np)
            for token in list(doc_np)[::-1]:
                if token.pos_ == "NOUN":
                    noun = token.lemma_
                    adj = np.replace(f" {noun}", "")
                    adj = adj.replace("the ", "")
                    adj = adj.replace("a ", "")
                    adj = adj.replace("an ", "")
                    break
            if j == 0:  #first noun phrase
                text_input = "According to the image and your previous answer, evaluate if there is {np} in the image?" \
                         "Give a score from 0 to 100, according the criteria:" \
                         "\n 100: there is {noun}, and {noun} is {adj}." \
                         "\n 75: there is {noun}, {noun} is mostly {adj}." \
                         "\n 20: there is {noun}, but it is not {adj}." \
                         "\n 10: no {noun} in the image." \
                         "\n Provide your analysis and explanation in JSON format with the following keys: score (e.g., 85)," \
                         " explanation (within 20 words). ".format(np=noun_list[j],noun=noun,adj=adj)
            else:
                text_input = "Please evaluate if there is {} in the image according to the same criteria. " \
                             "Provide your analysis and explanation in JSON format with the following keys: score (e.g., 85), " \
                             "explanation (within 20 words)".format(noun_list[j])

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

            score += score_1q_acc/ask_number
            res.append(str(score_1q_acc/ask_number))

        row.append(', '.join(res)) #scores for noun phrases
        row.append(chatbot)
        row.append(score/len(noun_list))
        score_acc += score/len(noun_list)
        writer.writerow(row)

    score_avg = score_acc/len(range(start,len(img_names),step))
    writer.writerow(['avarage score of all images:', score_avg])




