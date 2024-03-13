import argparse
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import time
from PIL import Image
import torch
from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from llava.conversation import (SeparatorStyle, conv_templates,
                                default_conversation)
from llava.mm_utils import (KeywordsStoppingCriteria, load_image_from_base64,
                            process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from transformers import TextIteratorStreamer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="Lin-Chen/ShareGPT4V-7B")
    parser.add_argument("--model-name", type=str,
                        default="llava-v1.5-7b")
    parser.add_argument("--file-path", type=str,
                        default=None, required=True)
    parser.add_argument("--folder-name", type=str,
                        default="sharegpt4v")
    parser.add_argument("--category", type=str,
                        default="color")
    parser.add_argument("--cot", action='store_true')
    args = parser.parse_args()
    return args

args = parse_args()
model_path = args.model_path
model_name = args.model_name
file_path = args.file_path

folder_name = args.folder_name
if not args.cot:
    folder_name += "_wocot"


os.makedirs(os.path.join(file_path, folder_name), exist_ok=True)
files = os.listdir(os.path.join(file_path, "samples"))

total = {}
score = {}

tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, False, False)


import matplotlib.pyplot as plt
def generate(model, input_ids, do_sample, temperature, top_p, max_new_tokens, streamer, stopping_criteria, image_args, stop_str):
    with torch.inference_mode():
        output_ids = model.generate(inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            **image_args
        )
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    return outputs

from tqdm import tqdm
import spacy

nlp = spacy.load("en_core_web_sm")
def split_prompt(prompt_name):
    global nlp
    doc = nlp(prompt_name)
    
    prompt = [] # save as the format as np, noun, adj.
    for chunk in doc.noun_chunks:
        # extract the noun and adj. separately
        chunk_np = chunk.text
        for token in list(chunk)[::-1]:
            if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                noun = token.lemma_
                adj = chunk_np.replace(f" {noun}", "")
                adj = adj.replace("the ", "")
                adj = adj.replace("a ", "")
                adj = adj.replace("an ", "")
                break
        prompt_each = [[f"{chunk_np}"], [f"{noun}"], [f"{adj}"]]

        prompt.append(prompt_each)
    return prompt

if args.category in ["color", "shape", "texture"]:
    map = {
            "1": 10,
            "2": 20,
            "3": 75,
            "4": 100,
        }
else:
    map = {
            "1": 20,
            "2": 40,
            "3": 60,
            "4": 80,
            "5": 100,
        }
    
for t, file in enumerate(tqdm(files)):
    img_id = int(file.split('.png')[0].split('_')[-1])
    prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n"
    img = [Image.open(os.path.join(file_path, "samples", file))]
    if args.cot:
        if args.category in ["color", "shape", "texture"]:
            prompt += f"You are my assistant to identify any objects and their {args.category} in the image. \
                Briefly describe what it is in the image within 50 words. ASSISTANT:"
        elif args.category == "spatial" or args.category == "3d_spatial":
            prompt += "You are my assistant to identify objects and their spatial layout in the image. \
                Briefly describe the image within 50 words. ASSISTANT:"
        elif args.category == "action":
            prompt += "You are my assistant to identify the actions, events, objects and their relationships in the image. \
                Briefly describe the image within 50 words. ASSISTANT:"
        elif args.category == "complex":
            prompt += "You are my assistant to evaluate the correspondence of the image to a given text prompt. \
                Briefly describe the image within 50 words, focus on the objects in the image and their attributes (such as color, shape, texture), \
                spatial layout and action relationships. ASSISTANT:"
        elif args.category == "numeracy":
            prompt += "You are my assistant to identify objects and their quantities in the image. \
                Briefly describe the image within 50 words, focus on the objects in the image and their quantity. ASSISTANT:"
        
    params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": 0.2,
        "top_p": 0.7,
        "max_new_tokens": 512,
        "stop": '</s>',
    }
    chunk = split_prompt(file.split('.png')[0].split('_')[0])
    img = process_images(img, image_processor, model.config)
    if type(img) is list:
        img = [i.to(model.device, dtype=torch.float16)
                    for i in img]
    else:
        img = img.to(model.device, dtype=torch.float16)

    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_context_length = getattr(
        model.config, 'max_position_embeddings', 2048)
    max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
    stop_str = params.get("stop", None)
    do_sample = True if temperature > 0.001 else False

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(
        keywords, tokenizer, input_ids)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)
    image_args = {
        'images': img,
    }
    total[img_id] = {}
    score[img_id] = {}

    if args.cot:
        outputs = generate(model, input_ids, do_sample, temperature, top_p, max_new_tokens, streamer, stopping_criteria, image_args, stop_str)
        total[img_id]['description'] = outputs
        prompt += outputs

    for i in range(len(chunk)):
        if args.category in ["color", "shape", "texture"]:
            query = f"According to the image and your previous answer, evaluate if there is {chunk[i][0][0]} in the image. \n \
                Give a score from 1 to 4, according the criteria:\n \
                4: there is {chunk[i][1][0]} in the image, and the {chunk[i][1][0]} is {chunk[i][2][0]}. \n \
                3: there is {chunk[i][1][0]} in the image, {chunk[i][1][0]} is mostly {chunk[i][2][0]}. \n \
                2: there is {chunk[i][1][0]} in the image, but {chunk[i][1][0]} is not {chunk[i][2][0]}. \n \
                1: no {chunk[i][1][0]} in the image. \n \
                Provide your analysis and explanation in JSON format with the following keys: score (e.g., 1), \n \
                explanation (within 20 words)."
        elif args.category in ["spatial", "3d_spatial"]:
            query = f"According to the image and your previous answer, evaluate if the text \"{file.split('.png')[0].split('_')[0]}\" is correctly portrayed in the image. \n \
                Give a score from 1 to 5, according the criteria: \n \
                1: image almost irrelevant to the text. \n \
                2: image not aligned properly with the text. \n \
                3: spatial layout not aligned properly with the text. \n \
                4: basically, spatial layout of objects matches the text. \n \
                5: correct spatial layout in the image for all objects mentioned in the text. \n \
                Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2), \n \
                explanation (within 20 words)."
        elif args.category == "action":
            query = f"According to the image and your previous answer, evaluate if the text \"{file.split('.png')[0].split('_')[0]}\" is correctly portrayed in the image. \
                Give a score from 1 to 5, according the criteria: \n \
                5: the image accurately portrayed the actions, events and relationships between objects described in the text. \n \
                4: the image portrayed most of the actions, events and relationships but with minor discrepancies. \n \
                3: the image depicted some elements, but action relationships between objects are not correct. \n \
                2: the image failed to convey the full scope of the text. \n \
                1: the image did not depict any actions or events that match the text. \n \
                Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2), \
                explanation (within 20 words)."
        elif args.category == "complex":
            query = f"According to the image and your previous answer, evaluate how well the image aligns with the text prompt: \"{file.split('.png')[0].split('_')[0]}\". \
                Give a score from 1 to 5, according the criteria: \n \
                5: the image perfectly matches the content of the text prompt, with no discrepancies. \n \
                4: the image portrayed most of the actions, events and relationships but with minor discrepancies. \n \
                3: the image depicted some elements in the text prompt, but ignored some key parts or details. \n \
                2: the image did not depict any actions or events that match the text. \n \
                1: the image failed to convey the full scope in the text prompt. \n \
                Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2), \
                explanation (within 20 words)."
        elif args.category == "numeracy":
            query = f" According to the image and your previous answer, evaluate how well the image aligns with the text prompt: \"{file.split('.png')[0].split('_')[0]}\" \
                Give a score from 1 to 5, according the criteria: \n\
                5: correct numerical content in the image for all objects mentioned in the text. \n \
                4: basically, numerical content of objects matches the text. \n  \
                3: numerical content not aligned properly with the text. \n \
                2: image not aligned properly with the text. \n \
                1: image almost irrelevant to the text. \n \
                Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2), explanation (within 20 words)."
            
        if args.cot:
            query_prompt = prompt + f"</s> USER: {query} ASSISTANT:"
        else:
            query_prompt = prompt + f"{query} ASSISTANT:"

        input_ids = tokenizer_image_token(
            query_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        
        outputs = generate(model, input_ids, do_sample, temperature, top_p, max_new_tokens, streamer, stopping_criteria, image_args, stop_str)
        total[img_id][f'ans{str(i+1)}'] = outputs
        score[img_id][f'ans{str(i+1)}'] = outputs

        if args.category in ["action", "spatial", "complex", "3d_spatial", "numeracy"]:
            break

    with open(os.path.join(file_path, folder_name, "total.json"), 'w') as f:
        json.dump(total, f, indent=4)

    with open(os.path.join(file_path, folder_name, "score.json"), 'w') as f:
        json.dump(score, f, indent=4)

total = 0
vqa = []
for k,v in score.items():
    score_i = 100
    for id, ans in v.items():
        if ans[-1] != "}":
            ans = ans + "\"\n}"
        try:
            js = json.loads(ans.replace("\n", ""))
            level = js["score"]
            score_i *= map[str(level)] / 100   
        except:
            continue
    vqa.append({"question_id": k, "answer": score_i})
    total += score_i

with open(os.path.join(file_path, folder_name, "vqa_result.json"), 'w') as f:
    json.dump(vqa, f)

with open(os.path.join(file_path, folder_name, "score.txt"), 'w') as f:
    f.write(f"total:{total} num:{len(score)} avg:{str(total / len(score))}")        