import base64
import requests
import re
import argparse
import os
import spacy
from tqdm import tqdm
import json
import time
import sys

# OpenAI API Key
api_key = "" # TODO add your api key

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation by GPT-4V.")
    parser.add_argument(
        "--image_path",
        type=str,
        default="examples/",
        help="Path to the image to be evaluated.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="color",
        help="Category of the image to be evaluated. eg. color, shape, texture, etc.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index of the image to be evaluated.",
    )
    parser.add_argument(
        "--step",
        type=int, 
        default=10, 
        help="Number of images to step.",
    )
    

    return parser.parse_args()

        
    

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    args = parse_args()
    
    # Path to your image
    try:
        image_path_total = os.path.join(args.image_path,"samples") # TODO
    except:
        image_path_total = args.image_path # TODO img path
    category = args.category
    start = args.start
    step = args.step
    image_file = os.listdir(image_path_total)
    image_file.sort(key=lambda x: int(x.split("_")[-1].split('.')[0]))
    
    gpt4v_record = []
    gpt4v_result = []
    for i in tqdm(range(start, len(image_file) ,step), desc="GPT-4V processing"):
        image_path = os.path.join(image_path_total, image_file[i])
        
        prompt_name = image_file[i].split("_")[0] # eg. "a green bench and a red car"
        

        if category == "color" or category == "shape" or category == "texture":
            # use spacy to extract the noun phrase
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(prompt_name)
            num_np = len(list(doc.noun_chunks))
            
            prompt = [] # save as the format as np, noun, adj.
            for chunk in doc.noun_chunks:
                # extract the noun and adj. separately
                chunk_np = chunk.text
                for token in list(chunk)[::-1]:
                    if token.pos_ == "NOUN":
                        noun = token.lemma_
                        adj = chunk_np.replace(f" {noun}", "")
                        adj = adj.replace("the ", "")
                        adj = adj.replace("a ", "")
                        adj = adj.replace("an ", "")
                        break
                prompt_each = [[f"{chunk_np}"], [f"{noun}"], [f"{adj}"]]

                prompt.append(prompt_each)
            
            question_for_gpt4v = []
            
            for k in range(num_np):
                text = f"You are my assistant to identify any objects and their {category} in the image. \
                    According to the image, evaluate if there is a {prompt[k][0][0]} in the image. \
                    Give a score from 0 to 100, according the criteria:\n\
                    4: there is {prompt[k][1][0]}, and {category} is {prompt[k][2][0]}.\n\
                    3: there is {prompt[k][1][0]}, {category} is mostly {prompt[k][2][0]}.\n\
                    2: there is {prompt[k][1][0]}, but it is not {prompt[k][2][0]}.\n\
                    1: no {prompt[k][1][0]} in the image.\n\
                    Provide your analysis and explanation in JSON format with the following keys: score (e.g., 1), \
                    explanation (within 20 words)."
                dic = {"type": "text", "text": text}
                question_for_gpt4v.append(dic)
                
        elif category == "spatial" or "3d_spatial":
            question_for_gpt4v = []
            num_np = 1
            text = f"You are my assistant to identify objects and their spatial layout in the image. \
                According to the image, evaluate if the text \"{prompt_name}\" is correctly portrayed in the image. \
                Give a score from 0 to 100, according the criteria: \n\
                    5: correct spatial layout in the image for all objects mentioned in the text. \
                    4: basically, spatial layout of objects matches the text. \
                    3: spatial layout not aligned properly with the text. \
                    2: image not aligned properly with the text. \
                    1: image almost irrelevant to the text. \
                    Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2), \
                    explanation (within 20 words)."
            dic = {"type": "text", "text": text}
            question_for_gpt4v.append(dic)   
        
        elif category == "action": 
            question_for_gpt4v = []
            num_np = 1
            text = f"You are my assistant to identify the actions, events, objects and their relationships in the image. \
            According to the image, evaluate if the text \"{prompt_name}\" is correctly portrayed in the image. \
            Give a score from 0 to 100, according the criteria: \n\
                5: the image accurately portrayed the actions, events and relationships between objects described in the text. \
                4: the image portrayed most of the actions, events and relationships but with minor discrepancies. \
                3: the image depicted some elements, but action relationships between objects are not correct. \
                2: the image failed to convey the full scope of the text. \
                1: the image did not depict any actions or events that match the text. \
                Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2), \
                explanation (within 20 words)."       
            dic = {"type": "text", "text": text}
            question_for_gpt4v.append(dic) 
        
        elif category == "numeracy":
            question_for_gpt4v = []
            num_np = 1
            text = f"You are my assistant to identify objects and their quantities in the image. \
            According to the image and your previous answer, evaluate how well the image aligns with the text prompt: \"{prompt_name}\" \
            Give a score from 0 to 100, according the criteria: \n\
                5: correct numerical content in the image for all objects mentioned in the text \
                4: basically, numerical content of objects matches the text \
                3: numerical content not aligned properly with the text \
                2: image not aligned properly with the text \
                1: image almost irrelevant to the text \
                Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2), explanation (within 20 words)."
            dic = {"type": "text", "text": text}
            question_for_gpt4v.append(dic) 
                            
        elif category == "complex":
            question_for_gpt4v = []
            num_np = 1
            text = "You are my assistant to evaluate the correspondence of the image to a given text prompt. \
                focus on the objects in the image and their attributes (such as color, shape, texture), spatial layout and action relationships. \
                According to the image and your previous answer, evaluate how well the image aligns with the text prompt: \"{prompt_name}\"  \
                        Give a score from 0 to 100, according the criteria: \n\
                        5: the image perfectly matches the content of the text prompt, with no discrepancies. \
                        4: the image portrayed most of the actions, events and relationships but with minor discrepancies. \
                        3: the image depicted some elements in the text prompt, but ignored some key parts or details. \
                        2: the image did not depict any actions or events that match the text. \
                        1: the image failed to convey the full scope in the text prompt. \
                        Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2), explanation (within 20 words)."
            dic = {"type": "text", "text": text}
            question_for_gpt4v.append(dic)     
                            
        
        
        # Getting the base64 string
        base64_image = encode_image(image_path)
        
        # question content
        content_list = [
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            },]+ [question_for_gpt4v[num_q] for num_q in range(num_np)]
            
        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }

        payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": content_list

            }
        ],
        "max_tokens": 300
        }
        max_attempts = 3  # Set the maximum number of attempts
        attempt_count = 0
        while True:
            try:
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

                # print(response.json())
                time.sleep(20)

                
                # Define a regular expression pattern to find all occurrences of the score
                pattern = r'"score": (\d+),'

                # Use re.findall to extract all scores as strings
                a=response.json()

                score_strings = re.findall(pattern, a["choices"][0]["message"]['content'])

                # Convert score strings to integers
                scores = [int(score) for score in score_strings]

                # Calculate the average score
                average_score = sum(scores) / len(scores) if len(scores) > 0 else 0
            
                break
            except:
                print("Error! Try again!")
                attempt_count += 1
                question_id = int(image_file[i].split("_")[-1].split('.')[0])
                # if exceed the RPD, stop the whole program
                if a["error"]["message"] == "Rate limit reached for gpt-4-vision-preview in organization org-GivOKvPvwSPhUtk2uGcbD1Ct on requests per day (RPD): Limit 2000, Used 2000, Requested 1. Please try again in 43.2s. Visit https://platform.openai.com/account/rate-limits to learn more.":
                    print("wait for 5 minutes, because of the RPD")
                    time.sleep(5*60)
                    if attempt_count >= max_attempts:
                        print("stop at question_id: ", question_id)
                        sys.exit()
                # Check if the maximum number of attempts has been reached
                if attempt_count >= max_attempts:
                    print(f"Exceeded maximum attempts ({max_attempts}). Exiting the loop. Error question_id: {question_id}")
                    average_score = 0
                    break  # Exit the loop even if the maximum attempts are not reached
                else:
                    continue
        
        # save image path and response to json
        outpath = os.path.join(args.image_path, "gpt4v")
        os.makedirs(outpath, exist_ok=True)
        
        gpt4v_record.append({"image_path": image_path, "response": response.json()})
        with open (f"{outpath}/gpt4v_record_{start}_{step}.json", "w") as f:
            json.dump(gpt4v_record, f)
        
        # save image number and score to json
        question_id = int(image_file[i].split("_")[-1].split('.')[0])
        gpt4v_result.append({"question_id": question_id, "answer": average_score})
        with open (f"{outpath}/gpt4v_result_{start}_{step}.json", "w") as f:
            json.dump(gpt4v_result, f)
    
    # calculate the avg
    score_list = [gpt4v_result[i]["answer"] for i in range(len(gpt4v_result))]
    avg_score = sum(score_list) / len(score_list)
    print(f"The average score is {avg_score}")
    print(f"Results save to {outpath}/gpt4v_result_{start}_{step}.json")
    with open (f"{outpath}/avg_score_{start}_{step}.txt", "w") as f:
        f.write(f"The average score is {avg_score}")

    
if __name__ == "__main__":
    main()   
    
