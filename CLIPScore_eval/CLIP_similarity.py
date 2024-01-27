# image and text similarity
# ref https://github.com/openai/CLIP
import os
import torch
import clip
from PIL import Image
import spacy
nlp=spacy.load('en_core_web_sm')

import json
import argparse



device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outpath",
        type=str,
        default=None,
        required=True,
        help="Path to read samples and output scores"
    )
    parser.add_argument(
        "--complex",
        type=bool,
        default=False,
        help="To evaluate on samples in complex category or not"
    )
    args = parser.parse_args()
    return args





def main():
    args = parse_args()

    outpath=args.outpath

    image_folder=os.path.join(outpath,'samples')
    file_names = os.listdir(image_folder)
    file_names.sort(key=lambda x: int(x.split("_")[-1].split('.')[0]))  # sort

    cnt = 0
    total = []

    # output annotation.json
    for file_name in file_names:


        image_path = os.path.join(image_folder,file_name)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        prompt = file_name.split("_")[0]

        if (args.complex):
            doc=nlp(prompt)
            prompt_without_adj=' '.join([token.text for token in doc if token.pos_ != 'ADJ']) #remove adj
            text = clip.tokenize(prompt_without_adj).to(device)
        else:
            text = clip.tokenize(prompt).to(device)




        with torch.no_grad():
            image_features = model.encode_image(image.to(device))
            image_features /= image_features.norm(dim=-1, keepdim=True)


            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)

             # Calculate the cosine similarity between the image and text features
            cosine_similarity = (image_features @ text_features.T).squeeze().item()

        similarity = cosine_similarity
        cnt+=1
        if (cnt % 100 == 0):
            print(f"CLIP image-text:{cnt} prompt(s) have been processed!")
        total.append(similarity)


    #save
    sim_dict=[]
    for i in range(len(total)):
        tmp={}
        tmp['question_id']=i
        tmp["answer"] = total[i]
        sim_dict.append(tmp)


    json_file = json.dumps(sim_dict)
    savepath = os.path.join(outpath,"annotation_clip") #todo
    os.makedirs(savepath, exist_ok=True)
    with open(f'{savepath}/vqa_result.json', 'w') as f:
        f.write(json_file)
    print(f"save to {savepath}")
    
    # score avg
    score=0
    for i in range(len(sim_dict)):
        score+=float(sim_dict[i]['answer'])
    with open(f'{savepath}/score_avg.txt', 'w') as f:
        f.write('score avg:'+str(score/len(sim_dict)))
    print("score avg:", score/len(sim_dict))

if __name__ == "__main__":
    main()






