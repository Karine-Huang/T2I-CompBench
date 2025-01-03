import os

import torch
import json
import argparse

from word2number import w2n
try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml
# import spacy_universal_sentence_encoder
from experts.model_bank import load_expert_model
from experts.obj_detection.generate_dataset import Dataset, collate_fn
from accelerate import Accelerator
from tqdm import tqdm
import spacy
import numpy as np

obj_label_map = torch.load('dataset/detection_features.pt')['labels']


with open("../examples/dataset/new_objects.txt", "r") as f:
    objects = f.read().splitlines()
    object_s, object_p = [obj.split(" - ")[0].strip().lower() for obj in objects], [obj.split(" - ")[1].strip().lower() for obj in objects]

def get_mask_labels(depth, instance_boxes, instance_id):
    obj_masks = []
    obj_ids = []
    obj_boundingbox = []
    for i in range(len(instance_boxes)):
        is_duplicate = False
        mask = torch.zeros_like(depth)
        x1, y1, x2, y2 = instance_boxes[i][0].item(), instance_boxes[i][1].item(), \
                         instance_boxes[i][2].item(), instance_boxes[i][3].item()
        mask[int(y1):int(y2), int(x1):int(x2)] = 1
        if not is_duplicate:
            obj_masks.append(mask)
            obj_ids.append(instance_id[i])
            obj_boundingbox.append([x1, y1, x2, y2])

    instance_labels = {}
    for i in range(len(obj_ids)):
        instance_labels[i] = obj_ids[i].item()
    return obj_boundingbox, instance_labels


def calculate_iou(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    intersection_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    area_bbox1 = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
    area_bbox2 = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)

    iou = intersection_area / float(area_bbox1 + area_bbox2 - intersection_area)

    if iou > 0.9 or (intersection_area / float(area_bbox1) > 0.9) or (intersection_area / float(area_bbox2) > 0.9):
        return 1
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="UniDet evaluation.")
    parser.add_argument(
        "--outpath",
        type=str,
        default="../examples/",
        help="Path to output score",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model, transform = load_expert_model(task='obj_detection', ckpt="R50")
    accelerator = Accelerator(mixed_precision='fp16')

    outpath = args.outpath
    data_path= outpath

    batch_size = 64
    dataset = Dataset(data_path,  transform)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    cnt = 0
    score_map = []
    total_score = 0
    model, data_loader = accelerator.prepare(model, data_loader)
    nlp = spacy.load('en_core_web_sm')
    with torch.no_grad():
        for i, test_data in enumerate(tqdm(data_loader)):
            flag = 0

            test_pred = model(test_data)
            for k in range(len(test_pred)):
                instance_boxes = test_pred[k]['instances'].get_fields()['pred_boxes'].tensor  # get the bbox of list
                instance_id = test_pred[k]['instances'].get_fields()['pred_classes']
                depth = test_data[k]['image'][0]

                obj_bounding_box, obj_labels_dict = get_mask_labels(depth, instance_boxes, instance_id)

                obj = []  
                for i in range(len(obj_bounding_box)):
                    obj_name = obj_label_map[obj_labels_dict[i]]  
                    obj.append(obj_name)
                new_obj = []
                new_bbox = []
                for i in range(len(obj)):
                    flag = 0
                    for j in range(len(new_obj)):
                        if calculate_iou(obj_bounding_box[i], new_bbox[j]) and obj[i] == new_obj[j]:
                            flag = 1
                            break
                    if flag == 0:
                        new_obj.append(obj[i])
                        new_bbox.append(obj_bounding_box[i])

                img_path_split = test_data[k]['image_path'].split('/')
                 
                prompt = img_path_split[-1].split('_')[0] # get prompt from file names
            
                doc = nlp(prompt)
                number = ["a", "an", "one", "two", "three", "four", "five", "six", "seven", "eight"]
                num_obj = []
                my_obj = []
                for i in range(len(doc)):
                    if doc[i].text in number:
                        if (i < len(doc) - 2) and (doc[i+1].text + " " + doc[i+2].text in object_s or doc[i+1].text + " " + doc[i+2].text in object_p):
                            if doc[i+1].text + " " + doc[i+2].text in object_p and doc[i].text not in ["a", "an", "one"]:
                                my_obj.append(object_s[object_p.index(doc[i+1].text + " " + doc[i+2].text)])
                                try:
                                    num_obj.append(w2n.word_to_num(doc[i].text))
                                except:
                                    pass
                            else:
                                num_obj.append(1)
                                my_obj.append(doc[i+1].text + " " + doc[i+2].text)
                        elif doc[i+1].text in object_s or doc[i+1].text in object_p:
                            if doc[i+1].text in object_s and doc[i].text in ["a", "an", "one"]:
                                num_obj.append(1)
                                my_obj.append(doc[i+1].text)
                            else:
                                my_obj.append(object_s[object_p.index(doc[i+1].text)])
                                try:
                                    num_obj.append(w2n.word_to_num(doc[i].text))
                                except:
                                    pass
                score = 0
                weight = 1.0 / len(my_obj)             
                for i, my_obj_i in enumerate(my_obj):
                    if my_obj_i in ["boy", "girl", "man", "woman"]:
                        my_obj_i = "person"
                    if my_obj_i == "ship":
                        my_obj_i = "boat"
                    if my_obj_i == "telivision":
                        my_obj_i = "tv"
                    if my_obj_i == "goldfish":
                        my_obj_i = "fish"
                    if my_obj_i == "painting":
                        my_obj_i = "picture"

                    if my_obj_i not in new_obj:
                        for j, obj_i in enumerate(new_obj):
                            if my_obj_i in obj_i:
                                new_obj[j] = my_obj_i

                    if my_obj_i in new_obj:
                        score += 0.5* weight
                        num_det = new_obj.count(my_obj_i)
                        if num_det == num_obj[i]:
                            score += 0.5* weight
                
                from copy import copy
                score_map.append({"question_id": int(img_path_split[-1].split(".png")[0].split("_")[1]), "answer": score})
                cnt += 1
                total_score += score
    
        os.makedirs(os.path.join(args.outpath, "annotation_num"), exist_ok=True)
        p = os.path.join(args.outpath, "annotation_num")
        with open(os.path.join(p, 'vqa_result.json'), 'w') as f:
            json.dump(score_map, f)

        with open(os.path.join(p, 'score.txt'), 'w') as f:
            f.write(f"total:{total_score} num:{cnt} avg:{str(total_score / cnt)}")


if __name__ == "__main__":
    main()





