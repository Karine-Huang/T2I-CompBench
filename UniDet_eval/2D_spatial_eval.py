
import os

import torch
import json
import argparse


try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml

from experts.model_bank import load_expert_model
from experts.obj_detection.generate_dataset import Dataset, collate_fn
from accelerate import Accelerator
from tqdm import tqdm
import spacy
import numpy as np

obj_label_map = torch.load('dataset/detection_features.pt')['labels']

def determine_position(locality, box1, box2, iou_threshold=0.1,distance_threshold=150):
    # Calculate centers of bounding boxes
    box1_center = ((box1['x_min'] + box1['x_max']) / 2, (box1['y_min'] + box1['y_max']) / 2)
    box2_center = ((box2['x_min'] + box2['x_max']) / 2, (box2['y_min'] + box2['y_max']) / 2)

    # Calculate horizontal and vertical distances
    x_distance = box2_center[0] - box1_center[0]
    y_distance = box2_center[1] - box1_center[1]

    # Calculate IoU
    x_overlap = max(0, min(box1['x_max'], box2['x_max']) - max(box1['x_min'], box2['x_min']))
    y_overlap = max(0, min(box1['y_max'], box2['y_max']) - max(box1['y_min'], box2['y_min']))
    intersection = x_overlap * y_overlap
    box1_area = (box1['x_max'] - box1['x_min']) * (box1['y_max'] - box1['y_min'])
    box2_area = (box2['x_max'] - box2['x_min']) * (box2['y_max'] - box2['y_min'])
    union = box1_area + box2_area - intersection
    iou = intersection / union

    # Determine position based on distances and IoU and give a soft score
    score=0
    if locality in ['next to', 'on side of', 'near']:
        if (abs(x_distance)< distance_threshold or abs(y_distance)< distance_threshold):
            score=1
        else:
            score=distance_threshold/max(abs(x_distance),abs(y_distance))
    elif locality == 'on the right of':
        if x_distance < 0:
            if abs(x_distance) > abs(y_distance) and iou < iou_threshold:
                score=1
            elif abs(x_distance) > abs(y_distance) and iou >= iou_threshold:
                score=iou_threshold/iou
        else:
            score=0
    elif locality == 'on the left of':
        if x_distance > 0:
            if abs(x_distance) > abs(y_distance) and iou < iou_threshold:
                score=1
            elif abs(x_distance) > abs(y_distance) and iou >= iou_threshold:
                score=iou_threshold/iou
        else:
            score=0
    elif locality =='on the bottom of':
        if y_distance < 0:
            if abs(y_distance) > abs(x_distance) and iou < iou_threshold:
                score=1
            elif abs(y_distance) > abs(x_distance) and iou >= iou_threshold:
                score=iou_threshold/iou
    elif locality =='on the top of':
        if y_distance > 0:
            if abs(y_distance) > abs(x_distance) and iou < iou_threshold:
                score=1
            elif abs(y_distance) > abs(x_distance) and iou >= iou_threshold:
                score=iou_threshold/iou
    else:
        score=0
    return score



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


#obj detection
def parse_args():
    parser = argparse.ArgumentParser(description="UniDet evaluation.")
    parser.add_argument(
        "--outpath",
        type=str,
        default="../examples/",
        help="Path to output score",
    )
    parser.add_argument(
        "--complex",
        type=bool,
        default=False,
        help="Prompt is simple structure or in complex category",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model, transform = load_expert_model(task='obj_detection', ckpt="RS200")
    accelerator = Accelerator(mixed_precision='fp16')

    config = yaml.load(open('configs/experts.yaml', 'r'), Loader=yaml.Loader)
    outpath = args.outpath
    data_path= outpath
    save_path= f'{outpath}/labels'

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

    model, data_loader = accelerator.prepare(model, data_loader)

    with torch.no_grad():
        result = []
        map_result = []
        for i, test_data in enumerate(tqdm(data_loader)):
            test_pred = model(test_data)
            for k in range(len(test_pred)):
                instance_boxes = test_pred[k]['instances'].get_fields()['pred_boxes'].tensor  # get the bbox of list
                instance_id = test_pred[k]['instances'].get_fields()['pred_classes']
                depth = test_data[k]['image'][0]

                # get score
                instance_score = test_pred[k]['instances'].get_fields()['scores']

                obj_bounding_box, obj_labels_dict = get_mask_labels(depth, instance_boxes, instance_id)

                obj = []  
                for i in range(len(obj_bounding_box)):
                    obj_name = obj_label_map[obj_labels_dict[i]]  
                    obj.append(obj_name)


                img_path_split = test_data[k]['image_path'].split('/')
                prompt = img_path_split[-1].split('_')[0] # get prompt from file names
                vocab_spatial = ['on side of', 'next to', 'near', 'on the left of', 'on the right of', 'on the bottom of', 'on the top of','on top of'] #locality words

                locality = None
                for word in vocab_spatial:
                    if word in prompt:
                        locality = word
                        break

                if (args.complex):
                    #for complex structure
                    nlp = spacy.load('en_core_web_sm')
                    # Define the sentence
                    sentence = prompt
                    # Process the sentence using spaCy
                    doc = nlp(sentence)
                    # Define the target prepositions
                    prepositions = ["on top of", "on bottom of", "on the left", "on the right",'next to','on side of','near']
                    # Extract objects before and after the prepositions
                    objects = []
                    for i in range(len(doc)):
                        if doc[i:i + 3].text in prepositions or doc[i:i + 2].text in prepositions or doc[i:i + 1].text in prepositions:
                            if doc[i:i + 3].text in prepositions:
                                k=3
                            elif doc[i:i + 2].text in prepositions:
                                k=2
                            elif doc[i:i + 1].text in prepositions:
                                k=1
                            preposition_phrase = doc[i:i + 3].text
                            for j in range(i - 1, -1, -1):
                                if doc[j].pos_ == 'NOUN':
                                    objects.append(doc[j].text)
                                    break
                                elif doc[j].pos_ == 'PROPN':
                                    objects.append(doc[j].text)
                                    break
                            flag=False
                            for j in range(i + k, len(doc)):
                                if doc[j].pos_ == 'NOUN':
                                    objects.append(doc[j].text)
                                    break
                                if(j==len(doc)-1):
                                    flag=True 
                            if flag:
                                for j in range(i + k, len(doc)):
                                    if (j+1<len(doc)) and doc[j].pos_ == 'PROPN' and doc[j+1].pos_ != 'PROPN':
                                        objects.append(doc[j].text)
                                        break
                    if (len(objects)==2):
                        obj1=objects[0]
                        obj2=objects[1]
                    else:
                        obj1=None
                        obj2=None
                else:
                    #for simple structure
                    nlp = spacy.load("en_core_web_sm")
                    doc = nlp(prompt)
                    obj1= [token.text for token in doc if token.pos_=='NOUN'][0]
                    obj2= [token.text for token in doc if token.pos_=='NOUN'][-1]

                person = ['girl','boy','man','woman']
                if obj1 in person:
                    obj1 = "person"
                if obj2 in person:
                    obj2 = "person"
                if obj1 in obj and obj2 in obj:
                    obj1_pos = obj.index(obj1)
                    obj2_pos = obj.index(obj2)
                    obj1_bb = obj_bounding_box[obj1_pos]
                    obj2_bb = obj_bounding_box[obj2_pos]
                    box1, box2={},{}

                    box1["x_min"] = obj1_bb[0]
                    box1["y_min"] = obj1_bb[1]
                    box1["x_max"] = obj1_bb[2]
                    box1["y_max"] = obj1_bb[3]
                    box2["x_min"] = obj2_bb[0]
                    box2["y_min"] = obj2_bb[1]
                    box2["x_max"] = obj2_bb[2]
                    box2["y_max"] = obj2_bb[3]


                    score = 0.25 * instance_score[obj1_pos].item() + 0.25 * instance_score[obj2_pos].item()  # score = avg across two objects score
                    score += determine_position(locality, box1, box2) / 2
                elif obj1 in obj:
                    obj1_pos = obj.index(obj1)  
                    score = 0.25 * instance_score[obj1_pos].item()
                elif obj2 in obj:
                    obj2_pos = obj.index(obj2)
                    score = 0.25 * instance_score[obj2_pos].item()
                else:
                    score = 0
                if (score<0.5):
                    score=0

                image_dict = {}
                image_dict['question_id']=int(img_path_split[-1].split('_')[-1].split('.')[0])
                image_dict['answer'] = score
                result.append(image_dict)

                # add mapping
                map_dict = {}
                map_dict['image'] = img_path_split[-1]
                map_dict['question_id']=int(img_path_split[-1].split('_')[-1].split('.')[0])
                map_result.append(map_dict)
        

        im_save_path = os.path.join(save_path, 'annotation_obj_detection_2d')
        os.makedirs(im_save_path, exist_ok=True)

        with open(os.path.join(im_save_path, 'vqa_result.json'), 'w') as f:
            json.dump(result, f)
        print('vqa result saved in {}'.format(im_save_path))

        # avg score
        avg_score = 0
        for i in range(len(result)):
            avg_score+=float(result[i]['answer'])
        with open(os.path.join(im_save_path, 'avg_score.txt'), 'w') as f:
            f.write('score avg:'+str(avg_score/len(result)))
        print("avg score:",avg_score/len(result))
        
        # save mapping
        with open(os.path.join(im_save_path, 'mapping.json'), 'w') as f:
            json.dump(map_result, f)

if __name__ == "__main__":
    main()





