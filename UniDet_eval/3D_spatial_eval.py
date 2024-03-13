# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE



import matplotlib.pyplot as plt
import os

import torch
import os
import json
import copy
import PIL.Image as Image
import glob
import random
import spacy

try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml

from experts.model_bank_3d import load_expert_model
from experts.obj_detection.generate_dataset_3d import Dataset, collate_fn
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np
import argparse
from experts.depth.generate_dataset import Dataset as Dataset_depth

obj_label_map = torch.load('dataset/detection_features.pt')['labels']

def parse_args():
    parser = argparse.ArgumentParser(description="UniDet evaluation.")
    parser.add_argument(
        "--outpath", 
        type=str,
        default="../examples/",
        help="Path to output score",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64, 
        help="Batch size",
    )
    args = parser.parse_args()
    return args

def determine_position(locality, box1, box2, iou_threshold=0.1,distance_threshold=150, depth_map=None, iou_threshold_3d=0.5):
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
    
    # 3d spatial relation
    elif locality == 'in front of':
        # use depth map to determine
        depth_A = depth_map[int(box1["x_min"]): int(box1["x_max"]), int(box1["y_min"]):int(box1["y_max"])]
        depth_B = depth_map[int(box2["x_min"]): int(box2["x_max"]), int(box2["y_min"]):int(box2["y_max"])]
        mean_depth_A = depth_A.mean()
        mean_depth_B = depth_B.mean()
        
        depth_diff = mean_depth_A - mean_depth_B
        # get the overlap of bbox1 and bbox2
        if iou > iou_threshold_3d:

            if (depth_diff > 0): # TODO set the threshold
                score=1
        
        elif iou < iou_threshold_3d and iou > 0:
            if (depth_diff > 0):
                score=iou/iou_threshold_3d
            else:
                score=0
        else:
            score = 0
        
    
    elif locality == 'behind' or locality == 'hidden':
        # use depth map to determine
        depth_A = depth_map[int(box1["x_min"]): int(box1["x_max"]), int(box1["y_min"]):int(box1["y_max"])]
        depth_B = depth_map[int(box2["x_min"]): int(box2["x_max"]), int(box2["y_min"]):int(box2["y_max"])]
        mean_depth_A = depth_A.mean()
        mean_depth_B = depth_B.mean()
        
        depth_diff = mean_depth_A - mean_depth_B
        # get the overlap of bbox1 and bbox2
        if iou > iou_threshold_3d:

            if (depth_diff < 0): # TODO set the threshold
                score=1
        
        elif iou < iou_threshold_3d and iou > 0:
            if (depth_diff < 0):
                score=iou/iou_threshold_3d
            else:
                score=0
        else:
            score = 0
     
        
    else:
        score=0
    return score




def main():
    args = parse_args()

    #  get depth map
    if not os.path.exists(f'{args.outpath}/labels/depth'):
        model, transform = load_expert_model(task='depth')
        accelerator = Accelerator(mixed_precision='fp16')

        outpath = args.outpath
        data_path= outpath
        save_path= f'{outpath}/labels'
        save_path = os.path.join(save_path, 'depth')

        batch_size = args.batch_size
        dataset = Dataset_depth(data_path, transform)
        data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
        )

        model, data_loader = accelerator.prepare(model, data_loader)

        with torch.no_grad():
            for i, (test_data, img_path, img_size) in enumerate(tqdm(data_loader)):
                test_pred = model(test_data)

                for k in range(len(test_pred)):
                    img_path_split = img_path[k].split('/')
                    ps = img_path[k].split('.')[-1]
                    im_save_path = os.path.join(save_path, img_path_split[-3], img_path_split[-2])
                    os.makedirs(im_save_path, exist_ok=True)

                    im_size = img_size[0][k].item(), img_size[1][k].item()
                    depth = test_pred[k]
                    depth = (depth - depth.min()) / (depth.max() - depth.min())
                    depth = torch.nn.functional.interpolate(depth.unsqueeze(0).unsqueeze(1), size=(im_size[1], im_size[0]),
                                                            mode='bilinear', align_corners=True)
                    depth_im = Image.fromarray(255 * depth[0, 0].detach().cpu().numpy()).convert('L')
                    depth_im.save(os.path.join(im_save_path, img_path_split[-1].replace(f'.{ps}', '.png')))
                    
        print('depth map saved in {}'.format(im_save_path))
    
    # get obj detection score
    
    model, transform = load_expert_model(task='obj_detection')
    accelerator = Accelerator(mixed_precision='fp16')

    outpath = args.outpath
    data_path= outpath
    save_path= f'{outpath}/labels'

    depth_path = os.path.join(save_path, 'depth', data_path.split('/')[-1])
    batch_size = args.batch_size
    dataset = Dataset(data_path, depth_path, transform)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model, data_loader = accelerator.prepare(model, data_loader)


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
    with torch.no_grad():
        result = []
        map_result = []
        for _, test_data in enumerate(tqdm(data_loader)):
            test_pred = model(test_data)
            for k in range(len(test_pred)):
                
                instance_boxes = test_pred[k]['instances'].get_fields()['pred_boxes'].tensor  
                instance_id = test_pred[k]['instances'].get_fields()['pred_classes']
                depth = test_data[k]['depth']

                # get score
                instance_score = test_pred[k]['instances'].get_fields()['scores']

                obj_bounding_box, obj_labels_dict = get_mask_labels(depth, instance_boxes, instance_id)

                obj = []  
                for i in range(len(obj_bounding_box)):
                    obj_name = obj_label_map[obj_labels_dict[i]]  
                    obj.append(obj_name)


                img_path_split = test_data[k]['image_path'].split('/')
                prompt = img_path_split[-1].split('_')[0] # get prompt from file names
                
                vocab_spatial_3d = ["in front of", "behind", "hidden"] 

                locality = None

                for word in vocab_spatial_3d:
                    if word in prompt:
                        locality = word
                        break

                nlp = spacy.load("en_core_web_sm")
                doc = nlp(prompt)
                obj1= [token.text for token in doc if token.pos_=='NOUN'][0]
                obj2= [token.text for token in doc if token.pos_=='NOUN'][-1]

                person = ['girl','boy','man','woman']
                if obj1 in person:
                    obj1 = "person"
                if obj2 in person:
                    obj2 = "person"
                # transform obj list to str
                obj_str = " ".join(obj)
                obj1_pos = None
                obj2_pos = None
                if obj1 in obj_str and obj2 in obj_str:
                    # get obj_pos
                    for i in range(len(obj)):
                        if obj1 in obj[i]:
                            obj1_pos = i
                        if obj2 in obj[i]:
                            obj2_pos = i
                        if (obj1_pos is not None) and (obj2_pos is not None):
                            break
                        
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
                    score += determine_position(locality, box1, box2, depth_map=depth) / 2
                elif obj1 in obj_str:
                    # get obj_pos
                    for i in range(len(obj)):
                        if obj1 in obj[i]:
                            obj1_pos = i
                            break
                    # obj1_pos = obj.index(obj1)  
                    score = 0.25 * instance_score[obj1_pos].item()
                elif obj2 in obj_str:
                    # get obj_pos
                    for i in range(len(obj)):
                        if obj2 in obj[i]:
                            obj2_pos = i
                            break
                    # obj2_pos = obj.index(obj2)
                    score = 0.25 * instance_score[obj2_pos].item()
                else:
                    score = 0


                image_dict = {}
                image_dict['question_id']=int(img_path_split[-1].split('_')[-1].split('.')[0])
                image_dict['answer'] = score
                result.append(image_dict)
                

        im_save_path = os.path.join(save_path, 'annotation_obj_detection_3d')
        os.makedirs(im_save_path, exist_ok=True)

        with open(os.path.join(im_save_path, 'vqa_result.json'), 'w') as f:
            json.dump(result, f)

        
        
        
        # get avg score
        score_list = []
        for i in range(len(result)):
            score_list.append(result[i]['answer'])
        with open(os.path.join(im_save_path, 'avg_result.txt'), 'w') as f:
            f.write('avg score is {}'.format(np.mean(score_list)))
        print('avg score is {}'.format(np.mean(score_list)))
        
        print('result saved in {}'.format(im_save_path))

if __name__ == '__main__':
    main()




