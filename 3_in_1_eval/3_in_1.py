import json
import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outpath",
        type = str,
        default = None,
        required=True,
        help = "read score from this path and output 3-in-1 scores",
    )
    parser.add_argument(
        "--data_path",
        type = str,
        default = "../examples/dataset",
        # required=True,
        help = "read prompts from this path",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    outpath=args.outpath
    data_path=args.data_path

    with open(f'{outpath}/annotation_blip/vqa_result.json', 'r') as f: #read attribute score
        attribute_score=json.load(f)
    with open(f'{outpath}/labels/annotation_obj_detection_2d/vqa_result.json', 'r') as f:
        spatial_score=json.load(f)

    with open(f'{outpath}/annotation_clip/vqa_result.json', 'r') as f: #read action score
        action_score=json.load(f)

    #change json to list
    attribute_score=[float(i['answer']) for i in attribute_score]
    spatial_score=[float(i['answer']) for i in spatial_score]
    action_score=[float(i['answer']) for i in action_score]


    #merge score with weight
    with open(f'{data_path}/complex_val_spatial.txt', 'r') as f:
        spatial=f.readlines()
        spatial=[i.strip('\n').split('.')[0].lower() for i in spatial]
    with open(f'{data_path}/complex_val_action.txt', 'r') as f:
        action=f.readlines()
        action=[i.strip('\n').split('.')[0].lower() for i in action]
    with open(f'{data_path}/complex_val.txt', 'r') as f:
        data=f.readlines()
        data=[i.strip('\n').split('.')[0].lower() for i in data]

    num=10 #number of images for each prompt
    dataset_num=len(data)
    total_score=np.zeros(num*dataset_num)
    spatial_score=np.array(spatial_score)
    action_score=np.array(action_score)
    attribute_score=np.array(attribute_score)


    for i in range(dataset_num):
        if data[i] in spatial:#contain spatial relation and attribute
            total_score[i*num:(i+1)*num]=(spatial_score[i*num:(i+1)*num]+attribute_score[i*num:(i+1)*num])*0.5
        elif data[i] in action:#contain action relation and attribute
            total_score[i*num:(i+1)*num]=(action_score[i*num:(i+1)*num]+attribute_score[i*num:(i+1)*num])*0.5
        else:##contain spatial, action relation and attribute
            total_score[i*num:(i+1)*num]=(attribute_score[i*num:(i+1)*num]+spatial_score[i*num:(i+1)*num]+action_score[i*num:(i+1)*num])/3

    total_score=total_score.tolist()

    result=[]
    for i in range(num*dataset_num):
        result.append({'question_id':i,'answer':total_score[i]})


    os.makedirs(f'{outpath}/annotation_3_in_1', exist_ok=True)
    with open(f'{outpath}/annotation_3_in_1/vqa_result.json', 'w') as f:
        json.dump(result,f)
    #calculate avg
    print("avg score:",sum(total_score)/len(total_score))
    with open(f'{outpath}/annotation_3_in_1/vqa_score.txt', 'w') as f:
        f.write("score avg:"+str(sum(total_score)/len(total_score)))

if __name__ == '__main__':
    main()
