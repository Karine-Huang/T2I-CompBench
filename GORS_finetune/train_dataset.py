import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
import os
import json


class T2I_CompBench_Dataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        tokenizer,
        size=512,
        center_crop=False,
        color_jitter=False,
        h_flip=False,
        resize=False,
        reward_root=None,
        dataset_root=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.resize = resize

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")


        #read image
        instance_images_path = os.listdir(instance_data_root)
        instance_prompt = os.listdir(instance_data_root)    #read prompt from file name
        instance_prompt.sort(key=lambda x: int(x.split("_")[1].split('.')[0]))  #sort
        with open(f"{reward_root}/vqa_result.json","r") as file:
            r = json.load(file)
        reward = torch.zeros((len(r), 1))
        reward_final = [] 
        instance_prompt_final=[]
        instance_prompt_index=[]
        # read test set
        with open(dataset_root,"r") as file:
            data = file.read().splitlines()
            data = [d.strip("\n").split("\t")[0] for d in data]

        for i in range(len(r)):
            if instance_prompt[i].split('_')[0] in data:
                reward[i] = float(r[i]["answer"])
                # for fixed template and natural prompts, if satisfy the reward, add to training set
                index = data.index(instance_prompt[i].split('_')[0])
                if index<560:
                    if (reward[i] >0.92):  #threshold # 0.92 for color;  0.85 for shape; 0.9 for texture; 0.8 for spatial 0.75 for non-spatial; 0.4 for complex 
                        reward_final.append(reward[i])
                        instance_prompt_final.append(instance_prompt[i].split('_')[0])
                        instance_prompt_index.append(instance_prompt[i].split('_')[1].split('.')[0])
                else:
                    if reward[i] >0.7:  #threshold # 0.7 for color;  0.6 for shape; 0.65 for texture; 0.8 for spatial 0.75 for non-spatial; 0.4 for complex
                        reward_final.append(reward[i])
                        instance_prompt_final.append(instance_prompt[i].split('_')[0])
                        instance_prompt_index.append(instance_prompt[i].split('_')[1].split('.')[0])

            else:
                continue
        self.reward = reward_final
        self.instance_images_path = instance_images_path
        self.instance_prompt = instance_prompt_final
        self.instance_prompt_index = instance_prompt_index

        self.num_instance_images = len(self.instance_prompt)
        self._length = self.num_instance_images


        img_transforms = []

        if resize:
            img_transforms.append(
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                )
            )
        if center_crop:
            img_transforms.append(transforms.CenterCrop(size))
        if color_jitter:
            img_transforms.append(transforms.ColorJitter(0.2, 0.1))
        if h_flip:
            img_transforms.append(transforms.RandomHorizontalFlip())

        self.image_transforms = transforms.Compose(
            [*img_transforms, transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(
            os.path.join(self.instance_data_root, f'{self.instance_prompt[index]}_{self.instance_prompt_index[index]}.png')
        )
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt[index],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        example["reward"] = self.reward[index]

        return example

tokenizer = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-2-base", subfolder="tokenizer", revision=None
    )

def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    reward = [example["reward"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = tokenizer.pad(
        {"input_ids": input_ids},
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "reward":reward
    }
    return batch






