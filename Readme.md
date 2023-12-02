
# T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation
Kaiyi Huang<sup>1</sup>, Kaiyue Sun<sup>1</sup>, Enze Xie<sup>2</sup>, Zhenguo Li<sup>2</sup>, and Xihui Liu<sup>1</sup>.

**<sup>1</sup>The University of Hong Kong, <sup>2</sup>Huawei Noah’s Ark Lab**

<a href='https://karine-h.github.io/T2I-CompBench/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/pdf/2307.06350.pdf'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 
<a href='https://connecthkuhk-my.sharepoint.com/:f:/g/personal/huangky_connect_hku_hk/Er_BhrcMwGREht6gnKGIErMBx8H8yRXLDfWgWQwKaObQ4w?e=YzT5wG'><img src='https://img.shields.io/badge/Dataset-T2I--CompBench-blue'></a> 

## 🚩 **New Features/Updates**
- ✅ Dec. 02, 2023. Release the inference code for generating images in metric evaluation.
- ✅ Nov. 03, 2023. 💥 Evaluation metric adopted by 🧨 [**DALL-E 3**](https://cdn.openai.com/papers/dall-e-3.pdf) as the evaluation metric for compositionality.
- ✅ Sep. 30, 2023. 💥 Evaluation metric adopted by 🧨 [**PixArt-α**](https://arxiv.org/pdf/2310.00426.pdf) as the evaluation metric for compositionality.
- ✅ Sep. 22, 2023. 💥 Paper accepted to Neurips 2023.
- ✅ Jul. 9, 2023. Release the dataset, training and evaluation code.
- [ ] Human evaluation of image-score pairs


### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then cd in the example folder  and run
```bash
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```


### Finetuning
1. LoRA finetuning

Use LoRA finetuning method, please refer to the link for downloading "lora_diffusion" directory: 

```
https://github.com/cloneofsimo/lora/tree/master
```
2. Example usage


```
export project_dir=/T2I-CompBench
cd $project_dir

export train_data_dir="examples/samples/"
export output_dir="examples/output/"
export reward_root="examples/reward/"
export dataset_root="examples/dataset/color.txt"
export script=GORS_finetune/train_text_to_image.py

accelerate launch --multi_gpu --mixed_precision=fp16 \
--num_processes=8 --num_machines=1 \
--dynamo_backend=no "${script}" \
--train_data_dir="${train_data_dir}" \
--output_dir="${output_dir}" \
--reward_root="${reward_root}" \
--dataset_root="${dataset_root}"

```
or run
```
cd T2I-CompBench
bash GORS_finetune/train.sh
```




The image directory should be a directory containing the images, e.g.,


```
examples/samples/
        ├── a green bench and a blue bowl_000000.png
        ├── a green bench and a blue bowl_000001.png
        └──...

```
The reward directory should include a json file named "vqa_result.json", and the json file should be a dictionary that maps from
`{"question_id", "answer"}`, e.g.,

```
[{"question_id": 0, "answer": "0.7110"},
 {"question_id": 1, "answer": "0.7110"},
 ...]
```

The dataset should be placed in the directory "examples/dataset/".


### Evaluation
1. Install the requirements

MiniGPT4 is based on the repository, please refer to the link for environment dependencies and weights: 
```
https://github.com/Vision-CAIR/MiniGPT-4
```

2. Example usage

For evaluation, the input images files are stored in the directory "examples/samples/", with the format the same as the training data.

#### BLIP-VQA:
```
export project_dir="BLIPvqa_eval/"
cd $project_dir
out_dir="examples/"
python BLIP_vqa.py --out_dir=$out_dir
```
or run
```
cd T2I-CompBench
bash BLIPvqa_eval/test.sh
```
The output files are formatted as a json file named "vqa_result.json" in "examples/annotation_blip/" directory.

#### UniDet:

download weight and put under repo experts/expert_weights:
```
mkdir -p UniDet_eval/experts/expert_weights
cd UniDet_eval/experts/expert_weights
wget https://huggingface.co/shikunl/prismer/resolve/main/expert_weights/Unified_learned_OCIM_RS200_6x%2B2x.pth
```


```
export project_dir=UniDet_eval
cd $project_dir

python determine_position_for_eval.py
```
To calculate prompts from the "complex" category, set the "--complex" parameter to True; otherwise, set it to False. 
The output files are formatted as a json file named "vqa_result.json" in "examples/labels/annotation_obj_detection" directory.

#### CLIPScore:
```
outpath="examples/"
python CLIPScore_eval/CLIP_similarity.py --outpath=${outpath}
```
or run
```
cd T2I-CompBench
bash CLIPScore_eval/test.sh
```
To calculate prompts from the "complex" category, set the "--complex" parameter to True; otherwise, set it to False. 
The output files are formatted as a json file named "vqa_result.json" in "examples/annotation_clip" directory.


#### 3-in-1:
```
export project_dir="3_in_1_eval/"
cd $project_dir
outpath="examples/"
data_path="examples/dataset/"
python "3_in_1.py" --outpath=${outpath} --data_path=${data_path}
```
The output files are formatted as a json file named "vqa_result.json" in "examples/annotation_3_in_1" directory.

#### MiniGPT4-CoT:
If the category to be evaluated is one of color, shape and texture:
```
export project_dir=Minigpt4_CoT_eval
cd $project_dir
category="color"
img_file="examples/samples/"
output_path="examples/"
python mGPT_cot_attribute.py --category=${category} --img_file=${img_file} --output_path=${output_path} 

```

If the category to be evaluated is one of spatial, non-spatial and complex:
```
export project_dir=MiniGPT4_CoT_eval/
cd $project_dir
category="non-spatial"
img_file="examples/samples/"
output_path="examples"
python mGPT_cot_general.py --category=${category} --img_file=${img_file} --output_path=${output_path} 

```
The output files are formatted as a csv file named "mGPT_cot_output.csv" in output_path.

### Inference
Run the inference.py to visualize the image.
```
export pretrained_model_path="checkpoint/color/lora_weight_e357_s124500.pt.pt"
export prompt="A bathroom with green tile and a red shower curtain"
python inference.py --pretrained_model_path "${pretrained_model_path}" --prompt "${prompt}"
```
**Generate images for metric calculation.** Run the inference_eval.py to generate images in the test set. As stated in the paper, 10 images are generated per prompt for **metric calculation**.
You can specify the test set by changing the "from_file" parameter among {color_val.txt, shape_val.txt, texture_val.txt, spatial_val.txt, non_spatial_val.txt, complex_val.txt}.
```
export from_file="../examples/dataset/color_val.txt"
python inference_eval.py  --from_file "${from_file}"
```

### Citation
If you're using T2I-CompBench in your research or applications, please cite using this BibTeX:
```bibtex
@article{huang2023t2icompbench,
      title={T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation}, 
      author={Kaiyi Huang and Kaiyue Sun and Enze Xie and Zhenguo Li and Xihui Liu},
      journal={arXiv preprint arXiv:2307.06350},
      year={2023},
}
```


  ### License

This project is licensed under the MIT License. See the "License.txt" file for details. 
