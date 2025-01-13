
# T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-image Generation
Kaiyi Huang<sup>1</sup>, Chengqi Duan<sup>3</sup>, Kaiyue Sun<sup>1</sup>, Enze Xie<sup>2</sup>, Zhenguo Li<sup>2</sup>, and Xihui Liu<sup>1</sup>.

**<sup>1</sup>The University of Hong Kong, <sup>2</sup>Huawei Noah’s Ark Lab, <sup>3</sup>Tsinghua University**

<a href='https://karine-h.github.io/T2I-CompBench-new/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://github.com/Karine-Huang/T2I-CompBench/blob/main/paper/T2I_CompBench%2B%2B.pdf'><img src='https://img.shields.io/badge/T2I--CompBench++-Paper-red'></a> 
<a href='https://arxiv.org/pdf/2307.06350.pdf'><img src='https://img.shields.io/badge/T2I--CompBench-Arxiv-red'></a> 
<a href='https://connecthkuhk-my.sharepoint.com/:f:/g/personal/huangky_connect_hku_hk/Er_BhrcMwGREht6gnKGIErMBx8H8yRXLDfWgWQwKaObQ4w?e=YzT5wG'><img src='https://img.shields.io/badge/Dataset-T2I--CompBench++-blue'></a> 
<a href='https://connecthkuhk-my.sharepoint.com/:u:/g/personal/huangky_connect_hku_hk/EfZUFShvGCdArofWOUHXkTgBjcemz2jN6VPmd8tM8mhjeQ?e=NsdFjK'><img src='https://img.shields.io/badge/Dataset-Human eval images-purple'></a> 

## 🚩 **New Features/Updates**
- ✅ Jan. 01, 2025. 💥 T2I-Compbench++ accepted to TPAMI.
- ✅ Nov. 06, 2024. Include evaluation results on [**FLUX.1**](https://github.com/black-forest-labs/flux).
- ✅ Jun. 27, 2024. Release [**human evaluation**](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/huangky_connect_hku_hk/EXEFBTzE6khPlsx2qPMjF9EBo6VR2scOo0KRY1M9g4CMkg?e=o5YT2k) of image-score pairs.
- ✅ Mar. 14, 2024. Release a more comprehensive version of compositional benchmark T2I-CompBench++.
- ✅ Mar. 05, 2024. 💥 Evaluation metric adopted by 🧨 [**Stable Diffusion 3**](https://arxiv.org/pdf/2403.03206).
- ✅ Dec. 02, 2023. Release the inference code for generating images in metric evaluation.
- ✅ Oct. 20, 2023. 💥 Evaluation metric adopted by 🧨 [**DALL-E 3**](https://cdn.openai.com/papers/dall-e-3.pdf) as the evaluation metric for compositionality.
- ✅ Sep. 30, 2023. 💥 Evaluation metric adopted by 🧨 [**PixArt-α**](https://arxiv.org/pdf/2310.00426.pdf) as the evaluation metric for compositionality.
- ✅ Sep. 22, 2023. 💥 Paper accepted to Neurips 2023.
- ✅ Jul. 9, 2023. Release the dataset, training and evaluation code.



## **Installing the dependencies**

Before running the scripts, make sure to install the library's training dependencies:

**Important**

We recommend using the **latest code** to ensure consistency with the results presented in the paper. To make sure you can successfully run the example scripts, execute the following steps in a new virtual environment.
We use the **diffusers version** as **0.15.0.dev0**
You can either install the development version from PyPI: 
```bash
pip install diffusers==0.15.0.dev0
```
or install from the provided source:
```bash
unzip diffusers.zip
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




## **Evaluation**


For evaluation, the input images files are stored in the directory "examples/samples/", with the following structures:

```
examples/samples/
        ├── a green bench and a blue bowl_000000.png
        ├── a green bench and a blue bowl_000001.png
        └──...

```
The evaluation result directory should include a json file named "vqa_result.json", and the json file should be a dictionary that maps from
`{"question_id", "answer"}`, e.g.,

```
[{"question_id": 0, "answer": "0.6900"},
 {"question_id": 1, "answer": "0.7110"},
 ...]
```
The question_id is the _last 6 digits_ of the image file name, for example, _a green bench and a blue bowl_000000.png_ is with question_id _0_, and the evaluation score _0.6900_.




#### 1. BLIP-VQA for Attribute Binding:
```
export project_dir="BLIPvqa_eval/"
cd $project_dir
out_dir="../examples/"
python BLIP_vqa.py --out_dir=$out_dir
```
or run
```
cd T2I-CompBench
bash BLIPvqa_eval/test.sh
```
The output files are formatted as a json file named "vqa_result.json" in "examples/annotation_blip/" directory.


#### 2. UniDet for 2D/3D-Spatial Relationships and Numeracy evaluation:

download weight and put under repo experts/expert_weights:
```
mkdir -p UniDet_eval/experts/expert_weights
cd UniDet_eval/experts/expert_weights
wget https://huggingface.co/shikunl/prismer/resolve/main/expert_weights/Unified_learned_OCIM_RS200_6x%2B2x.pth
wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt
pip install gdown
gdown https://docs.google.com/uc?id=1C4sgkirmgMumKXXiLOPmCKNTZAc3oVbq
```

for **2D-spatial** evaluation, run:
```
export project_dir=UniDet_eval
cd $project_dir

python 2D_spatial_eval.py
```
To calculate prompts from the **"complex" category**, set the **"--complex" parameter to True**; otherwise, set it to False. The output files are formatted as a json file named "vqa_result.json" in "examples/labels/annotation_obj_detection_2d" directory.

for **numeracy** evaluation, run:
```
export project_dir=UniDet_eval
cd $project_dir

python numeracy_eval.py
```
The output files are formatted as a json file named "vqa_result.json" in "examples/annotation_num" directory.

for **3D spatial** evaluation, run:
```
export project_dir=UniDet_eval
cd $project_dir
python 3D_spatial_eval.py 
```
The output files are formatted as a json file named "vqa_result.json" in "examples/labels/annotation_obj_detection_3d" directory.

#### 3. CLIPScore for Non-Spatial Relationships:
```
outpath="examples/"
python CLIPScore_eval/CLIP_similarity.py --outpath=${outpath}
```
or run
```
cd T2I-CompBench
bash CLIPScore_eval/test.sh
```
To calculate prompts from the **"complex" category**, set the **"--complex" parameter to True**; otherwise, set it to False. 
The output files are formatted as a json file named "vqa_result.json" in "examples/annotation_clip" directory.


#### 4. 3-in-1 for Complex Compositions:
```
export project_dir="3_in_1_eval/"
cd $project_dir
outpath="../examples/"
python "3_in_1.py" --outpath=${outpath}
```
The output files are formatted as a json file named "vqa_result.json" in "examples/annotation_3_in_1" directory.

#### 5. MLLM_eval:
If you want to use ShareGPT4V or GPT-4V as evaluation metrics as in TABLE XIII in [T2I-CompBench++](https://github.com/Karine-Huang/T2I-CompBench/blob/main/paper/T2I_CompBench%2B%2B.pdf), you can evaluate as follows.
Prepare the environment for ShareGPT4V.

ShareGPT4V is based on its repositories, please refer to the link for environment dependencies and weights: 
```
https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V
```

For convenience, you can try the following commands to install ShareGPT4V's environment and download the required weights. 
```
export project_dir=MLLM_eval/ShareGPT4V-CoT_eval/
cd $project_dir
conda create -n share4v python=3.10 -y
conda activate share4v
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
pip install spacy
python -m spacy download en_core_web_sm
mkdir -p Lin-Chen/
cd Lin-Chen/
git lfs install
git clone https://huggingface.co/Lin-Chen/ShareGPT4V-7B_Pretrained_vit-large336-l12
```

##### GPT-4V:

Add your openai api key [(instructions)](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key) at line 13. 

```
export project_dir=MLLM_eval
cd $project_dir
python MLLM_eval/gpt4v_eval.py --category "color" --start 0 --step 10
```

The output files are formatted as a json file named "gpt4v_result\_{start}\_{step}.json" in "examples/gpt4v" directory.

In the paper we test 600 images, setting {start=0, step=10}, and {start=1, step=10} from existing 3000 images each category.

##### ShareGPT4V-CoT:
For ShareGPT4V evaluation, run the following commands:
```
export project_dir=MLLM_eval/ShareGPT4V-CoT_eval/
cd $project_dir
category="color"
output_path="../../examples/"
python Share_eval.py --category ${category} --file-path ${output_path} --cot
```
The output files are formatted as a json file named "vqa_result.json" in "examples/sharegpt4v" directory.




## **Finetuning**
If you want to use GORS method to finetune your diffusion model, please follow the steps to prepare the training code and training images.

The training images are stored in the directory "examples/samples/", with the format the same as the evaluation data.

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
[{"question_id": 0, "answer": "0.6900"},
 {"question_id": 1, "answer": "0.7110"},
 ...]
```

The dataset should be placed in the directory "examples/dataset/".


### Inference
Run the inference.py to visualize the image.
```
export pretrained_model_path="checkpoint/color/lora_weight_e357_s124500.pt.pt"
export prompt="A bathroom with green tile and a red shower curtain"
python inference.py --pretrained_model_path "${pretrained_model_path}" --prompt "${prompt}"
```
**Generate images for metric calculation.** Run the inference_eval.py to generate images in the test set. As stated in the paper, 10 images are generated per prompt for **metric calculation**, and we use the fixed seed across all methods.
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
