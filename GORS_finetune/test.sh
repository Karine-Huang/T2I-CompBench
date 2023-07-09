export project_dir="GORS_finetune/"
cd $project_dir
export pretrained_model_path="checkpoint/color/lora_weight_e357_s124500.pt"
export prompt="A bathroom with green tile and a red shower curtain"

python inference.py --pretrained_model_path "${pretrained_model_path}" --prompt "${prompt}"
