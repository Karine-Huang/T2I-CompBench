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
