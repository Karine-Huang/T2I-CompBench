import argparse
import os
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
from PIL import Image
from tqdm import tqdm, trange
import sys
# add parent path to sys.path to import lora_diffusion
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lora_diffusion import patch_pipe

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from_file",
        type=str,
        default="../examples/dataset/color_val.txt",
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=10,
        help="number of iterations to run for each prompt",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1, 
        help="batch size for each prompt",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="you can choose to specify the prompt instead of reading from file",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="../examples" # TODO
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        # default="checkpoint/color/lora_weight_e357_s124500.pt", # TODO
        help="to load the finetuned checkpoint or not",
    )
    args = parser.parse_args()
    return args

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def main():
    opt = parse_args()
    
    model_id = "stabilityai/stable-diffusion-2-base"

    # Use the Euler scheduler here instead
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    if opt.pretrained_model_path is not None:
        patch_pipe(pipe, opt.pretrained_model_path, patch_text=True, patch_unet=True, patch_ti=False,
                unet_target_replace_module=["CrossAttention"],
                text_target_replace_module=["CLIPAttention"])

    pipe = pipe.to("cuda")

    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [opt.batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = [d.strip("\n").split("\t")[0] for d in data] 
            data = [opt.batch_size * [d] for d in data]

    # run inference
    outpath = opt.outdir
    os.makedirs(outpath, exist_ok=True)
    sample_path = os.path.join(outpath, f"samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    with torch.no_grad():
        for prompts in tqdm(data, desc="data"):
            images = []
            generator = torch.Generator(device="cuda").manual_seed(42)
            for n in trange(opt.n_iter, desc="Sampling"):

                image = pipe(prompts, num_inference_steps=30, generator=generator).images
                generator = torch.Generator(device="cuda").manual_seed(42 + n + 1)
                for i in range(len(image)):
                    image[i].save(os.path.join(sample_path, f"{prompts[0]}_{base_count:06}.png"))
                    images.append(image[i])
                    base_count += 1
            grid = image_grid(images, rows=opt.n_iter, cols=opt.batch_size)
            grid.save(os.path.join(outpath, f'{prompts[0]}-grid-{grid_count:05}.png'))
            grid_count += 1

if __name__ == "__main__":
    main()
