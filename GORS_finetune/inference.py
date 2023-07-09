import os
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import argparse
from lora_diffusion import patch_pipe

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="A bathroom with green tile and a red shower curtain",
        help="Prompt used for generation",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model_id = "stabilityai/stable-diffusion-2-base"

    # Use the Euler scheduler here instead
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)


    patch_pipe(pipe,args.pretrained_model_path , patch_text=True, patch_unet=True, patch_ti=False,
               unet_target_replace_module=["CrossAttention"],
               text_target_replace_module=["CLIPAttention"])


    pipe = pipe.to("cuda")

    prompt =args.prompt

    # run inference
    generator = torch.Generator(device="cuda").manual_seed(42)
    image = pipe(prompt, num_inference_steps=50, generator=generator).images
    image[0].save(f"{prompt}.png")

if __name__ == "__main__":
    main()