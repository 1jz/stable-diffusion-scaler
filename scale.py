import random
import importlib
import logging
import math
import diffusers
from PIL import Image
# import torch
# import time
# from datetime import datetime
global torch

def random_seed():
    """Generates random seed"""
    seed = random.randrange(2**64-1)
    return seed

def create_pipe():
    """create diffuser pipeline"""
    globals()["torch"] = importlib.import_module("torch")
    model_id = os.environ.get("UPSCALER_DIRECTORY", "stabilityai/stable-diffusion-x4-upscaler")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = diffusers.StableDiffusionUpscalePipeline.from_pretrained(model_id).to(device)
    pipe.set_use_memory_efficient_attention_xformers(True)
    return pipe


def upscale(pipe, img, prompt="", negative_prompt="", scale=7, steps=20, noise_level=10, seed=None):
    """Scale an image x4

    Args:
        pipe: Preconfigured diffuser pipeline
        img: PIL Image to be scaled
        prompt: prompt to guide scaling
        negative_prompt: negative guides
        scale: configures prompt strength
        steps: # of inference steps
        noise_level: level of noise added to low-res input, higher values give blurrier results
        seed: seed value for deterministic scaling

    Returns:
        A upscaled PIL Image of the input `img`
    """
    global torch
    if seed is not None:
        torch.manual_seed(seed)
    else:
        seed = random_seed()
        torch.manual_seed(seed)
    upscaled_image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=img, guidance_scale=scale, num_inference_steps=steps, noise_level=noise_level).images[0]
    # upscaled_image.save(f"{int(time.mktime(datetime.now().timetuple()))}-{seed}.png")
    return upscaled_image

def segment_img(img, segment_size=256, seg_type="overlay"):
    """Segment input image into array of images with coordinates
    
    Args:
        img: PIL Image to segment
        segment_size: size of each segment in pixels

    Returns:
        list of tuples containing images and coordinates: (x, y, img)
    """
    w = img.size[0]
    h = img.size[1]
    i = 0
    
    segments = []
    while (h > 0):
        while (w > 0):
            w_offset = w if w >= segment_size or seg_type == "crop" else segment_size
            h_offset = h if h >= segment_size or seg_type == "crop"  else segment_size
            cropped_img = img.crop((
                img.size[0] - w_offset, 
                img.size[1] - h_offset, 
                img.size[0] - w_offset + segment_size, 
                img.size[1] - h_offset + segment_size
            ))
            segments.append((img.size[0] - w_offset, img.size[1] - h_offset, cropped_img))
            i += 1
            w -= segment_size
        w = img.size[0]
        h -= segment_size
    
    return segments

def merge_segments(img, img_segments, segment_size=256):
    """Merge image segments into single image
    
    Args:
        img: original image that was segmented
        img_segments: array of segments
    """
    segments = len(img_segments)
    canvas = Image.new("RGB", (img.size[0]*4, img.size[1]*4))
    #canvas = Image.new("RGB", (math.ceil(img.size[0] / segment_size) * segment_size * 4, math.ceil(img.size[1] / segment_size) * segment_size * 4))
    
    for im in range(segments):
        canvas.paste(img_segments[im][2],(img_segments[im][0]*4, img_segments[im][1]*4))
    return canvas

if __name__ == "__main__":
    import argparse
    import os
    os.environ["DISABLE_TELEMETRY"] = "YES"
    logger = logging.getLogger("upscaler")

    parser = argparse.ArgumentParser(
        prog = "stable-scaler",
        description = "Image upscaler using stable-diffusion-x4-upscaler with image segmentation"
    )
    parser.add_argument("-i", "--input", type=str, required=True, help="input image for scaling")
    parser.add_argument("-o", "--output", type=str, default=None, help="output path for final image")
    parser.add_argument("-t", "--type", type=str, default="overlay", choices=["overlay", "crop"])
    parser.add_argument("-n", "--noise_level", type=int, default=10)
    parser.add_argument("-s", "--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--size", type=int, default=256, choices=[64, 128, 256, 512])
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    seed = args.seed

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.input):
        logger.error(f"{args.input} doesnt exist")
        exit(1)

    if args.size < 64 or args.size > 512:
        logger.error(f"{args.size} is invalid, choose a value between 64-512")
        exit(1)


    if seed is None:
        seed = random_seed()
    
    print(f"seed: {seed}")
    if args.verbose:
        logger.info("loading model...")

    pipe = create_pipe()
    original_img = Image.open(args.input).convert("RGB")
    segments = segment_img(original_img, segment_size=args.size)
    segment_count = len(segments)
    upscaled_segments = []

    if args.verbose:
        logger.info(f"upscaling {len(segments)} segments")
    for i, segment in enumerate(segments):
        print(f"{i+1}/{segment_count}:")
        img = upscale(pipe, segment[2], noise_level=args.noise_level, seed=seed, steps=args.steps)
        upscaled_segments.append((segment[0], segment[1], img))
    
    final_img = merge_segments(original_img, upscaled_segments, args.size)

    if os.path.exists(args.output):
        user_input = input(f"{args.output} already exists, replace? (y/N): ")
        if user_input.lower() != "y":
            if args.verbose:
                logger.error(f"Not replacing, exiting")
            exit(2)
    
    if args.verbose:
        logger.info(f"saving final image to: {args.output}")
    final_img.save(args.output)