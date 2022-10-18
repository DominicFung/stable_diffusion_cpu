# -- coding: utf-8 --`
import argparse, datetime
from os.path import exists

# engine
from stable_diffusion_engine import StableDiffusionEngine
# scheduler
from diffusers import LMSDiscreteScheduler, PNDMScheduler
# utils
import cv2
import numpy as np

import boto3 

BUCKET_NAME = "TextToShirtAIBucket"

def iso_date_time():
    return datetime.datetime.now().isoformat()

def build(args):
  print("build pipeline start:", iso_date_time())
  scheduler = LMSDiscreteScheduler(
      beta_start=args.beta_start,
      beta_end=args.beta_end,
      beta_schedule=args.beta_schedule,
      tensor_format="np"
  )
  print("enter stable deffusion:", iso_date_time())
  StableDiffusionEngine(
      model = args.model,
      scheduler = scheduler,
      tokenizer = args.tokenizer
  )
  print("build pipeline end:", iso_date_time())

def main(args):
    print("load pipeline start:", iso_date_time())
    if args.seed is not None:
        np.random.seed(args.seed)
    if args.init_image is None:
        scheduler = LMSDiscreteScheduler(
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            tensor_format="np"
        )
    else:
        scheduler = PNDMScheduler(
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            skip_prk_steps = True,
            tensor_format="np"
        )
    engine = StableDiffusionEngine(
        model = args.model,
        scheduler = scheduler,
        tokenizer = args.tokenizer
    )
    image = engine(
        prompt = args.prompt,
        init_image = None if args.init_image is None else cv2.imread(args.init_image),
        mask = None if args.mask is None else cv2.imread(args.mask, 0),
        strength = args.strength,
        num_inference_steps = args.num_inference_steps,
        guidance_scale = args.guidance_scale,
        eta = args.eta
    )
    cv2.imwrite(args.output, image)

    env_vars = { "profile": None }
    if exists("./.env.local"):
      with open("./.env.local") as f:
        for line in f:
          if line.startswith('#') or not line.strip():
              continue
          key, value = line.strip().split('=', 1)
          env_vars[key] = value
      
    client=None
    if env_vars["profile"] != None:
      session = boto3.Session(profile_name=env_vars["profile"])
      client = session.client('s3')
    else:
      client = boto3.client("s3")

    session = boto3.Session()
    credentials = session.get_credentials()

    current_credentials = credentials.get_frozen_credentials()
    print(current_credentials.access_key)
    print(current_credentials.secret_key)
    print(current_credentials.token)
    
    client.upload_file(args.output, args.s3bucket, args.output)
    print("completed pipeline:", iso_date_time(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # pipeline configure
    parser.add_argument("--model", type=str, default="bes-dev/stable-diffusion-v1-4-openvino", help="model name")
    # randomizer params
    parser.add_argument("--seed", type=int, default=None, help="random seed for generating consistent images per prompt")
    # scheduler params
    parser.add_argument("--beta-start", type=float, default=0.00085, help="LMSDiscreteScheduler::beta_start")
    parser.add_argument("--beta-end", type=float, default=0.012, help="LMSDiscreteScheduler::beta_end")
    parser.add_argument("--beta-schedule", type=str, default="scaled_linear", help="LMSDiscreteScheduler::beta_schedule")
    # diffusion params
    parser.add_argument("--num-inference-steps", type=int, default=32, help="num inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="guidance scale")
    parser.add_argument("--eta", type=float, default=0.0, help="eta")
    # tokenizer
    parser.add_argument("--tokenizer", type=str, default="openai/clip-vit-large-patch14", help="tokenizer")
    # prompt
    parser.add_argument("--prompt", type=str, default="Street-art painting of Emilia Clarke in style of Banksy, photorealism", help="prompt")
    # img2img params
    parser.add_argument("--init-image", type=str, default=None, help="path to initial image")
    parser.add_argument("--strength", type=float, default=0.5, help="how strong the initial image should be noised [0.0, 1.0]")
    # inpainting
    parser.add_argument("--mask", type=str, default=None, help="mask of the region to inpaint on the initial image")
    # output name
    parser.add_argument("--output", type=str, default="output.png", help="output image name")

    parser.add_argument("--build", action='store_true', help="Just builds the model to be cached with the image.")
    parser.add_argument("--s3bucket", type=str, default=BUCKET_NAME, help="S3 Bucket Name in us-east-1 to store the output")
    args = parser.parse_args()

    if (args.build):
      build(args)
    else:
      main(args)
