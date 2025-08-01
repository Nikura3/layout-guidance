from logging import config
import time
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, LMSDiscreteScheduler
import logger
from my_model import unet_2d_condition
import json
from PIL import Image
from utils import compute_ca_loss, Pharse2idx
import os
from tqdm import tqdm
from conf.config import RunConfig
import numpy as np
import pandas as pd
import math
from pathlib import Path

import torchvision.utils
import torchvision.transforms.functional as tf



def readPromptsCSV(path):
    df = pd.read_csv(path, dtype={'id': str})
    conversion_dict = {}
    for i in range(len(df)):
        entry = {'prompt': df.at[i, 'prompt']}
        # Dynamically find all obj/bbox columns and keep original naming
        for col in df.columns:
            if col.startswith('obj'):
                idx = col[3:]
                bbox_col = f'bbox{idx}'
                obj_val = df.at[i, col]
                bbox_val = df.at[i, bbox_col] if bbox_col in df.columns else None
                # Always include obj and bbox, even if NaN for retro compatibility
                entry[col] = obj_val
                if bbox_col in df.columns:
                    entry[bbox_col] = bbox_val
        conversion_dict[df.at[i, 'id']] = entry
    return conversion_dict

def inference(device, unet, vae, tokenizer, text_encoder, prompt, bboxes, phrases, generator, config:RunConfig):

    #Convert a list of string into a unique string separated by ';'
    phrases = "; ".join(phrases)
    object_positions = Pharse2idx(prompt, phrases)

    # Encode Classifier Embeddings
    uncond_input = tokenizer(
        [""] * config.batch_size, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    # Encode Prompt
    input_ids = tokenizer(
            [prompt] * config.batch_size,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )

    cond_embeddings = text_encoder(input_ids.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
    #generator = torch.manual_seed(config.rand_seed)  # Seed generator to create the initial latent noise

    noise_scheduler = LMSDiscreteScheduler(beta_start=config.beta_start, beta_end=config.beta_end,
                                           beta_schedule=config.beta_schedule, num_train_timesteps=config.num_train_timesteps)

    latents = torch.randn(#random noise initiliazed with a manual seed
        (config.batch_size, 4, 64, 64),
        generator=generator,
    ).to(device)

    noise_scheduler.set_timesteps(config.timesteps)

    latents = latents * noise_scheduler.init_noise_sigma

    loss = torch.tensor(10000)

    for index, t in enumerate(tqdm(noise_scheduler.timesteps)):
        iteration = 0

        while loss.item() / config.loss_scale > config.loss_threshold and iteration < config.max_iter and index < config.max_index_step:
            latents = latents.requires_grad_(True)
            latent_model_input = latents
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                unet(latent_model_input, t, encoder_hidden_states=cond_embeddings)

            # update latents with guidance
            loss = compute_ca_loss(attn_map_integrated_mid, attn_map_integrated_up, bboxes=bboxes,
                                   object_positions=object_positions) * config.loss_scale

            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]

            latents = latents - grad_cond * noise_scheduler.sigmas[index] ** 2
            iteration += 1
            torch.cuda.empty_cache()

        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                unet(latent_model_input, t, encoder_hidden_states=text_embeddings)

            noise_pred = noise_pred.sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            torch.cuda.empty_cache()

    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images


#@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def main(config: RunConfig, unet, vae, tokenizer, text_encoder, l, device):

    gen_images = []
    gen_bboxes_images=[]

    for seed in config.seeds:
        print(f"Current seed is : {seed}")

        #start stopwatch
        start=time.time()

        #setup the noise generator on cpu
        g = torch.Generator('cpu').manual_seed(seed)
        
        #adapted for inference method (normalized and different list structure)
        converted_bboxes=[]
        for b in config.bboxes:
            normalized_bbox = [round(x / 512,2) for x in b]
            converted_bboxes.append([normalized_bbox])

        # Inference
        # Clean prompt for Pharse2idx compatibility (no commas)
        clean_prompt = config.prompt.replace(",", "")

        # Also clean phrases (remove commas from individual words)
        clean_phrases = [p.replace(",", "") for p in config.phrases]

        image = inference(device, unet, vae, tokenizer, text_encoder, clean_prompt, converted_bboxes, clean_phrases, g, config)[0]

        
        #end stopwatch
        end = time.time()
        #save to logger
        l.log_time_run(start,end)


        #image.save(prompt_output_path / f'{seed}.png')
        image.save(config.output_path +"/"+ str(seed) + ".jpg")
        #list of tensors
        gen_images.append(tf.pil_to_tensor(image))
        
        #draw the bounding boxes
        image=torchvision.utils.draw_bounding_boxes(tf.pil_to_tensor(image),
                                                    torch.Tensor(config.bboxes),
                                                    labels=config.phrases,
                                                    colors=['green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green'],
                                                    width=4,
                                                    font="font.ttf",
                                                    font_size=20)
        #list of tensors
        gen_bboxes_images.append(image)
        tf.to_pil_image(image).save(config.output_path+str(seed)+"_bboxes.png")

    # save a grid of results across all seeds without bboxes
    tf.to_pil_image(torchvision.utils.make_grid(tensor=gen_images,nrow=4,padding=0)).save(str(config.output_path) +"/"+ config.prompt + ".png")
 
    # save a grid of results across all seeds with bboxes
    tf.to_pil_image(torchvision.utils.make_grid(tensor=gen_bboxes_images,nrow=4,padding=0)).save(str(config.output_path) +"/"+ config.prompt + "_bboxes.png")
 


if __name__ == "__main__":
    height = 512
    width = 512
    seeds = range(1,17)

    #bench=make_QBench()
    bench=readPromptsCSV(os.path.join("prompts","fullNewDataset.csv"))

    model_name="fullNewDataset-SD_CAG"
    
    if (not os.path.isdir("./results/"+model_name)):
            os.makedirs("./results/"+model_name)
    
    #intialize logger
    l=logger.Logger("./results/"+model_name+"/")
    
    model_path = 'CompVis/stable-diffusion-v1-4'
    unet_config = Path('./conf/unet/config.json')

    # build and load model
    with open(unet_config) as f:
        unet_config = json.load(f)
        
    unet = unet_2d_condition.UNet2DConditionModel(**unet_config).from_pretrained(model_path, subfolder="unet")
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(device)}")

    unet.to(device)
    text_encoder.to(device)
    vae.to(device)
    
    # ids to iterate the dict
    ids = []
    for i in range(0,len(bench)):
        ids.append(str(i).zfill(4))
    
    for id in ids:
        bboxes = []
        phrases = []
        # Dynamically find all obj/bbox columns for each id
        for col in bench[id]:
            if col.startswith('obj') and bench[id][col] is not None and not (isinstance(bench[id][col], (int, float)) and math.isnan(bench[id][col])):
                idx = col[3:]  # get the number after 'obj'
                bbox_col = f'bbox{idx}'
                if bbox_col in bench[id] and bench[id][bbox_col] is not None:
                    phrases.append(bench[id][col])
                    bboxes.append([int(x) for x in bench[id][bbox_col].split(',')])

        
        output_path = "./results/"+model_name+"/"+ id +'_'+bench[id]['prompt'] + "/"

        if (not os.path.isdir(output_path)):
            os.makedirs(output_path)
            
        print("Sample number ", id)
        torch.cuda.empty_cache()

        main(RunConfig(
            prompt_id=id,
            prompt=bench[id]['prompt'],
            phrases=phrases,
            seeds=seeds,
            bboxes=bboxes,
            output_path=output_path,
        ), 
        unet, 
        vae, 
        tokenizer, 
        text_encoder, 
        l,
        device
        )
    #log gpu stats
    l.log_gpu_memory_instance()
    #save to csv_file
    l.save_log_to_csv(model_name)
    print("End of generation process for ", model_name)