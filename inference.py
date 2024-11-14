import time
import torch
from omegaconf import OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, LMSDiscreteScheduler
import logger
from my_model import unet_2d_condition
import json
from PIL import Image
from utils import compute_ca_loss, Pharse2idx, draw_box, setup_logger
import hydra
import os
from tqdm import tqdm
from utils import load_text_inversion
from conf.config import RunConfig
import numpy as np
import pandas as pd
import math

import torchvision.utils
import torchvision.transforms.functional as tf

def make_Samples():
    prompts = ["A hello kitty toy is playing with a purple ball.", #0
               "A bus and a bench" #1
               ]    

    bbox = [
        [[51,102,256,410],[384,307,486,410]],#0
        [[2,121,251,460], [274,345,503,496]]#1 
            ]
    
    phrases = [["hello kitty","ball"],
               ["bus", "bench"]#1
               ]

    data_dict = {
    i: {
        "prompt": prompts[i],
        "bbox": bbox[i],
        "phrases": phrases[i]
    }
    for i in range(len(prompts))
    }
    return data_dict

def make_QBench():

    prompts = ["A bus", #0
               "A bus and a bench", #1
               "A bus next to a bench and a bird", #2
               "A bus next to a bench with a bird and a pizza", #3
               "A green bus", #4
               "A green bus and a red bench", #5
               "A green bus next to a red bench and a pink bird", #6
               "A green bus next to a red bench with a pink bird and a yellow pizza", #7
               "A bus on the left of a bench", #8
               "A bus on the left of a bench and a bird", #9
               "A bus and a pizza on the left of a bench and a bird", #10
               "A bus and a pizza on the left of a bench and below a bird", #11
               ]

    ids = []

    for i in range(len(prompts)):
        ids.append(str(i).zfill(3))
    

    bboxes = [[[2,121,251,460]],#0
            [[2,121,251,460], [274,345,503,496]],#1
            [[2,121,251,460], [274,345,503,496],[344,32,500,187]],#2
            [[2,121,251,460], [274,345,503,496],[344,32,500,187],[58,327,187,403]],#3
            [[2,121,251,460]],#4
            [[2,121,251,460], [274,345,503,496]],#5
            [[2,121,251,460], [274,345,503,496],[344,32,500,187]],#6
            [[2,121,251,460], [274,345,503,496],[344,32,500,187],[58,327,187,403]],#7
            [[2,121,251,460],[274,345,503,496]],#8
            [[2,121,251,460],[274,345,503,496],[344,32,500,187]],#9
            [[2,121,251,460], [58,327,187,403], [274,345,503,496],[344,32,500,187]],#10
            [[2,121,251,460], [58,327,187,403], [274,345,503,496],[344,32,500,187]],#11
            ]

    phrases = [["bus"],#0
               ["bus", "bench"],#1
               ["bus", "bench", "bird"],#2
               ["bus","bench","bird","pizza"],#3
               ["bus"],#4
               ["bus", "bench"],#5
               ["bus", "bench", "bird"],#6
               ["bus","bench","bird","pizza"],#7
               ["bus","bench"],#8
               ["bus","bench","bird"],#9
               ["bus","pizza","bench","bird"],#11
               ["bus","pizza","bench","bird"]#12
               ]

    data_dict = {
    i: {
        "id": ids[i],
        "prompt": prompts[i],
        "bboxes": bboxes[i],
        "phrases": phrases[i]
    }
    for i in range(len(prompts))
    }
    return data_dict

def readPromptsCSV(path):
    df = pd.read_csv(path, dtype={'id': str})
    conversion_dict={}
    for i in range(0,len(df)):
        conversion_dict[df.at[i,'id']] = {
            'prompt': df.at[i,'prompt'],
            'obj1': df.at[i,'obj1'],
            'bbox1':df.at[i,'bbox1'],
            'obj2': df.at[i,'obj2'],
            'bbox2':df.at[i,'bbox2'],
            'obj3': df.at[i,'obj3'],
            'bbox3':df.at[i,'bbox3'],
            'obj4': df.at[i,'obj4'],
            'bbox4':df.at[i,'bbox4'],
        }
    
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
def main(config:RunConfig):

    # build and load model
    with open(config.unet_config) as f:
        unet_config = json.load(f)
    unet = unet_2d_condition.UNet2DConditionModel(**unet_config).from_pretrained(config.model_path, subfolder="unet")
    tokenizer = CLIPTokenizer.from_pretrained(config.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config.model_path, subfolder="vae")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet.to(device)
    text_encoder.to(device)
    vae.to(device)

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
        image = inference(device, unet, vae, tokenizer, text_encoder, config.prompt, converted_bboxes, config.phrases, g, config)[0]

        
        #end stopwatch
        end = time.time()
        #save to logger
        l.log_time_run(start,end)


        #image.save(prompt_output_path / f'{seed}.png')
        image.save(output_path +"/"+ str(seed) + ".jpg")
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
        tf.to_pil_image(image).save(output_path+str(seed)+"_bboxes.png")

    # save a grid of results across all seeds without bboxes
    tf.to_pil_image(torchvision.utils.make_grid(tensor=gen_images,nrow=4,padding=0)).save(str(config.output_path) +"/"+ config.prompt + ".png")
 
    # save a grid of results across all seeds with bboxes
    tf.to_pil_image(torchvision.utils.make_grid(tensor=gen_bboxes_images,nrow=4,padding=0)).save(str(config.output_path) +"/"+ config.prompt + "_bboxes.png")
 
if __name__ == "__main__":
    height = 512
    width = 512
    seeds = range(1,17)

    #bench=make_QBench()
    bench=readPromptsCSV(os.path.join("prompts","prompt_collection_bboxes.csv"))

    model_name="PromptCollection-SD_CAG"
    
    if (not os.path.isdir("./results/"+model_name)):
            os.makedirs("./results/"+model_name)
    
    #intialize logger
    l=logger.Logger("./results/"+model_name+"/")
    
    # ids to iterate the dict
    ids = []
    for i in range(0,len(bench)):
        ids.append(str(i).zfill(3))
    
    for id in ids:
        bboxes=[]
        phrases=[]
        
        if not (isinstance(bench[id]['obj1'], (int,float)) and math.isnan(bench[id]['obj1'])):
            phrases.append(bench[id]['obj1'])
            bboxes.append([int(x) for x in bench[id]['bbox1'].split(',')])
        if not (isinstance(bench[id]['obj2'], (int,float)) and math.isnan(bench[id]['obj2'])):
            phrases.append(bench[id]['obj2'])
            bboxes.append([int(x) for x in bench[id]['bbox2'].split(',')])
        if not (isinstance(bench[id]['obj3'], (int,float)) and math.isnan(bench[id]['obj3'])):
            phrases.append(bench[id]['obj3'])
            bboxes.append([int(x) for x in bench[id]['bbox3'].split(',')])
        if not (isinstance(bench[id]['obj4'], (int,float)) and math.isnan(bench[id]['obj4'])):
            phrases.append(bench[id]['obj4'])
            bboxes.append([int(x) for x in bench[id]['bbox4'].split(',')])
        
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
        )) 
    #log gpu stats
    l.log_gpu_memory_instance()
    #save to csv_file
    l.save_log_to_csv(model_name)