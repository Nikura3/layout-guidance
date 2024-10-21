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

import torchvision.utils
import torchvision.transforms.functional as tf

def NormalizeData(data,size=512):
    data=np.divide(data,size)
    return np.round(data,2)

def make_Samples():
    prompts = ["A hello kitty toy is playing with a purple ball." #0
               ]    

    bbox = [
        [[51,102,256,410],[384,307,486,410]] #0
            ]
    """ bbox = [
        [[[0.1, 0.2, 0.5, 0.8]], [[0.75, 0.6, 0.95, 0.8]]]
        ]
 """
    phrases = [["hello kitty","ball"]]

    data_dict = {
    i: {
        "prompt": prompts[i],
        "bbox": bbox[i],
        "phrases": phrases[i]
    }
    for i in range(len(prompts))
    }
    return data_dict

def inference(device, unet, vae, tokenizer, text_encoder, prompt, bboxes, phrases, generator, config:RunConfig):


    """ logger.info("Inference")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Phrases: {phrases}") """

    # Get Object Positions

    #logger.info("Convert Phrases to Object Positions")
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

    """ if config.general.real_image_editing:
        text_encoder, tokenizer = load_text_inversion(text_encoder, tokenizer, config.real_image_editing.placeholder_token, config.real_image_editing.text_inversion_path)
        unet.load_state_dict(torch.load(config.real_image_editing.dreambooth_path)['unet'])
        text_encoder.load_state_dict(torch.load(config.real_image_editing.dreambooth_path)['encoder']) """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet.to(device)
    text_encoder.to(device)
    vae.to(device)

    # ------------------ example input ------------------
    """ examples = {"prompt": "A hello kitty toy is playing with a purple ball.",
                "phrases": "hello kitty; ball",
                "bboxes": [[[0.1, 0.2, 0.5, 0.8]], [[0.75, 0.6, 0.95, 0.8]]],
                'save_path': config.general.save_path
                } """

    """ # ------------------ real image editing example input ------------------
    if config.general.real_image_editing:
        examples = {"prompt": "A {} is standing on grass.".format(config.real_image_editing.placeholder_token),
                    "phrases": "{}".format(config.real_image_editing.placeholder_token),
                    "bboxes": [[[0.4, 0.2, 0.9, 0.9]]],
                    'save_path': config.general.save_path
                    }
    # --------------------------------------------------- """
    
    #logger = setup_logger(save_path, __name__)

    #logger.info(config)
    
    # Save config
    #logger.info("save config to {}".format(os.path.join(save_path, 'config.yaml')))
    #OmegaConf.save(config, os.path.join(save_path, 'config.yaml'))

    #intialize logger
    l=logger.Logger(output_path)

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
        for b in config.bbox:
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
        image=torchvision.utils.draw_bounding_boxes(tf.pil_to_tensor(image),torch.Tensor(config.bbox),labels=config.phrases,colors=['blue', 'red', 'purple', 'orange', 'green', 'yellow', 'black', 'gray', 'white'],width=4)
        #list of tensors
        gen_bboxes_images.append(image)
        tf.to_pil_image(image).save(output_path+str(seed)+"_bboxes.png")

    #log gpu stats
    l.log_gpu_memory_instance()
    #save to csv_file
    l.save_log_to_csv(config.prompt)

    # save a grid of results across all seeds without bboxes
    tf.to_pil_image(torchvision.utils.make_grid(tensor=gen_images,nrow=4,padding=0)).save(str(config.output_path) +"/"+ config.prompt + ".png")
    #joined_image = vis_utils.get_image_grid(gen_images)
    #joined_image.save(str(config.output_path) +"/"+ config.prompt + ".png")

    # save a grid of results across all seeds with bboxes
    tf.to_pil_image(torchvision.utils.make_grid(tensor=gen_bboxes_images,nrow=4,padding=0)).save(str(config.output_path) +"/"+ config.prompt + "_bboxes.png")
    #joined_image = vis_utils.get_image_grid(gen_bboxes_images)
    #joined_image.save(str(config.output_path) +"/"+ config.prompt + "_bboxes.png")


    """ # Save example images
    for index, pil_image in enumerate(pil_images):
        image_path = os.path.join(config.general.save_path, examples['prompt']+'.png')
        logger.info('save example image to {}'.format(image_path))
        draw_box(pil_image, examples['bboxes'], examples['phrases'], image_path)
 """
if __name__ == "__main__":
    height = 512
    width = 512
    seeds = range(1,17)

    bench=make_Samples()

    model_name="Sample_SD_CAG"
    
    for sample_to_generate in range(0,len(bench)):
        output_path = "./results/"+model_name+"/"+ bench[sample_to_generate]['prompt'] + "/"

        if (not os.path.isdir(output_path)):
            os.makedirs(output_path)
        print("Sample number ",sample_to_generate)
        torch.cuda.empty_cache()

        main(RunConfig(
            prompt=bench[sample_to_generate]['prompt'],
            phrases=bench[sample_to_generate]['phrases'],
            seeds=seeds,
            bbox=bench[sample_to_generate]['bbox'],
            output_path=output_path,
        )) 