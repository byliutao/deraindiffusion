import json
import os
import torch 
from PIL import Image
from dataclasses import dataclass
from transformers import Blip2Processor, Blip2ForConditionalGeneration

import supp_single_gen
import ptp_utils

@dataclass
class TestConfig:
    model_path = "/data/model/stable-diffusion-v1-5"
    seed = 2024
    base_save_path = "/data/liutao/supp"
    save_path = ""
    tao = 0.8
    self_replace_steps = 0.0
    cross_replace_steps = 0.0
    verbose = True
    device = "cuda"
    low_resource = False
    num_ddim_steps= 50
    guidance_scale = 7.5
    max_num_words = 77
    learning_rate = 1e1
    num_inner_steps = 10
    json_file_path = '/data/dataset/gqa-inpaint/test_instructions.json'
    dataset_path = '/data/dataset/gqa-inpaint'
    blip2_model_path = '/data/model/blip2-opt-2.7b'
    save_sub_dir_list = ["source","inpainted_groudtrue","reconstruct","w+inpainted"]



def get_image_path_by_id(id):
    split_str = id.split("-")
    source_image_id = split_str[0][1:]
    inpainted_image_id = split_str[1][1:]
    source_image_path = os.path.join(config.dataset_path,"images",source_image_id+".jpg")
    inpainted_image_path = os.path.join(config.dataset_path,"images_inpainted",source_image_id,inpainted_image_id+".jpg")
    return source_image_path, inpainted_image_path

def load_blip(path):
    blip2_model = Blip2ForConditionalGeneration.from_pretrained(config.blip2_model_path, torch_dtype=torch.float16).to(config.device)
    blip2_processor = Blip2Processor.from_pretrained(config.blip2_model_path)
    try:
        blip2_model.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")
    return blip2_model, blip2_processor

def image_to_text(image, blip2_model, blip2_processor):
    inputs = blip2_processor(images=image, return_tensors="pt").to(config.device, torch.float32)
    generated_ids = blip2_model.generate(**inputs)
    generated_text = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

def get_negatvie_promt(str):
    words = str.split()
    if words[0] == "remove":
        words.pop(0)  
    result = ' '.join(words)
    return result

if __name__ == "__main__":
    config = TestConfig()
    dir_name = f"self:{config.self_replace_steps}|cross:{config.cross_replace_steps}|tao:{config.tao}_1"
    config.save_path = os.path.join(config.base_save_path,dir_name)
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    for sub_dir in config.save_sub_dir_list:
        sub_dir_path = os.path.join(config.save_path,sub_dir)
        if not os.path.exists(sub_dir_path):
            os.makedirs(sub_dir_path)
    
    
    with open(config.json_file_path, 'r') as file:
        data = json.load(file)
    config.verbose = False
    ldm_stable = supp_single_gen.load_model(config)
    blip2_model, blip2_processor = load_blip(config.blip2_model_path)


    for key, value in data.items():
        print(key,value)
        source_image_path, inpainted_image_path = get_image_path_by_id(key)
        remove_instruction = value
        source_image = Image.open(source_image_path)
        inpainted_image = Image.open(inpainted_image_path)
        config.image_path = source_image_path
        config.negative_prompt = get_negatvie_promt(remove_instruction)
        config.prompt = image_to_text(source_image,blip2_model,blip2_processor)+" and "+config.negative_prompt
        config.seed = config.seed
        reconstruct_image, w_inpainted_image = supp_single_gen.remove_by_negative_prompt(config, ldm_stable)
        reconstruct_image = ptp_utils.get_view_images(reconstruct_image,verbose=False)
        w_inpainted_image = ptp_utils.get_view_images(w_inpainted_image,verbose=False)
        image_list = [source_image, inpainted_image, reconstruct_image, w_inpainted_image]
        save_name_list = [config.prompt,value,config.prompt,value]
        for sub_dir, image, save_name in zip(config.save_sub_dir_list, image_list, save_name_list):
            image_save_path = os.path.join(config.save_path,sub_dir,save_name+".jpg")
            image.save(image_save_path)
        


