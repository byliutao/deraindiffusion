import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import os
import argparse

import wplus_utils
import ptp_utils

def load_model(args):
    # load model
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1)
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    ldm_stable = StableDiffusionPipeline.from_pretrained(args.model_path, local_files_only=True, scheduler=scheduler).to(device)
    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")

    torch.Generator(device=device).manual_seed(args.seed)
    wplus_utils.init_global(args, ldm_stable.tokenizer)
    return ldm_stable


def load_model_and_remove(args):
    # load model
    ldm_stable = load_model(args)
    remove_by_negative_prompt(args, ldm_stable)


def remove_by_negative_prompt(args, ldm_stable):

    # W optimize
    matrix_inversion = wplus_utils.MatrixInversion(ldm_stable,inner_steps_num=None,lambda_norm=0e-7,use_freq=False,use_attn_loss=False) #derain 1e-7
    (image_gt, image_enc), x_t, uncond_embeddings, w_matrices = matrix_inversion.invert(args.image_path, args.prompt, offsets=(0,0,0,0), num_inner_steps=args.num_inner_steps, verbose=args.verbose, learning_rate=args.learning_rate)

    prompts = [args.prompt]
    # image remove (use negative prompt)
    controller = wplus_utils.WplusAttentionStore(cross_replace_steps=args.cross_replace_steps,self_replace_steps=args.self_replace_steps)
    image_remove, x_t = wplus_utils.run_and_display(ldm_stable, prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=None, optimize_matrices=w_matrices, negative_prompt=args.negative_prompt, verbose=False, tao=args.tao)
    # image reconstruct (use empty negetive prompt)
    controller = wplus_utils.AttentionStore()
    image_inv, x_t = wplus_utils.run_and_display(ldm_stable, prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=None, optimize_matrices=w_matrices, verbose=False)
    result = ptp_utils.get_view_images([image_gt, image_inv[0], image_remove[0]],verbose=args.verbose)
    if args.save_path is not None:
        result.save(os.path.join(args.save_path,args.prompt+"_remove_"+args.negative_prompt+".jpg"))
    return image_inv[0], image_remove[0]




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data/model/stable-diffusion-v1-4", help="Model name or path")
    parser.add_argument("--image_path", type=str, default="/home/liutao/workspace/deraindiffusion/images/test_image/rain/a bird standing on a fence in the rain.jpg", help="Model name or path")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--prompt", type=str, default="a bird standing on a fence in the rain", help="Prompt text")
    parser.add_argument("--negative_prompt", type=str, default="rain", help="Negative prompt text")
    parser.add_argument("--save_path", type=str, default="/home/liutao/workspace/deraindiffusion/images/test_result")
    parser.add_argument("--tao", type=float, default=1.0, help="define which step use negative prompt, 0.9 means 90% of step use negative prompt")
    parser.add_argument("--self_replace_steps", type=float, default=0.0, help="define which step use self_attn_replace")
    parser.add_argument("--cross_replace_steps", type=float, default=0.0, help="define which step use cross_attn_replace")
    parser.add_argument("--verbose", type=bool, default=False, help="")
    parser.add_argument("--device", type=str, default="cuda", help="")   
    parser.add_argument("--low_resource", type=bool, default=False, help="") 
    parser.add_argument("--num_ddim_steps", type=str, default=50, help="") 
    parser.add_argument("--guidance_scale", type=str, default=7.5, help="") 
    parser.add_argument("--max_num_words", type=str, default=77, help="")
    parser.add_argument("--num_inner_steps", type=int, default=10, help="") 
    parser.add_argument("--learning_rate", type=float, default=1e1, help="")


    args = parser.parse_args() 
    load_model_and_remove(args)
    torch.cuda.empty_cache()

