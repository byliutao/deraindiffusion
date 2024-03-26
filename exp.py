import os
import torch
from dataclasses import dataclass
import wplus_utils
from diffusers import StableDiffusionPipeline, DDIMScheduler
import ptp_utils
import seq_aligner

@dataclass
class TestConfig:
    model_path = "/data/model/stable-diffusion-v1-5"
    image_path = "/home/liutao/workspace/deraindiffusion/images/test_image/rain/a bird standing on a fence in the rain.jpg"
    seed = 2024
    prompt = "a bird standing on a fence in the rain"
    negative_prompt = "rain"
    base_save_path = "/home/liutao/workspace/deraindiffusion/images/test_result"
    tao = 1.0
    regular = 1e-7
    self_replace_steps = 0.0
    cross_replace_steps = 0.0
    verbose = True
    device = "cuda"
    num_ddim_steps= 50
    guidance_scale = 7.5
    max_num_words = 77
    learning_rate = 1e1
    num_inner_steps = 10
    yaml_file_path = "/home/liutao/workspace/deraindiffusion/text_images_info.yaml"
    save_path = ""
args = TestConfig()

if __name__ == "__main__":
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

    # W optimize
    matrix_inversion = wplus_utils.MatrixInversion(ldm_stable,inner_steps_num=None,lambda_norm=args.regular,use_freq=False,use_attn_loss=False) #derain 1e-7, 0 
    (image_gt, image_enc), x_t, uncond_embeddings, w_matrices = matrix_inversion.invert(args.image_path, args.prompt, offsets=(0,0,0,0), num_inner_steps=args.num_inner_steps, verbose=args.verbose, learning_rate=args.learning_rate)

    prompts = [args.prompt]
    # image reconstruct (use empty negetive prompt)
    controller = wplus_utils.AttentionStore()
    image_inv, x_t = wplus_utils.run_and_display(ldm_stable, prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=None, optimize_matrices=w_matrices, verbose=False)
    wplus_utils.show_cross_attention(controller,16,["up","mid","down"],prompts,ldm_stable,0)
    # image remove (use negative prompt)
    controller = wplus_utils.WplusAttentionStore(cross_replace_steps=args.cross_replace_steps,self_replace_steps=args.self_replace_steps)
    image_remove, x_t = wplus_utils.run_and_display(ldm_stable, prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=None, optimize_matrices=w_matrices, negative_prompt=args.negative_prompt, verbose=False, tao=args.tao)
    wplus_utils.show_cross_attention(controller,16,["up","mid","down"],prompts,ldm_stable,0,negative_prompt=args.negative_prompt)
    result = ptp_utils.get_view_images([image_gt, image_inv[0], image_remove[0]],verbose=args.verbose)
    if args.save_path is not None:
        result.save(os.path.join(args.save_path,args.prompt+"_remove_"+args.negative_prompt+".jpg"))