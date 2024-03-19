from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
from torch.optim.adam import Adam
from PIL import Image


def init_global(args, sd_tokenizer):
    global LOW_RESOURCE, NUM_DDIM_STEPS, GUIDANCE_SCALE, MAX_NUM_WORDS, device, tokenizer
    LOW_RESOURCE = args.low_resource 
    NUM_DDIM_STEPS =  args.num_ddim_steps
    GUIDANCE_SCALE = args.guidance_scale
    MAX_NUM_WORDS = args.max_num_words
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = sd_tokenizer


class LocalBlend:
    
    def get_mask(self, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(self.x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask
    
    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
           
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], x_t, substruct_words=None, start_blend=0.2, th=(.3, .3)):
        self.x_t = x_t
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0 
        self.th=th

       
class EmptyControl:
    
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if LOW_RESOURCE:
            attn = self.forward(attn, is_cross, place_in_unet)
        else:
            h = attn.shape[0]
            attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class SpatialReplace(EmptyControl):
    
    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)
        

class WplusAttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}
    
    @staticmethod
    def get_replace_index():
        return {"down_cross": 0, "mid_cross": 0, "up_cross": 0,
                "down_self": 0,  "mid_self": 0,  "up_self": 0}
    

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # print(self.step_replace)
        # print("cond:",self.cond,"curr_step:",self.cur_step,"curr_layer:",self.cur_att_layer,"is_cross:",is_cross)
        if self.cond is True: # self.cond dicide current branch
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
                self.step_store[key].append(attn)
            return attn
        else:
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
                self.step_store_uncond[key].append(attn)
            if is_cross is False and self.cur_step < self.self_replace_steps * NUM_DDIM_STEPS:
                # print("key:",key,"len:",len(self.step_store[key]),"idx:",self.step_replace[key])
                attn = self.step_store[key][self.step_replace[key]]
                self.step_replace[key] += 1
            elif is_cross is True and self.cur_step < self.cross_replace_steps * NUM_DDIM_STEPS:
                attn = self.step_store[key][self.step_replace[key]]
                self.step_replace[key] += 1
            return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]

        if len(self.attention_store_uncond) == 0:
            self.attention_store_uncond = self.step_store_uncond
        else:
            for key in self.attention_store_uncond:
                for i in range(len(self.attention_store_uncond[key])):
                    self.attention_store_uncond[key][i] += self.step_store_uncond[key][i]
        self.step_store = self.get_empty_store()
        self.step_store_uncond = self.get_empty_store()
        self.step_replace = self.get_replace_index()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def get_average_uncond_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store_uncond[key]] for key in self.attention_store_uncond}
        return average_attention

    def reset(self):
        super(WplusAttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.step_store_uncond = self.get_empty_store()
        self.attention_store = {}
        self.attention_store_uncond = {}

    def __init__(self, cross_replace_steps: Union[float, Tuple[float, float]], self_replace_steps: Union[float, Tuple[float, float]]):
        super(WplusAttentionStore, self).__init__()
        self.step_replace = self.get_replace_index()
        self.step_store = self.get_empty_store()
        self.step_store_uncond = self.get_empty_store()
        self.attention_store = {}
        self.attention_store_uncond = {}
        self.cross_replace_steps = cross_replace_steps
        self.self_replace_steps = self_replace_steps
        self.cond = True
        
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

def aggregate_attention(prompts, attention_store: Union[AttentionStore, WplusAttentionStore], res: int, from_where: List[str], is_cross: bool, select: int, uncond = False):
    out = []
    if uncond is False:
        attention_maps = attention_store.get_average_attention()
    else:
        attention_maps = attention_store.get_average_uncond_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def make_controller(prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float], self_replace_steps: float, x_t, blend_words=None, equilizer_params=None) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words, x_t=x_t)
    if is_replace_controller:
        controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    else:
        controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"])
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller)
    return controller


def show_cross_attention(attention_store: Union[AttentionStore, WplusAttentionStore], res: int, from_where: List[str], prompts, model, select: int = 0, negative_prompt = None):
    tokens = model.tokenizer.encode(prompts[select])
    decoder = model.tokenizer.decode
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.get_view_images(np.stack(images, axis=0))

    if negative_prompt is not None:
        tokens = model.tokenizer.encode(negative_prompt)
        decoder = model.tokenizer.decode
        attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select, uncond = True)
        images = []
        for i in range(len(tokens)):
            image = attention_maps[:, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
            images.append(image)
        ptp_utils.get_view_images(np.stack(images, axis=0))
    

def show_self_attention_comp(prompts, attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.get_view_images(np.concatenate(images, axis=1))

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


class NullInversion:
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.)) #origin: 1e-2
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            #     print("step_j:",j,"loss:",loss_item)
            print("step_i:",i,"loss:",loss_item)
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list
    
    def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings
        
    
    def __init__(self, model):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None


def w_modify_start(w_matrices, t):
    identity_tensor = torch.ones(64).unsqueeze(0).unsqueeze(0).to(device)
    identity_tensor = identity_tensor.expand(1, 4, 64, 64).to(device) 
    for i in range(NUM_DDIM_STEPS):
        if(i < NUM_DDIM_STEPS * t):
            w_matrices[i] =  identity_tensor 
            
def w_modify_end(w_matrices, t):
    identity_tensor = torch.ones(64).unsqueeze(0).unsqueeze(0).to(device)
    identity_tensor = identity_tensor.expand(1, 4, 64, 64).to(device) 
    for i in range(NUM_DDIM_STEPS):
        if(i > NUM_DDIM_STEPS * t):
            w_matrices[i] =  identity_tensor 


def circular_conv(A, B):
    # 获取张量的形状信息
    _, _, N = A.shape
    
    # 创建与 A 相同大小的结果张量
    C = torch.zeros_like(A)

    for n in range(N):
        for k in range(N):
            # 计算循环卷积
            C[0, 0, n] += A[0, 0, k] * B[0, 0, (n - k) % N]
    
    return C

def get_circulant_matrix(v):
    assert v.shape == (1, 1, v.shape[-1]), "Input tensor must have shape [1, 1, N]"
    result = torch.cat([f := v.flip(2), f[..., :-1]], dim=2).unfold(2, v.shape[-1], 1).flip(2)
    return result

def fast_circular_conv1d(a, b):
    _, channel, size = a.shape
    channel_results = []
    for i in range(channel):
        channel_a = a[:, i:i+1, :]
        channel_b = b[:, i:i+1, :]
        circ_mat = get_circulant_matrix(channel_a).squeeze()
        b_mat = channel_b.squeeze().view(size,1)
        channel_conv_result = torch.matmul(circ_mat,b_mat).view(1,1,size)
        channel_results.append(channel_conv_result)
    return torch.cat(channel_results, dim=1)

def inverse_circular_conv1d(a, b):
    _, channel, size = a.shape
    channel_results = []
    for i in range(channel):
        channel_a = a[:, i:i+1, :]
        channel_b = b[:, i:i+1, :]
        circ_mat = get_circulant_matrix(channel_a).squeeze()
        circ_mat_inv = torch.inverse(circ_mat)
        b_mat = channel_b.squeeze().view(size,1)
        channel_conv_result = torch.matmul(circ_mat_inv,b_mat).view(1,1,size)
        channel_results.append(channel_conv_result)
    return torch.cat(channel_results, dim=1)

class MatrixInversion:
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None, matrix=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        if matrix is None:
            noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        else:
            noise_pred = noise_pred_uncond + matrix * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt
        
    @torch.no_grad()
    def get_prompt_embeddings(self, prompt: str):
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        return text_embeddings

    @torch.no_grad()
    def ddim_loop(self, latent):
        #get vae ddim attn and store
        if self.use_attn_loss is True:
            controller = AttentionStore()
            ptp_utils.register_attention_control(self.model, controller)
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
            attn_store = {}
            if self.use_attn_loss is True:
                for key, value in controller.step_store.items():
                    if key != 'mid_cross' and key != 'mid_self':
                        attn_store[key] = [v for v in value if v.shape[1]==16**2]
                self.ddim_inv_attn.append(attn_store)
        if self.use_attn_loss is True:
            ptp_utils.register_attention_control(self.model, None)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)            
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def matrix_optimization(self, latents, num_inner_steps, epsilon, learning_rate, verbose=True):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        w_matrices_list = []
        latent_cur = latents[-1]
        identity_tensor = torch.ones(64).unsqueeze(0).unsqueeze(0).to(self.model.device)  # Shape: (1, 1, 64, 64)
        identity_tensor = identity_tensor.expand(1, 4, 64, 64).to(self.model.device) 
        if verbose:
            bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        w_matrix_norm = GUIDANCE_SCALE * identity_tensor
        lambda_norm =  self.lambda_norm
        
        if self.inner_steps_num is None:
            num_inner_steps_list = np.linspace(num_inner_steps,num_inner_steps,NUM_DDIM_STEPS)
        else:
            num_inner_steps_list = self.inner_steps_num
        num_inner_steps_list = [int(num) for num in num_inner_steps_list]
        
        if self.use_attn_loss is True:
            controller = WplusAttentionStore(0.0,0.0)
            ptp_utils.register_attention_control(self.model, controller)
            
        for i in range(NUM_DDIM_STEPS):
            curr_attn = {}
            #TO DO: get wplus current attn
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                if self.use_attn_loss is True:
                    controller.cond = True
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
                if self.use_attn_loss is True:
                    controller.cond = False
                    for key, value in controller.step_store.items():
                        if key != 'mid_cross' and key != 'mid_self':
                            curr_attn[key] = [v for v in value if v.shape[1]==16**2]
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings) 
            
            ## how to optimize W with attn loss, attn loss didn't influence by W, it only depend on latent_cur, t, and cond_embeddings
            if self.use_attn_loss is True:
                pass
            self.cond_noises.append(noise_pred_cond)
            self.uncond_noises.append(noise_pred_uncond)
            
            if self.use_freq is False:
                w_matrices_cond = GUIDANCE_SCALE * identity_tensor
                w_matrices_cond = w_matrices_cond.clone().detach()
                w_matrices_cond.requires_grad = True
                optimizer = Adam([w_matrices_cond], lr=learning_rate * (1. + 0 * (i / 20.))) ## rain
            else:
                w_matrices_cond = GUIDANCE_SCALE * identity_tensor
                
                ## convert noise_pred_start to freq space, and use (formular in freq) to computer w_freq
                ## convert to frequency domain
                freq_domain_cond = torch.fft.fft(noise_pred_cond.view(1,4,64*64))
                freq_domain_uncond = torch.fft.fft(noise_pred_uncond.view(1,4,64*64))
                freq_domian_w = torch.fft.fft(w_matrices_cond.view(1,4,64*64))
                
                freq_domian_noise = freq_domain_uncond + fast_circular_conv1d((freq_domain_cond-freq_domain_uncond),freq_domian_w)
                freq_domain_w_matrices_cond = inverse_circular_conv1d((freq_domain_cond-freq_domain_uncond),(freq_domian_noise-freq_domain_uncond))
                ##
                
                # freq_domain_w_matrices_cond = torch.fft.fft((GUIDANCE_SCALE * identity_tensor).view(1,4,64*64))
                freq_domain_w_matrices_cond = freq_domain_w_matrices_cond.clone().detach()
                freq_domain_w_matrices_cond.requires_grad = True
                optimizer = Adam([freq_domain_w_matrices_cond], lr=learning_rate * (1. + 0 * (i / 20.))) ## rain

            inner_step = num_inner_steps_list[i]
            prev_loss = 1e100
            for j in range(inner_step):                
                ##1. use w+ mult frequency domain's noise
                ##2. convert mult result to space domain
                ##3. calculate mse_loss      
                if self.use_freq:
                    freq_domain_noise = freq_domain_uncond + fast_circular_conv1d(freq_domain_w_matrices_cond,(freq_domain_cond - freq_domain_uncond)) / (64*64)
                    noise_pred = torch.real(torch.fft.ifft(freq_domain_noise)).view(1,4,64,64)
                    w_matrices_cond = torch.real(torch.fft.ifft(freq_domain_w_matrices_cond)).view(1,4,64,64)
                else:
                    noise_pred = noise_pred_uncond + w_matrices_cond * (noise_pred_cond - noise_pred_uncond)
                    # noise_pred = w_matrices_cond * noise_pred_uncond + w_fix * noise_pred_cond
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur) #t to t-1
                loss = nnf.mse_loss(latents_prev_rec, latent_prev) + lambda_norm * nnf.mse_loss(w_matrices_cond, w_matrix_norm)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                if verbose:
                    bar.update()
                    # print("step_j:",j,"loss:",loss_item)
                # if loss > prev_loss:
                #     break
                prev_loss = loss
            if verbose:
                print("   step_i:",i,"loss:",loss_item)
                for j in range(j + 1, num_inner_steps):
                    bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            w_matrices_list.append(w_matrices_cond)
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context, w_matrices_cond)
        if verbose:
            bar.close()
        return uncond_embeddings_list, w_matrices_list
    
    def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0),num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False, learning_rate = 1e-0):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Guidance Matrix optimization...")
        uncond_embeddings, w_matrices = self.matrix_optimization(ddim_latents, num_inner_steps, early_stop_epsilon, learning_rate, verbose=verbose)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings, w_matrices
        
    
    def __init__(self, model, lambda_norm=1e-5, inner_steps_num=None, use_freq=False, use_attn_loss=False):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None
        self.cond_noises = []
        self.uncond_noises = []
        self.ddim_inv_attn = []
        self.lambda_norm = lambda_norm
        self.inner_steps_num = inner_steps_num
        self.use_freq = use_freq
        self.use_attn_loss = use_attn_loss


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    optimize_matrices=None,
    optimize_matrices_=None,
    negative_prompt=None, # if None, another branch will be empty prompt
    start_time=50,
    return_type='image',
    tao=1.0,
    verbose_bar=True
):
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]

    null_input = model.tokenizer(
        "",
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    null_embeddings = model.text_encoder(null_input.input_ids.to(model.device))[0]

    if negative_prompt is not None: # negative prompt
        uncond_input = model.tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0] 
    elif uncond_embeddings is not None: # null-text optimized embedding
        uncond_embeddings_ = None
    else: # null embedding
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    if verbose_bar:
        bar = tqdm(model.scheduler.timesteps[-start_time:])
    else:
        bar = model.scheduler.timesteps[-start_time:]
    for i, t in enumerate(bar):
        if i < NUM_DDIM_STEPS * (1 - tao): #decide which step use negative prompt embedding
            if not LOW_RESOURCE:
                context = torch.cat([null_embeddings, text_embeddings])
            else:
                context = [null_embeddings, text_embeddings]
        else:
            if uncond_embeddings_ is None:
                if not LOW_RESOURCE:
                    context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
                else:
                    context = [uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings]
            else:
                if not LOW_RESOURCE:
                    context = torch.cat([uncond_embeddings_, text_embeddings])
                else:
                    context = [uncond_embeddings_, text_embeddings]
            
        if (optimize_matrices is None) and (optimize_matrices_ is None): #origin
            latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=LOW_RESOURCE)
        elif (optimize_matrices is not None) and (optimize_matrices_ is None): # W+
            optimize_matrix = optimize_matrices[i].to(model.device)
            latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale,
             optimize_matrix=optimize_matrix, low_resource=LOW_RESOURCE)
        elif (optimize_matrices is not None) and (optimize_matrices_ is not None): # W+ with two matrix(abondon)
            optimize_matrix = optimize_matrices[i].to(model.device)
            optimize_matrix_ = optimize_matrices_[i].to(model.device)
            latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale,
             optimize_matrix=optimize_matrix, optimize_matrix_=optimize_matrix_, low_resource=LOW_RESOURCE)
        
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent



def run_and_display(model, prompts, controller, latent=None, run_baseline=False, generator=None, uncond_embeddings=None, optimize_matrices=None, optimize_matrices_uncond=None, negative_prompt=None, verbose=True, tao=1.0, verbose_bar=True):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = text2image_ldm_stable(model, prompts, controller, latent=latent, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings, optimize_matrices=optimize_matrices, optimize_matrices_=optimize_matrices_uncond, negative_prompt=negative_prompt, tao=tao, verbose_bar=verbose_bar)
    if verbose:
        ptp_utils.get_view_images(images)
    return images, x_t


def getFreq(image):
    grayscale = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

    image = np.array(grayscale)
    f_transform = np.fft.fft2(image)
    
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.uint8)
    mask[center_row - 30:center_row + 30, center_col - 30:center_col + 30] = 1
    
    f_transform_shifted *= mask
    
    low_frequency_component = np.fft.ifft2(np.fft.ifftshift(f_transform_shifted)).real
    high_frequency_component = image - low_frequency_component
    
    return low_frequency_component.astype(int), high_frequency_component.astype(int)

def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)