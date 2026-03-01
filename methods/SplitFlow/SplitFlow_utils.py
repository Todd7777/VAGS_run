from typing import Optional, Tuple, Union
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Optional

import torch.nn.functional as F
import matplotlib.pyplot as plt


from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps


def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Foward process in flow-matching

    Args:
        sample (`torch.FloatTensor`):
            The input sample.
        timestep (`int`, *optional*):
            The current timestep in the diffusion chain.

    Returns:
        `torch.FloatTensor`:
            A scaled input sample.
    """
    # if scheduler.step_index is None:
    scheduler._init_step_index(timestep)

    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample

    return sample


# for flux
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu



def calc_v_sd3(pipe, src_tar_latent_model_input, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(src_tar_latent_model_input.shape[0])


    with torch.no_grad():
        # # predict the noise for the source prompt
        noise_pred_src_tar = pipe.transformer(
            hidden_states=src_tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds,
            pooled_projections=src_tar_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        # perform guidance source
        # if pipe.do_classifier_free_guidance:
        if True:
            src_noise_pred_uncond, src_noise_pred_text, tar_noise_pred_uncond, tar_noise_pred_text = noise_pred_src_tar.chunk(4)
            noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
            noise_pred_tar = tar_noise_pred_uncond + tar_guidance_scale * (tar_noise_pred_text - tar_noise_pred_uncond)

    return noise_pred_src, noise_pred_tar

def get_prompt_for_t(t, prompt_schedule):
    """
    주어진 timestep t에 대해, 가장 가까운 과거 시점의 프롬프트를 반환
    """
    prompt_schedule_sorted = sorted(prompt_schedule, key=lambda x: x[0])
    # pdb.set_trace()
    for step, prompt in reversed(prompt_schedule_sorted):
        if t >= step:
            return prompt
    return prompt_schedule_sorted[0][1]



@torch.no_grad()
def SplitFlowSD3(pipe, # SD3 Pipeline
                       scheduler,
                       x_src, # Source latent
                       src_prompt: str, # Explicit source prompt
                       prompt_schedule: List[Tuple[int, str]], # Chain-of-Thought schedule for edits
                       negative_prompt: str,
                       T_steps: int = 50,
                       n_avg: int = 1,
                       src_guidance_scale: float = 3.5, # Guidance for the source prompt (Vt_src)
                       edit_guidance_scale: float = 13.5, # Guidance for the scheduled edit prompts (Vt_tar)
                       n_min: int = 0,
                       n_max: int = 15):

    device = x_src.device

    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)
    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

   

    (
        src_pos_embeds, # Positive embeds for the source prompt
        src_neg_embeds, # Negative embeds (used for both src and tar guidance)
        src_pos_pooled,
        src_neg_pooled,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None, prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True, # Need both pos/neg embeds
        device=device,
    )

    # Decomposition Step
    current_tar_pos_embeds_list = []
    current_tar_pos_pooled_list = []
    for prompt_idx , current_edit_prompt in enumerate(prompt_schedule):

        print(current_edit_prompt)
        (current_tar_pos_embeds, _, current_tar_pos_pooled, _) = pipe.encode_prompt(
            prompt=current_edit_prompt, prompt_2=None, prompt_3=None,
            negative_prompt=None, do_classifier_free_guidance=True, device=device,
        )
        current_tar_pos_embeds_list.append(current_tar_pos_embeds)
        current_tar_pos_pooled_list.append(current_tar_pos_pooled)

    zt_edit = x_src.clone() # Z_t^{FE} path starts at source image
    zt_edit_list = [zt_edit.clone() for _ in range(len(prompt_schedule))]
    zt_tar_list = [0]*(len(prompt_schedule))


    for i, t in tqdm(enumerate(timesteps)):

        if T_steps - i > n_max:
            continue

        t_i = t / 1000.0
        if i + 1 < len(timesteps):
            t_im1 = timesteps[i + 1] / 1000.0
        else:
            t_im1 = torch.tensor(0.0, device=device, dtype=t_i.dtype)

        current_step = T_steps-i

        n_middle = 28 
        edit_guidance_scale_cot = 13.5

        if current_step >= (n_middle-1):

            V_delta_avg_list = [torch.zeros_like(x_src) for _ in range(len(prompt_schedule))]

            for prompt_idx , current_edit_prompt in enumerate(prompt_schedule):
                fwd_noise = torch.randn_like(x_src)
                
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise 
 
                zt_tar_list[prompt_idx] = zt_edit_list[prompt_idx] + zt_src - x_src

                latent_model_input = torch.cat([zt_src] * 2 + [zt_tar_list[prompt_idx]] * 2)

                combined_prompt_embeds = torch.cat([
                    src_neg_embeds, src_pos_embeds,
                    src_neg_embeds, current_tar_pos_embeds_list[prompt_idx] # Negative는 공유, Positive는 현재 스텝 따라 결정
                ], dim=0)
                combined_pooled_embeds = torch.cat([
                    src_neg_pooled, src_pos_pooled,
                    src_neg_pooled, current_tar_pos_pooled_list[prompt_idx]
                ], dim=0)

                Vt_src, Vt_tar = calc_v_sd3(pipe, latent_model_input, combined_prompt_embeds, combined_pooled_embeds,
                                            src_guidance_scale, edit_guidance_scale_cot, t)

                V_delta_avg_list[prompt_idx] = (1 / n_avg) * (Vt_tar - Vt_src)

                if current_step == (n_middle-1):
                    continue

                ## Compute independent latent trajectory 
                zt_edit_list[prompt_idx] = zt_edit_list[prompt_idx].to(torch.float32)
                zt_edit_list[prompt_idx] = zt_edit_list[prompt_idx] + (t_im1 - t_i) * V_delta_avg_list[prompt_idx]
                zt_edit_list[prompt_idx] = zt_edit_list[prompt_idx].to(V_delta_avg_list[prompt_idx].dtype)
            
            if current_step == n_middle:
                print(f"\n{'='*20} AGGREGATION  at {n_middle} {'='*20}")
                zt_edit = Aggregation_SD3(zt_edit_list[:-1],zt_edit_list[-1])

                zt_edit_list = [zt_edit.clone()]*len(prompt_schedule)

            elif current_step == (n_middle-1): 
                v_list = torch.stack(V_delta_avg_list[:-1], dim=1)[0]
                N, C, H, W = v_list.shape

                Temperature = 1

                v_norm = F.normalize(v_list, dim=1, eps=1e-8)  
                cs_matrix = torch.einsum('nchw,mchw->nmhw', v_norm, v_norm)
                cs_sum = cs_matrix.sum(dim=1) - 1  
                weights = F.softmax(cs_sum*Temperature, dim=0).unsqueeze(1)  
                v_agg = (weights * v_list).sum(dim=0)  

                zt_edit_list[-1] = zt_edit_list[-1].to(torch.float32) 
                zt_edit_list[-1] = zt_edit_list[-1] + (t_im1 - t_i) * v_agg
                zt_edit_list[-1] = zt_edit_list[-1].to(v_agg.dtype)

                zt_edit = zt_edit_list[-1].clone()
  

        elif (T_steps-i < n_middle):
            V_delta_avg = torch.zeros_like(x_src)
            for k in range(n_avg):
                fwd_noise = torch.randn_like(x_src)
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src

                latent_model_input = torch.cat([zt_src] * 2 + [zt_tar] * 2)

                combined_prompt_embeds = torch.cat([
                    src_neg_embeds, src_pos_embeds,
                    src_neg_embeds, current_tar_pos_embeds #
                ], dim=0)
                combined_pooled_embeds = torch.cat([
                    src_neg_pooled, src_pos_pooled,
                    src_neg_pooled, current_tar_pos_pooled
                ], dim=0)

                Vt_src, Vt_tar = calc_v_sd3(pipe, latent_model_input, combined_prompt_embeds, combined_pooled_embeds,
                                            src_guidance_scale, edit_guidance_scale, t)

                V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src)


            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

    return zt_edit 


def Aggregation_SD3(v_list, v_tgt, alpha=0.):
    N = len(v_list)
    device = v_list[0].device
    B, C, H, W = v_tgt.shape

    v_list_tensor = torch.stack(v_list, dim=1)  
    v_agg_list = []

    for b in range(B):
        v_sub = v_list_tensor[b]  
        v_tgt_b = v_tgt[b]        

        v_tgt_norm = F.normalize(v_tgt_b, dim=0, eps=1e-8)  

        numerator = (v_sub * v_tgt_norm.unsqueeze(0)).sum(dim=1) 

        s = numerator 

        proj = s.unsqueeze(1) * v_tgt_norm.unsqueeze(0)  

        v_agg = proj.mean(0)
        v_agg_list.append(v_agg)
    v_agg = torch.stack(v_agg_list, dim=0)  
    return v_agg
