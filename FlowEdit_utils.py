from typing import Optional, Union
import torch
from tqdm import tqdm
import numpy as np
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
    scheduler._init_step_index(timestep)

    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample

    return sample

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

def calc_v_sd3(
        pipe, src_tar_latent_model_input,
        src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
        src_guidance_scale, tar_guidance_scale, t):
    timestep = t.expand(src_tar_latent_model_input.shape[0])

    with torch.no_grad():
        noise_pred_src_tar = pipe.transformer(
            hidden_states=src_tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds,
            pooled_projections=src_tar_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        if pipe.do_classifier_free_guidance:
            src_noise_pred_uncond, src_noise_pred_text, tar_noise_pred_uncond, tar_noise_pred_text = noise_pred_src_tar.chunk(4)
            noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
            noise_pred_tar = tar_noise_pred_uncond + tar_guidance_scale * (tar_noise_pred_text - tar_noise_pred_uncond)

    return noise_pred_src, noise_pred_tar

def calc_v_flux(
        pipe, latents, prompt_embeds,
        pooled_prompt_embeds, guidance,
        text_ids, latent_image_ids, t):
    timestep = t.expand(latents.shape[0])

    with torch.no_grad():
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    return noise_pred

@torch.no_grad()
def FlowEditSD3(pipe,
                scheduler,
                x_src,
                src_prompt,
                tar_prompt,
                negative_prompt,
                T_steps: int = 50,
                n_avg: int = 1,
                src_guidance_scale: float = 3.5,
                tar_guidance_scale: float = 13.5,
                n_min: int = 0,
                n_max: int = 15,):

    device = x_src.device

    x_src = x_src.to(torch.float16)
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    timesteps, T_steps = retrieve_timesteps(
        scheduler, T_steps, device, timesteps=None)
    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)
    pipe._guidance_scale = src_guidance_scale

    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    src_tar_prompt_embeds = torch.cat(
        [src_negative_prompt_embeds, src_prompt_embeds,
         tar_negative_prompt_embeds, tar_prompt_embeds], dim=0)
    src_tar_pooled_prompt_embeds = torch.cat(
        [src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds,
         tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)

    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps)):

        torch.cuda.empty_cache()

        if T_steps - i > n_max:
            continue

        t_i = t/1000
        if i+1 < len(timesteps): 
            t_im1 = (timesteps[i+1])/1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)

        if T_steps - i > n_min:

            V_delta_avg = torch.zeros_like(x_src)
            for k in range(n_avg):

                fwd_noise = torch.randn_like(x_src).to(x_src.device)

                zt_src = (1-t_i)*x_src + (t_i)*fwd_noise

                zt_tar = zt_edit + zt_src - x_src

                src_tar_latent_model_input = torch.cat(
                    [zt_src, zt_src, zt_tar, zt_tar]) if pipe.do_classifier_free_guidance else (zt_src, zt_tar)

                Vt_src, Vt_tar = calc_v_sd3(
                    pipe, src_tar_latent_model_input, src_tar_prompt_embeds,
                    src_tar_pooled_prompt_embeds, src_guidance_scale,
                    tar_guidance_scale, t)

                V_delta_avg += (1/n_avg) * (Vt_tar - Vt_src)

            zt_edit = zt_edit.to(torch.float32)

            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg

            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:

            if i == T_steps-n_min:
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            src_tar_latent_model_input = torch.cat(
                [xt_tar, xt_tar, xt_tar, xt_tar]) if pipe.do_classifier_free_guidance else (xt_src, xt_tar)

            _, Vt_tar = calc_v_sd3(
                pipe, src_tar_latent_model_input,
                src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                src_guidance_scale, tar_guidance_scale, t)

            xt_tar = xt_tar.to(torch.float32)

            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)

            prev_sample = prev_sample.to(noise_pred_tar.dtype)

            xt_tar = prev_sample

    torch.cuda.empty_cache()
    gc.collect()
    return zt_edit if n_min == 0 else xt_tar

@torch.no_grad()
def FlowEditFLUX(pipe,
                 scheduler,
                 x_src,
                 src_prompt,
                 tar_prompt,
                 negative_prompt,
                 T_steps: int = 28,
                 n_avg: int = 1,
                 src_guidance_scale: float = 1.5,
                 tar_guidance_scale: float = 5.5,
                 n_min: int = 0,
                 n_max: int = 24,):

    device = x_src.device

    x_src = x_src.to(torch.float16)
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    orig_height, orig_width = x_src.shape[2]*pipe.vae_scale_factor//2, x_src.shape[3]*pipe.vae_scale_factor//2
    num_channels_latents = pipe.transformer.config.in_channels // 4

    pipe.check_inputs(
        prompt=src_prompt,
        prompt_2=None,
        height=orig_height,
        width=orig_width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    )

    x_src, latent_src_image_ids = pipe.prepare_latents(batch_size=x_src.shape[0], num_channels_latents=num_channels_latents, height=orig_height, width=orig_width, dtype=x_src.dtype, device=x_src.device, generator=None, latents=x_src)
    x_src_packed = pipe._pack_latents(
        x_src, x_src.shape[0], num_channels_latents,
        x_src.shape[2], x_src.shape[3])
    latent_tar_image_ids = latent_src_image_ids

    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = x_src_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, T_steps = retrieve_timesteps(
        scheduler,
        T_steps,
        device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
        )

    num_warmup_steps = max(len(timesteps) - T_steps * pipe.scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    (
        src_prompt_embeds,
        src_pooled_prompt_embeds,
        src_text_ids,

    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        device=device,
    )

    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_text_ids,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        device=device,
    )

    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device)
        src_guidance = src_guidance.expand(x_src_packed.shape[0])
        tar_guidance = torch.tensor([tar_guidance_scale], device=device)
        tar_guidance = tar_guidance.expand(x_src_packed.shape[0])
    else:
        src_guidance = None
        tar_guidance = None

    zt_edit = x_src_packed.clone()

    for i, t in tqdm(enumerate(timesteps)):

        torch.cuda.empty_cache()

        if T_steps - i > n_max:
            continue
        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        if i < len(timesteps):
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        else:
            t_im1 = t_i

        if T_steps - i > n_min:

            V_delta_avg = torch.zeros_like(x_src_packed)

            for k in range(n_avg):

                fwd_noise = torch.randn_like(
                    x_src_packed).to(x_src_packed.device)

                zt_src = (1-t_i)*x_src_packed + (t_i)*fwd_noise

                zt_tar = zt_edit + zt_src - x_src_packed

                Vt_src = calc_v_flux(pipe,
                                     latents=zt_src,
                                     prompt_embeds=src_prompt_embeds,
                                     pooled_prompt_embeds=src_pooled_prompt_embeds,
                                     guidance=src_guidance,
                                     text_ids=src_text_ids,
                                     latent_image_ids=latent_src_image_ids,
                                     t=t)

                Vt_tar = calc_v_flux(pipe,
                                     latents=zt_tar,
                                     prompt_embeds=tar_prompt_embeds,
                                     pooled_prompt_embeds=tar_pooled_prompt_embeds,
                                     guidance=tar_guidance,
                                     text_ids=tar_text_ids,
                                     latent_image_ids=latent_tar_image_ids,
                                     t=t)

                V_delta_avg += (1/n_avg)*(Vt_tar - Vt_src)

            zt_edit = zt_edit.to(torch.float32)

            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg

            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:

            if i == T_steps-n_min:
                fwd_noise = torch.randn_like(
                    x_src_packed).to(x_src_packed.device)
                xt_src = scale_noise(
                    scheduler, x_src_packed, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_packed

            Vt_tar = calc_v_flux(pipe,
                                 latents=xt_tar,
                                 prompt_embeds=tar_prompt_embeds,
                                 pooled_prompt_embeds=tar_pooled_prompt_embeds,
                                 guidance=tar_guidance,
                                 text_ids=tar_text_ids,
                                 latent_image_ids=latent_tar_image_ids,
                                 t=t)

            xt_tar = xt_tar.to(torch.float32)

            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)

            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample
    torch.cuda.empty_cache()
    gc.collect()
    out = zt_edit if n_min == 0 else xt_tar
    unpacked_out = pipe._unpack_latents(
        out, orig_height, orig_width, pipe.vae_scale_factor)
    return unpacked_out
