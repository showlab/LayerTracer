#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Minimum Inference Code for FLUX - Modified for Multiple Prompts

import argparse
import datetime
import math
import os
import random
from typing import Callable, List, Optional
import einops
import numpy as np

import torch
from tqdm import tqdm
from PIL import Image
import accelerate
from transformers import CLIPTextModel
from safetensors.torch import load_file

from library import device_utils
from library.device_utils import init_ipex, get_preferred_device
from networks import oft_flux

init_ipex()

from library.utils import setup_logging, str_to_dtype

setup_logging()
import logging

logger = logging.getLogger(__name__)

import networks.lora_flux as lora_flux
from library import flux_models, flux_utils, sd3_utils, strategy_flux


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: flux_models.Flux,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    vec: torch.Tensor,
    timesteps: list[float],
    guidance: float = 4.0,
    t5_attn_mask: Optional[torch.Tensor] = None,
    neg_txt: Optional[torch.Tensor] = None,
    neg_vec: Optional[torch.Tensor] = None,
    neg_t5_attn_mask: Optional[torch.Tensor] = None,
    cfg_scale: Optional[float] = None,
):
    # this is ignored for schnell
    logger.info(f"guidance: {guidance}, cfg_scale: {cfg_scale}")
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    # prepare classifier free guidance
    if neg_txt is not None and neg_vec is not None:
        b_img_ids = torch.cat([img_ids, img_ids], dim=0)
        b_txt_ids = torch.cat([txt_ids, txt_ids], dim=0)
        b_txt = torch.cat([neg_txt, txt], dim=0)
        b_vec = torch.cat([neg_vec, vec], dim=0)
        if t5_attn_mask is not None and neg_t5_attn_mask is not None:
            b_t5_attn_mask = torch.cat([neg_t5_attn_mask, t5_attn_mask], dim=0)
        else:
            b_t5_attn_mask = None
    else:
        b_img_ids = img_ids
        b_txt_ids = txt_ids
        b_txt = txt
        b_vec = vec
        b_t5_attn_mask = t5_attn_mask

    for t_curr, t_prev in zip(tqdm(timesteps[:-1]), timesteps[1:]):
        t_vec = torch.full((b_img_ids.shape[0],), t_curr, dtype=img.dtype, device=img.device)

        # classifier free guidance
        if neg_txt is not None and neg_vec is not None:
            b_img = torch.cat([img, img], dim=0)
        else:
            b_img = img

        pred = model(
            img=b_img,
            img_ids=b_img_ids,
            txt=b_txt,
            txt_ids=b_txt_ids,
            y=b_vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            txt_attention_mask=b_t5_attn_mask,
        )

        # classifier free guidance
        if neg_txt is not None and neg_vec is not None:
            pred_uncond, pred = torch.chunk(pred, 2, dim=0)
            pred = pred_uncond + cfg_scale * (pred - pred_uncond)

        img = img + (t_prev - t_curr) * pred

    return img


def do_sample(
    accelerator: Optional[accelerate.Accelerator],
    model: flux_models.Flux,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    l_pooled: torch.Tensor,
    t5_out: torch.Tensor,
    txt_ids: torch.Tensor,
    num_steps: int,
    guidance: float,
    t5_attn_mask: Optional[torch.Tensor],
    is_schnell: bool,
    device: torch.device,
    flux_dtype: torch.dtype,
    neg_l_pooled: Optional[torch.Tensor] = None,
    neg_t5_out: Optional[torch.Tensor] = None,
    neg_t5_attn_mask: Optional[torch.Tensor] = None,
    cfg_scale: Optional[float] = None,
):
    logger.info(f"num_steps: {num_steps}")
    timesteps = get_schedule(num_steps, img.shape[1], shift=not is_schnell)

    # denoise initial noise
    if accelerator:
        with accelerator.autocast(), torch.no_grad():
            x = denoise(
                model,
                img,
                img_ids,
                t5_out,
                txt_ids,
                l_pooled,
                timesteps,
                guidance,
                t5_attn_mask,
                neg_t5_out,
                neg_l_pooled,
                neg_t5_attn_mask,
                cfg_scale,
            )
    else:
        with torch.autocast(device_type=device.type, dtype=flux_dtype), torch.no_grad():
            x = denoise(
                model,
                img,
                img_ids,
                t5_out,
                txt_ids,
                l_pooled,
                timesteps,
                guidance,
                t5_attn_mask,
                neg_t5_out,
                neg_l_pooled,
                neg_t5_attn_mask,
                cfg_scale,
            )

    return x


def generate_image(
    model,
    clip_l: CLIPTextModel,
    t5xxl,
    ae,
    prompt: str,
    seed: Optional[int],
    image_width: int,
    image_height: int,
    steps: Optional[int],
    guidance: float,
    negative_prompt: Optional[str],
    cfg_scale: float,
    args,
    accelerator: Optional[accelerate.Accelerator],
    device: torch.device,
    flux_dtype: torch.dtype,
):
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    logger.info(f"Seed: {seed}")

    # make first noise with packed shape
    packed_latent_height, packed_latent_width = math.ceil(image_height / 16), math.ceil(image_width / 16)
    noise_dtype = torch.float32 if is_fp8(flux_dtype) else flux_dtype
    noise = torch.randn(
        1,
        packed_latent_height * packed_latent_width,
        16 * 2 * 2,
        device=device,
        dtype=noise_dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

    # prepare img and img ids
    img_ids = flux_utils.prepare_img_ids(1, packed_latent_height, packed_latent_width)

    # prepare embeddings
    logger.info("Encoding prompts...")
    clip_l = clip_l.to(device)
    t5xxl = t5xxl.to(device)

    def encode(prpt: str):
        tokens_and_masks = tokenize_strategy.tokenize(prpt)
        with torch.no_grad():
            if is_fp8(clip_l_dtype):
                with accelerator.autocast():
                    l_pooled, _, _, _ = encoding_strategy.encode_tokens(tokenize_strategy, [clip_l, None], tokens_and_masks)
            else:
                with torch.autocast(device_type=device.type, dtype=clip_l_dtype):
                    l_pooled, _, _, _ = encoding_strategy.encode_tokens(tokenize_strategy, [clip_l, None], tokens_and_masks)

            if is_fp8(t5xxl_dtype):
                with accelerator.autocast():
                    _, t5_out, txt_ids, t5_attn_mask = encoding_strategy.encode_tokens(
                        tokenize_strategy, [clip_l, t5xxl], tokens_and_masks, args.apply_t5_attn_mask
                    )
            else:
                with torch.autocast(device_type=device.type, dtype=t5xxl_dtype):
                    _, t5_out, txt_ids, t5_attn_mask = encoding_strategy.encode_tokens(
                        tokenize_strategy, [None, t5xxl], tokens_and_masks, args.apply_t5_attn_mask
                    )
        return l_pooled, t5_out, txt_ids, t5_attn_mask

    # Encode positive prompt
    l_pooled, t5_out, txt_ids, t5_attn_mask = encode(prompt)
    # Encode negative prompt if provided
    if negative_prompt:
        neg_l_pooled, neg_t5_out, _, neg_t5_attn_mask = encode(negative_prompt)
    else:
        neg_l_pooled, neg_t5_out, neg_t5_attn_mask = None, None, None

    # NaN check
    if torch.isnan(l_pooled).any():
        raise ValueError("NaN in l_pooled")
    if torch.isnan(t5_out).any():
        raise ValueError("NaN in t5_out")

    if args.offload:
        clip_l = clip_l.cpu()
        t5xxl = t5xxl.cpu()
    device_utils.clean_memory()

    # generate image
    logger.info("Generating image...")
    model = model.to(device)
    if steps is None:
        steps = 4 if is_schnell else 50

    img_ids = img_ids.to(device)
    t5_attn_mask = t5_attn_mask.to(device) if args.apply_t5_attn_mask else None

    x = do_sample(
        accelerator,
        model,
        noise,
        img_ids,
        l_pooled,
        t5_out,
        txt_ids,
        steps,
        guidance,
        t5_attn_mask,
        is_schnell,
        device,
        flux_dtype,
        neg_l_pooled,
        neg_t5_out,
        neg_t5_attn_mask,
        cfg_scale,
    )
    if args.offload:
        model = model.cpu()
    device_utils.clean_memory()

    # unpack
    x = x.float()
    x = einops.rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=packed_latent_height, w=packed_latent_width, ph=2, pw=2)

    # decode
    logger.info("Decoding image...")
    ae = ae.to(device)
    with torch.no_grad():
        if is_fp8(ae_dtype):
            with accelerator.autocast():
                x = ae.decode(x)
        else:
            with torch.autocast(device_type=device.type, dtype=ae_dtype):
                x = ae.decode(x)
    if args.offload:
        ae = ae.cpu()

    x = x.clamp(-1, 1)
    x = x.permute(0, 2, 3, 1)
    img = Image.fromarray((127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)[0])

    # save image with timestamp and prompt hash to ensure uniqueness
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    prompt_hash = abs(hash(prompt)) % (10 ** 8)  # Simple hash to avoid filesystem issues
    filename = f"{timestamp}_{prompt_hash}.png"
    output_path = os.path.join(args.output_dir, filename)
    img.save(output_path)

    logger.info(f"Saved image to {output_path}")


def read_prompts(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"Read {len(prompts)} prompts from {file_path}")
    return prompts

if __name__ == "__main__":
    target_height = 768  # 1024
    target_width = 1360  # 1024

    device = get_preferred_device()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--clip_l", type=str, required=False)
    parser.add_argument("--t5xxl", type=str, required=False)
    parser.add_argument("--ae", type=str, required=False)
    parser.add_argument("--apply_t5_attn_mask", action="store_true")
    # Change to accept multiple prompts
    parser.add_argument("--prompts", type=str, nargs='+', default=["A photo of a cat"], help="List of prompts")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="base dtype")
    parser.add_argument("--clip_l_dtype", type=str, default=None, help="dtype for clip_l")
    parser.add_argument("--ae_dtype", type=str, default=None, help="dtype for ae")
    parser.add_argument("--t5xxl_dtype", type=str, default=None, help="dtype for t5xxl")
    parser.add_argument("--flux_dtype", type=str, default=None, help="dtype for flux")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None, help="Number of steps. Default is 4 for schnell, 50 for dev")
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--offload", action="store_true", help="Offload to CPU")
    parser.add_argument(
        "--lora_weights",
        type=str,
        nargs="*",
        default=[],
        help="LoRA weights, only supports networks.lora_flux and lora_oft, each argument is a path;multiplier (semi-colon separated)",
    )
    parser.add_argument("--merge_lora_weights", action="store_true", help="Merge LoRA weights to model")
    parser.add_argument("--width", type=int, default=target_width)
    parser.add_argument("--height", type=int, default=target_height)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    seed = args.seed
    steps = args.steps
    guidance_scale = args.guidance

    def is_fp8(dt):
        return dt in [torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz]
    
    args.prompts = read_prompts(args.prompts[0])

    dtype = str_to_dtype(args.dtype)
    clip_l_dtype = str_to_dtype(args.clip_l_dtype, dtype)
    t5xxl_dtype = str_to_dtype(args.t5xxl_dtype, dtype)
    ae_dtype = str_to_dtype(args.ae_dtype, dtype)
    flux_dtype = str_to_dtype(args.flux_dtype, dtype)

    logger.info(f"Dtypes for clip_l, t5xxl, ae, flux: {clip_l_dtype}, {t5xxl_dtype}, {ae_dtype}, {flux_dtype}")

    loading_device = "cpu" if args.offload else device

    use_fp8 = [is_fp8(d) for d in [dtype, clip_l_dtype, t5xxl_dtype, ae_dtype, flux_dtype]]
    if any(use_fp8):
        accelerator = accelerate.Accelerator(mixed_precision="bf16")
    else:
        accelerator = None
    
    # Create output directory before processing
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory '{args.output_dir}' is ready.")
    except Exception as e:
        logger.error(f"Failed to create output directory '{args.output_dir}': {e}")
        raise

    # Load clip_l
    logger.info(f"Loading clip_l from {args.clip_l}...")
    clip_l = flux_utils.load_clip_l(args.clip_l, clip_l_dtype, loading_device)
    clip_l.eval()

    # Load t5xxl
    logger.info(f"Loading t5xxl from {args.t5xxl}...")
    t5xxl = flux_utils.load_t5xxl(args.t5xxl, t5xxl_dtype, loading_device)
    t5xxl.eval()
    
    # Load AE
    logger.info(f"Loading AE from {args.ae}...")
    ae = flux_utils.load_ae(args.ae, ae_dtype, loading_device)
    ae.eval()
    
    # Load the main model once
    logger.info(f"Loading main model from {args.ckpt_path}...")
    is_schnell, model = flux_utils.load_flow_model(args.ckpt_path, None, loading_device)
    model.eval()
    logger.info(f"Casting model to {flux_dtype}")
    model.to(flux_dtype)  # make sure model is dtype

    t5xxl_max_length = 256 if is_schnell else 512
    tokenize_strategy = strategy_flux.FluxTokenizeStrategy(t5xxl_max_length)
    encoding_strategy = strategy_flux.FluxTextEncodingStrategy()

    # Move models to device if not offloading
    if not args.offload:
        clip_l = clip_l.to(device)
        t5xxl = t5xxl.to(device)
        model = model.to(device)
        ae = ae.to(device)

    # Load LoRA weights
    lora_models: List[lora_flux.LoRANetwork] = []
    for weights_file in args.lora_weights:
        if ";" in weights_file:
            weights_file, multiplier = weights_file.split(";")
            multiplier = float(multiplier)
        else:
            multiplier = 1.0

        weights_sd = load_file(weights_file)
        is_lora = is_oft = False
        for key in weights_sd.keys():
            if key.startswith("lora"):
                is_lora = True
            if key.startswith("oft"):
                is_oft = True
            if is_lora or is_oft:
                break

        module = lora_flux if is_lora else oft_flux
        lora_model, _ = module.create_network_from_weights(multiplier, None, ae, [clip_l, t5xxl], model, weights_sd, True)

        if args.merge_lora_weights:
            lora_model.merge_to([clip_l, t5xxl], model, weights_sd)
        else:
            lora_model.apply_to([clip_l, t5xxl], model)
            info = lora_model.load_state_dict(weights_sd, strict=True)
            logger.info(f"Loaded LoRA weights from {weights_file}: {info}")
            lora_model.eval()
            lora_model.to(device)

        lora_models.append(lora_model)

    # Process each prompt
    for idx, prompt in enumerate(tqdm(args.prompts, desc="Processing Prompts")):
        current_seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1) + idx
        generate_image(
            model=model,
            clip_l=clip_l,
            t5xxl=t5xxl,
            ae=ae,
            prompt=prompt,
            seed=current_seed,
            image_width=args.width,
            image_height=args.height,
            steps=steps,
            guidance=guidance_scale,
            negative_prompt=args.negative_prompt,
            cfg_scale=args.cfg_scale,
            args=args,
            accelerator=accelerator,
            device=device,
            flux_dtype=flux_dtype,
        )

    # Clean up
    if args.offload:
        clip_l = clip_l.cpu()
        t5xxl = t5xxl.cpu()
        model = model.cpu()
        ae = ae.cpu()
    device_utils.clean_memory()

    logger.info("All images generated successfully!")

