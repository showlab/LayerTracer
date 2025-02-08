#!/bin/bash

CHECKPOINT_PATH="/path/to/unet/flux1-dev-fp8.safetensors"
CLIP_L_MODEL="/path/to/clip/clip_l.safetensors"
T5XXL_MODEL="/path/to/clip/t5xxl_fp8_e4m3fn.safetensors"
AE_MODEL="/path/to/vae/ae.safetensors"
LORA_PATH="/path/to/loras/flux_lora_icon_2458.safetensors"
OUTPUT_DIR="/path/to/output/directory"
PROMPTS_FILE="/path/to/prompts.txt"
IMAGE_WIDTH=768  
IMAGE_HEIGHT=768  
STEPS=25  

python3 "Inference_text2sequence.py" \
  --clip_l "$CLIP_L_MODEL" \
  --t5xxl "$T5XXL_MODEL" \
  --ae "$AE_MODEL" \
  --ckpt_path "$CHECKPOINT_PATH" \
  --lora_weights "$LORA_PATH" \
  --prompts "$PROMPTS_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --width "$IMAGE_WIDTH" \
  --height "$IMAGE_HEIGHT" \
  --steps "$STEPS" \
  --apply_t5_attn_mask \
  --dtype bf16 \
  --offload


