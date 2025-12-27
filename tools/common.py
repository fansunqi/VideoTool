import cv2
import sys
import time
import torch
import base64
import openai

sys.path.append("projects/Grounded-Video-LLM")
from models.llava_next_video import LLAVA_NEXT_VIDEO
from inference import parse_args, parse_time_interval

args = parse_args()

# Resize the image while keeping aspect ratio
def image_resize_for_vlm(frame, inter=cv2.INTER_AREA):
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    max_short_side = 768
    max_long_side = 2000
    if aspect_ratio > 1:
        new_width = min(width, max_long_side)
        new_height = int(new_width / aspect_ratio)
        if new_height > max_short_side:
            new_height = max_short_side
            new_width = int(new_height * aspect_ratio)
    else:
        new_height = min(height, max_long_side)
        new_width = int(new_height * aspect_ratio)
        if new_width > max_short_side:
            new_width = max_short_side
            new_height = int(new_width / aspect_ratio)
    resized_frame = cv2.resize(
        frame, (new_width, new_height), interpolation=inter)
    return resized_frame



def load_temporal_model(weight_path, device, llm_type):
    config_path = f"{weight_path}/Phi-3.5-vision-instruct"
    tokenizer_path = f"{weight_path}/Phi-3.5-mini-instruct"
    pretrained_video_path = f"{weight_path}/internvideo/vision-encoder-InternVideo2-stage2_1b-224p-f4.pt"
    pretrained_vision_proj_llm_path = f"{weight_path}/Phi-3.5-vision-instruct-seperated"
    ckpt_path = f"{weight_path}/ckpt/sft_llava_next_video_phi3.5_mix_sft_multi_modal_projector_video_projecter_language_model.pth"
    
    print("Start loading temporal model...\n")
    
    # TODO 查看一下这里各个参数的含义
    model = LLAVA_NEXT_VIDEO(
        dtype=args.dtype, 
        stage=args.stage, 
        max_txt_len=args.max_txt_len, 
        num_frames=args.num_frames,
        num_segs=args.num_segs,
        num_temporal_tokens=args.num_temporal_tokens,
        lora=args.lora,
        llm=llm_type,
        attn_implementation=args.attn_implementation,
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        pretrained_video_path=pretrained_video_path,
        pretrained_vision_proj_llm_path=pretrained_vision_proj_llm_path, 
    )
    ckpt = torch.load(ckpt_path, map_location='cpu')['model']
    if 'multi_modal_projector' in ckpt.keys():
        model.multi_modal_projector.load_state_dict(ckpt['multi_modal_projector'])
    if 'video_projecter' in ckpt.keys():
        model.video_projecter.load_state_dict(ckpt['video_projecter'])
    if 'language_model' in ckpt.keys():
        model.language_model.load_state_dict(ckpt['language_model'])  
    model.eval()
    model.to(device)
    print("Finish loading temporal model.\n")

    return model