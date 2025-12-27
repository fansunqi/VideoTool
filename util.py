import os
import cv2
import sys
import pdb
import json
import torch
import ffmpeg
import shutil
import pickle
from langchain_core.tools import Tool

sys.path.append("projects/Grounded-Video-LLM")
from models.llava_next_video import LLAVA_NEXT_VIDEO
from inference import parse_args, parse_time_interval


args = parse_args()


def save_to_json(output_data, output_file):
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

def adjust_video_resolution(video_path: str):
    # 解析视频路径
    dir_name, file_name = os.path.split(video_path)
    file_base, file_ext = os.path.splitext(file_name)
    backup_path = os.path.join(dir_name, f"{file_base}_org{file_ext}")
    
    # 获取视频信息
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if not video_stream:
        print(f"\nError: Cannot find video stream in {video_path}")
        return
    
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    
    # 检查是否需要裁剪
    new_width = width if width % 2 == 0 else width - 1
    new_height = height if height % 2 == 0 else height - 1
    if new_width == width and new_height == height:
        # print("No need to crop. The resolution is already even.")
        return
    
    # 备份原视频
    os.rename(video_path, backup_path)
    
    # 处理视频
    ffmpeg.input(backup_path).filter('crop', new_width, new_height, 0, 0).output(video_path).run()
    
    print(f"\nVideo cropped to even resolution and saved as {video_path}, original saved as {backup_path}")  


def backup_file(opt, conf, timestamp, is_test=False):
    current_script_path = os.path.abspath(__file__) 
    current_script_dir = os.path.dirname(current_script_path)
    
    if not os.path.exists(conf.output_path):
        os.makedirs(conf.output_path)
    
    if is_test:
        test_file_path = os.path.join(current_script_dir, "test_tools_all_video.py")
        shutil.copy(test_file_path, os.path.join(conf.output_path, f"test_tools_all_video_{timestamp}.py"))
    else:
        main_file_path = os.path.join(current_script_dir, "main.py")
        shutil.copy(main_file_path, os.path.join(conf.output_path, f"main_{timestamp}.py"))

        reansoning_file_path = os.path.join(current_script_dir, "reasoning.py")
        shutil.copy(reansoning_file_path, os.path.join(conf.output_path, f"reasoning_{timestamp}.py"))
    
    config_basename = os.path.basename(opt.config).split('.')[0]
    shutil.copy(opt.config, os.path.join(conf.output_path, f"{config_basename}_{timestamp}.yaml"))   
    

def load_cache(mannual_cache_file):
    if os.path.exists(mannual_cache_file):
        print(f"Loading LLM cache from {mannual_cache_file}...\n")
        with open(mannual_cache_file, "rb") as f:
            mannual_cache = pickle.load(f)
    else:
        print(f"Creating LLM cache: {mannual_cache_file}...\n")
        mannual_cache = {}
    return mannual_cache


def save_cache(mannual_cache, query, steps, mannual_cache_file):
    mannual_cache[query] = steps
    print("\nSaving cache...")
    with open(mannual_cache_file, "wb") as f:
        pickle.dump(mannual_cache, f)


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


def read_lvb_subtitles(subtitles):
    # with open(subtitle_path, 'r', encoding='utf-8') as f:
    #     subtitles = json.load(f)
    
    desp_all = ""
    desp_line_template = """{start} - {end}: {line}\n"""
    if 'start' in subtitles[0]:
        for subtitle in subtitles:
            desp_line = desp_line_template.format(
                start=subtitle['start'],
                end=subtitle['end'],
                line=subtitle['line']
            )
            desp_all += desp_line
    elif 'timestamp' in subtitles[0]:
        for subtitle in subtitles:
            desp_line = desp_line_template.format(
                start=subtitle['timestamp'][0],
                end=subtitle['timestamp'][1],
                line=subtitle['text']
            )
            desp_all += desp_line
    else:
        raise ValueError("Invalid subtitle format. Expected 'start' or 'timestamp' key.")
        
    return desp_all
    

if __name__ == "__main__":
    # test_video_path = "/share_data/NExT-QA/NExTVideo/0071/2617504308.mp4"
    # adjust_video_resolution(test_video_path)
    
    subtitle_path = "/mnt/Shared_03/fsq/LongVideoBench/subtitles/__Bchxr3ejw_en.json"
    desp_all = read_lvb_subtitles(subtitle_path)
    
    pdb.set_trace()
