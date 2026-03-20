"""
STAR 单视频推理入口脚本。

用法:
    python run_single_video.py \
        --config config/star_single_video.yaml \
        --video_path /path/to/video.mp4 \
        --question "What is the man doing in the video?"

    # 带选项的问题
    python run_single_video.py \
        --config config/star_single_video.yaml \
        --video_path /path/to/video.mp4 \
        --question "What is the man doing?" \
        --options "A. Running" "B. Swimming" "C. Reading" "D. Cooking"
"""

import os
import sys
import argparse
import datetime

import torch
torch.manual_seed(12345)
torch.cuda.manual_seed(12345)

from omegaconf import OmegaConf

from visible_frames import get_video_info, VisibleFrames
from util import adjust_video_resolution, load_temporal_model, load_llava_model
from star_reasoning import star_reasoning

# 工具导入（与 main.py 一致，用于 globals() 动态实例化）
from tools.yolo_tracker import YOLOTracker
from tools.image_captioner import ImageCaptioner
from tools.frame_selector import FrameSelector
from tools.image_qa import ImageQA
from tools.temporal_grounding import TemporalGrounding
from tools.image_grid_qa import ImageGridQA
from tools.summarizer import Summarizer
from tools.patch_zoomer import PatchZoomer
from tools.temporal_qa import TemporalQA
from tools.video_qa import VideoQA
from tools.image_captioner_llava import ImageCaptionerLLaVA
from tools.image_grid_select import ImageGridSelect
from tools.temporal_referring import TemporalReferring


def get_tool_instances(conf):
    """实例化工具列表"""
    tool_list = conf.tool.tool_list
    tool_instances = []
    for tool_name in tool_list:
        if tool_name in globals():
            print(f"Initializing tool: {tool_name}")
            tool_instances.append(globals()[tool_name](conf))
        else:
            print(f"Warning: Tool class '{tool_name}' not found, skipping")
    return tool_instances


def main():
    parser = argparse.ArgumentParser(description="STAR Single Video Reasoning")
    parser.add_argument('--config', default="config/star_single_video.yaml", type=str,
                        help="Path to YAML config file")
    parser.add_argument('--video_path', required=True, type=str,
                        help="Path to input video file")
    parser.add_argument('--question', required=True, type=str,
                        help="Question about the video")
    parser.add_argument('--options', nargs='*', default=None,
                        help="Optional answer choices, e.g. 'A. Running' 'B. Swimming'")
    parser.add_argument('--max_iterations', type=int, default=None,
                        help="Override max iterations from config")
    
    args = parser.parse_args()
    
    # 加载配置
    conf = OmegaConf.load(args.config)
    
    # 覆盖 max_iterations
    if args.max_iterations is not None:
        if "star" not in conf:
            conf.star = {}
        conf.star.max_iterations = args.max_iterations
    
    # 构建带选项的问题
    question = args.question
    if args.options:
        question_w_options = question + "\n" + "\n".join(args.options)
    else:
        question_w_options = question
    
    video_path = args.video_path
    
    # 检查视频文件
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"STAR Single Video Reasoning")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Question: {question}")
    if args.options:
        print(f"Options: {args.options}")
    print(f"Config: {args.config}")
    print(f"{'='*60}")
    
    # 调整视频分辨率（确保偶数）
    adjust_video_resolution(video_path)
    
    # 获取视频信息
    video_info = get_video_info(video_path)
    print(f"Video duration: {video_info['duration']:.2f}s, "
          f"frames: {video_info['total_frames']}, "
          f"fps: {video_info['fps']:.1f}")
    
    # 创建 VisibleFrames
    visible_frames = VisibleFrames(
        video_path=video_path,
        init_sec_interval=conf.visible_frames.init_sec_interval,
        init_interval_num=conf.visible_frames.init_interval_num,
        min_interval=conf.visible_frames.min_interval,
        min_sec_interval=conf.visible_frames.min_sec_interval,
    )
    print(f"Initial visible frames: {visible_frames.get_frame_count()}")
    
    # 实例化工具
    tool_instances = get_tool_instances(conf)
    
    # 加载时序模型（如果需要）
    if any(isinstance(t, (TemporalGrounding, TemporalQA, TemporalReferring)) for t in tool_instances):
        temporal_model = load_temporal_model(
            weight_path=conf.tool.temporal_model.weight_path,
            device=conf.tool.temporal_model.device,
            llm_type=conf.tool.temporal_model.llm_type
        )
        for t in tool_instances:
            if isinstance(t, (TemporalGrounding, TemporalQA, TemporalReferring)):
                t.set_model(temporal_model)
    
    # 加载 LLaVA 模型（如果需要，ImageQA 和 ImageCaptionerLLaVA model_path 相同则共用）
    has_image_qa = any(isinstance(t, ImageQA) for t in tool_instances)
    has_captioner_llava = any(isinstance(t, ImageCaptionerLLaVA) for t in tool_instances)
    if has_image_qa or has_captioner_llava:
        qa_model_path = conf.tool.image_qa.model_path
        qa_device = conf.tool.image_qa.device
        cap_model_path = conf.tool.image_captioner_llava.model_path
        cap_device = conf.tool.image_captioner_llava.device
        # 加载 ImageQA 的模型
        llava_tok, llava_mdl, llava_proc = None, None, None
        if has_image_qa:
            llava_tok, llava_mdl, llava_proc = load_llava_model(qa_model_path, qa_device)
            for t in tool_instances:
                if isinstance(t, ImageQA):
                    t.set_model(llava_tok, llava_mdl, llava_proc)
        # ImageCaptionerLLaVA：model_path 相同则共用，否则单独加载
        if has_captioner_llava:
            if has_image_qa and cap_model_path == qa_model_path:
                for t in tool_instances:
                    if isinstance(t, ImageCaptionerLLaVA):
                        t.set_model(llava_tok, llava_mdl, llava_proc)
            else:
                cap_tok, cap_mdl, cap_proc = load_llava_model(cap_model_path, cap_device)
                for t in tool_instances:
                    if isinstance(t, ImageCaptionerLLaVA):
                        t.set_model(cap_tok, cap_mdl, cap_proc)
    
    # 给所有工具设置 visible_frames 和 video_path
    for tool_instance in tool_instances:
        tool_instance.set_frames(visible_frames)
        tool_instance.set_video_path(video_path)
    
    # 执行 STAR 推理
    answer = star_reasoning(
        question=question,
        question_w_options=question_w_options,
        tool_instances=tool_instances,
        visible_frames=visible_frames,
        conf=conf,
    )
    
    print(f"\n{'='*60}")
    print(f"FINAL ANSWER: {answer}")
    print(f"{'='*60}")
    
    # 保存结果
    os.makedirs(conf.output_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    import json
    result = {
        "video_path": video_path,
        "question": question,
        "question_w_options": question_w_options,
        "answer": answer,
        "video_info": video_info,
        "visible_frames_num": visible_frames.get_frame_count(),
    }
    
    output_file = os.path.join(conf.output_path, f"star_result_{timestamp}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Result saved to: {output_file}")
    
    return answer


if __name__ == "__main__":
    main()
