import sys
sys.path.append("projects/Grounded-Video-LLM")
import pdb
import torch
import decord
import numpy as np
from decord import VideoReader
from omegaconf import OmegaConf


from models.llava_next_video import LLAVA_NEXT_VIDEO
from inference import parse_args, parse_time_interval
from mm_utils.video_utils import get_frame_indices
from mm_utils.utils import *
from datasets.chat.base_template import LLaMA3_Template, Vicuna_Template, Phi_3_5_Template, DEFAULT_IMAGE_TOKEN, GROUNDING_TOKEN

_temporal_args = None

def _get_temporal_args():
    global _temporal_args
    if _temporal_args is None:
        original_argv = sys.argv
        sys.argv = sys.argv[:1]
        _temporal_args = parse_args()
        sys.argv = original_argv
    return _temporal_args


def prompts(name, description):
    
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator

class TemporalQA:
    def __init__(
        self,
        conf = None, 
    ):
        self.conf = conf
        
        self.visible_frames = None
        self.video_path = None

        self.model = None

        self.llm_type = conf.tool.temporal_model.llm_type
        self.device = conf.tool.temporal_model.device

        self.mode = conf.tool.temporal_qa.mode
    
    def set_frames(self, visible_frames):
        self.visible_frames = visible_frames  
    
    def set_video_path(self, video_path):
        self.video_path = video_path  
    
    def set_model(self, model):
        self.model = model
     
    def read_frames_decord(self, video_path):
        video_reader = VideoReader(video_path, num_threads=1)
        
        vlen = len(video_reader)
        fps = video_reader.get_avg_fps()
        duration = vlen / float(fps)
        
        frame_indices = get_frame_indices(
            num_frames = 96,   # 默认的值
            vlen = vlen, 
            sample = "middle", 
            fix_start = None,
            input_fps = fps,
            max_num_frames = -1
        )
        
        try:
            frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
        except decord.DECORDError as e:
            print(f'解码错误: {video_path}, {vlen}, {fps}, {duration}')
            print(f'decord.DECORDError报错: {e}')
        except Exception as e:
            print(f'解码错误: {video_path}, {vlen}, {fps}, {duration}')
            print(f'Exception报错: {e}')

        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8, 0-255, RGB

        return frames, frame_indices, float(fps), vlen, duration

    def create_inputs(self, video_path, prompt_videoqa):
        video_processor = frame_transform(image_size=224, mean=INTERNVIDEO_MEAN, std=INTERNVIDEO_STD)
        image_processor = frame_transform(image_size=336, mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
        
        args = _get_temporal_args()
        if self.mode == "by_video_path":
            pixel_values, frame_indices, fps, total_frame_num, duration = self.read_frames_decord(video_path)
        elif self.mode == "by_visible_frames":
            pixel_values = self.visible_frames.get_images_rgb_tchw(args.num_frames)
        else:
            raise ValueError("temporal_qa.mode error")

        temporal_pixel_values = []
        for i in range(pixel_values.shape[0]): 
            temporal_pixel_values.append(video_processor(pixel_values[i]))
        temporal_pixel_values = torch.tensor(np.array(temporal_pixel_values)) # [num_frames, 3, 224, 224]
        temporal_pixel_values = temporal_pixel_values.unsqueeze(0)

        num_frames_per_seg = int(args.num_frames // args.num_segs)
        indices_spatial = [(i*num_frames_per_seg) + int(num_frames_per_seg/2) for i in range(args.num_segs)]
        spatial_pixel_values = []
        for i_spatial in indices_spatial:
            spatial_pixel_values.append(image_processor(pixel_values[i_spatial]))
        spatial_pixel_values = torch.tensor(np.array(spatial_pixel_values)) # [num_segs, 3, 336, 336]
        spatial_pixel_values = spatial_pixel_values.unsqueeze(0)
        
        chat_template = {'phi3.5': Phi_3_5_Template(), 'llama3': LLaMA3_Template(), 'vicuna': Vicuna_Template()}[self.llm_type]
        # grounding, qa 中的 qa
        conv = [
            {"from": "human", "value": DEFAULT_IMAGE_TOKEN + '\n'+ prompt_videoqa},
            {"from": "gpt", "value": ''}                
        ]
        sep, eos = chat_template.separator.apply()
        prompt = chat_template.encode(conv).replace(eos, '')

        samples = {
            "video_ids": [video_path],
            "question_ids": [video_path],
            "prompts": [prompt],
            "temporal_pixel_values": temporal_pixel_values.to(self.device),
            "spatial_pixel_values": spatial_pixel_values.to(self.device),
        }
    
        return samples
    
    @prompts(
        name = "temporal-qa-tool",
        description = "placeholder"
    )
    def inference(self, input):
        samples_videoqa = self.create_inputs(self.video_path, input)
        
        args = _get_temporal_args()
        generate_kwargs = {
            "do_sample": args.do_sample,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "temperature":args.temperature,
            "top_p":args.top_p,
        }
        
        with torch.cuda.amp.autocast(enabled=True, dtype=self.model.dtype):
            with torch.inference_mode():
                pred_texts_videoqa = self.model.generate(samples_videoqa, **generate_kwargs)[0]
        
        return pred_texts_videoqa
        
        
        

if __name__ == "__main__":
    import json
    from visible_frames import VisibleFrames
    from util import load_temporal_model

    conf = OmegaConf.load("config/star_single_video.yaml")
    with open("testcases/testcase.json") as f:
        tc = json.load(f)

    video_path = tc["video_path"]
    question = tc["question"]

    visible_frames = VisibleFrames(
        video_path=video_path,
        init_interval_num=conf.visible_frames.init_interval_num,
        min_sec_interval=conf.visible_frames.min_sec_interval,
    )
    print(f"Initial visible frames: {visible_frames.get_frame_count()}")

    temporal_qa = TemporalQA(conf)
    temporal_qa.set_frames(visible_frames)
    temporal_qa.set_video_path(video_path)

    temporal_model = load_temporal_model(
        weight_path=conf.tool.temporal_model.weight_path,
        device=conf.tool.temporal_model.device,
        llm_type=conf.tool.temporal_model.llm_type,
    )
    temporal_qa.set_model(temporal_model)

    result = temporal_qa.inference(input=question)
    print(f"Result: {result}")

# python -m tools.temporal_qa