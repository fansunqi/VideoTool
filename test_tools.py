import argparse
from omegaconf import OmegaConf

from visible_frames import get_video_info, VisibleFrames
from tools.image_captioner import ImageCaptioner
from tools.frame_selector import FrameSelector
from tools.image_qa import ImageQA
from tools.yolo_tracker import YOLOTracker
from tools.temporal_grounding import TemporalGrounding
from tools.image_grid_qa import ImageGridQA
from tools.temporal_qa import TemporalQA
from tools.summarizer import Summarizer
from tools.image_captioner_llava import ImageCaptionerLLaVA

import pdb

parser = argparse.ArgumentParser(description="demo")               
parser.add_argument('--config', default="config/videomme.yaml",type=str)                           
opt = parser.parse_args()
conf = OmegaConf.load(opt.config)

# video_path = "/home/fsq/video_agent/ToolChainVideo/projects/Grounded-Video-LLM/experiments/_3klvlS4W7A.mp4"

video_path = "/share_data/NExT-QA/NExTVideo/1164/3238737531.mp4"
question = "How many children are in the video? Choose your answer from below selections: A.one, B.three, C.seven, D.two, E.five."
init_sec_interval = 10

# 创建可见帧管理器
visible_frames = VisibleFrames(video_path=video_path, init_sec_interval=init_sec_interval)

'''
# 再打印信息
print("\n可见帧描述:")
print(visible_frames.get_frame_descriptions())

# frame selector
frame_selector = FrameSelector(conf=conf)
frame_selector.set_frames(visible_frames=visible_frames)
frame_selector.inference(input=question)

# 查看其它工具的 visible_frames 是否改变
# 再打印信息
print("\n可见帧描述:")
print(visible_frames.get_frame_descriptions())
'''

'''
from util import load_temporal_model
temporal_model = load_temporal_model(
    weight_path=conf.tool.temporal_model.weight_path,
    device=conf.tool.temporal_model.device,
    llm_type=conf.tool.temporal_model.llm_type
)
    
# temporal_qa
temporal_qa = TemporalQA(conf)
temporal_qa.set_video_path(video_path)
temporal_qa.set_frames(visible_frames)
temporal_qa.set_model(temporal_model)
result = temporal_qa.inference(input=question)
print(f"Result: {result}")
'''

'''
# image_grid_qa
image_grid_qa = ImageGridQA(conf)
image_grid_qa.set_video_path(video_path)
image_grid_qa.set_frames(visible_frames)
result = image_grid_qa.inference(input=question)
print(result)
'''

'''
# temporal grounding
temporal_grounding = TemporalGrounding(conf=conf)
temporal_grounding.set_frames(visible_frames)
'''

'''
# YOLO Tracker
yolo_tracker = YOLOTracker(conf=conf)
yolo_tracker.set_frames(visible_frames)
results = yolo_tracker.inference(input="children")
'''

'''
# image_qa
image_qa = ImageQA(conf=conf)
image_qa.set_frames(visible_frames)
# question = "How many children are in the video? Choose your answer from below selections: A.one, B.three, C.seven, D.two, E.five."
question = "How many children are in the video?"
image_qa.inference(input=question)
qa_desp = visible_frames.get_qa_descriptions()
print(qa_desp)
'''

image_captioner_llava = ImageCaptionerLLaVA(conf=conf)
image_captioner_llava.set_frames(visible_frames)
image_captioner_llava.inference(input="placeholder")
desp = visible_frames.get_frame_descriptions()
print(desp)


# summarizer
summarizer = Summarizer(conf=conf)
summarizer.set_frames(visible_frames=visible_frames)
output = summarizer.inference(input=question)
print(output)


'''
# image_captioner
image_captioner = ImageCaptioner(conf=conf)
image_captioner.set_frames(visible_frames)
image_captioner.inference(input="placeholder")

# 再打印信息
print("\n可见帧描述:")
print(visible_frames.get_frame_descriptions())

# frame selector
frame_selector = FrameSelector(conf=conf)
frame_selector.set_frames(visible_frames=visible_frames)
frame_selector.inference(input=question)

# 查看其它工具的 visible_frames 是否改变
# 再打印信息
print("\n可见帧描述:")
print(visible_frames.get_frame_descriptions())
'''



