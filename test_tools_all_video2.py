import os
import sys
import pdb
import json
import datetime
import shutil
import pickle
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf

seed = 12345
import random
random.seed(seed)

import numpy as np
np.random.seed(seed)

import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from prompts import (
    QUERY_PREFIX,
    TOOLS_RULE,
    ASSISTANT_ROLE,
    QUERY_PREFIX_DES,
)

from dataset import get_dataset

from util import (
    save_to_json,
    adjust_video_resolution,
    backup_file,
)


# from tools.yolo_tracker import YOLOTracker
# from tools.image_captioner import ImageCaptioner
# from tools.frame_selector import FrameSelector
# from tools.temporal_qa import TemporalQA
# from tools.video_qa import VideoQA
# from tools.image_grid_qa import ImageGridQA
# from tools.temporal_grounding import TemporalGrounding
from tools.video_qa_internvl import VideoQAInternVL

from visible_frames import VisibleFrames


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demo")               
    parser.add_argument('--config', default="config/videomme.yaml",type=str)                           
    opt = parser.parse_args()
    conf = OmegaConf.load(opt.config)

    backup_file(opt, conf, timestamp, is_test=True)   

    if conf.to_txt:
        log_path = os.path.join(conf.output_path, f"log_{timestamp}.txt")
        f = open(log_path, "w")
        sys.stdout = f
  
    # image_captioner = ImageCaptioner()
    # temporal_qa = TemporalQA(conf=conf)
    # video_qa = VideoQA(conf=conf)
    # image_grid_qa = ImageGridQA(conf=conf)
    # temporal_grounding = TemporalGrounding(conf=conf)
    video_qa_internvl = VideoQAInternVL(conf=conf)

    # 数据集
    quids_to_exclude = conf["quids_to_exclude"] if "quids_to_exclude" in conf else None
    num_examples_to_run = conf["num_examples_to_run"] if "num_examples_to_run" in conf else -1
    start_num = conf["start_num"] if "start_num" in conf else 0
    specific_quids = conf["specific_quids"] if "specific_quids" in conf else None
    dataset = get_dataset(conf, quids_to_exclude, num_examples_to_run, start_num, specific_quids)

    try_num = conf.try_num
    all_results = []

    # temporal_grounding 结果存储
    # temporal_grounding_results = {}

    for data in tqdm(dataset):

        print(f"\n\nProcessing: {data['quid']}")

        video_path = data["video_path"]
        question = data["question"]
        options = data["options"]
        question_w_options = data["question_w_options"]

        visible_frames = VisibleFrames(video_path=video_path)
        
        print(video_path)
        print(visible_frames.video_info["duration"])

        # temporal_qa.set_video_path(video_path)
        # video_qa.set_video_path(video_path)
        # video_qa.set_frames(visible_frames)
        # image_grid_qa.set_video_path(video_path)
        # image_grid_qa.set_frames(visible_frames)
        # temporal_grounding.set_video_path(video_path)
        # temporal_grounding.set_frames(visible_frames)
        video_qa_internvl.set_video_path(video_path)
        video_qa_internvl.set_frames(visible_frames)
        
        result = data
        result["answers"] = []
        result["question_w_options"] = question_w_options

        # trim
        adjust_video_resolution(video_path)
        
        for try_count in range(try_num):

            input_prompt = question_w_options
            print("Input prompt: ", input_prompt)

            if conf.try_except_mode:
                try:
                    # output = temporal_qa.inference(input = input_prompt)
                    # output = video_qa.inference(input = input_prompt)
                    # output = image_grid_qa.inference(input=input_prompt)
                    # output = temporal_grounding.inference(input=input_prompt)
                    output = video_qa_internvl.inference(input=input_prompt)
                except Exception as e:
                    output = "Error"
                    print(f"Error during inference: {e}")
                    continue
            else:
                # output = temporal_qa.inference(input = input_prompt)
                # output = video_qa.inference(input = input_prompt)
                # output = image_grid_qa.inference(input=input_prompt)
                # output = temporal_grounding.inference(input=input_prompt)
                output = video_qa_internvl.inference(input=input_prompt)
            
            # 存储 tg 数据
            # temporal_grounding_results["question"] = output
        
            print("Output Answer: ", output)
            
            result["answers"].append(output)

        all_results.append(result)

    output_file = os.path.join(conf.output_path, f"results_{timestamp}.json")
    save_to_json(all_results, output_file)
    print(f"\n{str(len(all_results))} results saved")  

    # tg 结果存储
    # tg_output_file = os.path.join(conf.output_path, f"tg_results_{timestamp}.json")
    # with open(tg_output_file, 'w') as f:
    #     json.dump(temporal_grounding_results, f, indent=4) 

    if conf.to_txt:
        sys.stdout = sys.__stdout__
        f.close()

    