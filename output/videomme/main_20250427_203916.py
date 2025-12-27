import os
import sys
import pdb
import datetime


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


from dataset import get_dataset

from util import (
    save_to_json, 
    adjust_video_resolution,
    backup_file,
    load_cache,
    load_temporal_model, 
)


# from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from engine.openai import ChatOpenAI

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

from visible_frames import get_video_info, VisibleFrames

from reasoning import (
    langgraph_reasoning,
    spatiotemporal_reasoning,
    spatiotemporal_reasoning_nextqa,
)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")



def get_tools(conf):
    # TODO 更好地打印工具
    tool_list = conf.tool.tool_list
    tool_instances = []
    for tool_name in tool_list:
        tool_instances.append(globals()[tool_name](conf))
    # print(f"tool_instances: {str(tool_instances)}")
    
    tools = []
    for tool_instance in tool_instances:
        for e in dir(tool_instance):
            if e.startswith("inference"):
                func = getattr(tool_instance, e)
                tools.append(Tool(name=func.name, description=func.description, func=func))
    # print(f"tools: {str(tools)}")
    
    return tool_instances, tools


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demo")               
    parser.add_argument('--config', default="config/videomme.yaml",type=str)                           
    opt = parser.parse_args()
    conf = OmegaConf.load(opt.config)

    backup_file(opt, conf, timestamp)

    if conf.to_txt:
        log_path = os.path.join(conf.output_path, f"log_{timestamp}.txt")
        f = open(log_path, "w")
        sys.stdout = f
    
    mannual_cache_file = conf.mannual_cache_file
    mannual_cache = load_cache(mannual_cache_file)

    tool_instances, tools = get_tools(conf)
    
    if any(isinstance(tool_instance, (TemporalGrounding, TemporalQA)) for tool_instance in tool_instances):
        temporal_model = load_temporal_model(
            weight_path=conf.tool.temporal_model.weight_path,
            device=conf.tool.temporal_model.device,
            llm_type=conf.tool.temporal_model.llm_type
        )
        for tool_instance in tool_instances:
            if isinstance(tool_instance, (TemporalGrounding, TemporalQA)):
                tool_instance.set_model(temporal_model)

    tool_planner_llm = None

    # evaluator_llm
    print(f"\nInitializing eval-LLM for rephrase, model: {conf.EVAL_MODEL_NAME}")
    eval_llm = ChatOpenAI(model_string=conf.EVAL_MODEL_NAME, is_multimodal=False)

    # evaluator_cache
    eval_cache_file = None
    eval_cache = None

    quids_to_exclude = conf["quids_to_exclude"] if "quids_to_exclude" in conf else None
    num_examples_to_run = conf["num_examples_to_run"] if "num_examples_to_run" in conf else -1
    start_num = conf["start_num"] if "start_num" in conf else 0
    specific_quids = conf["specific_quids"] if "specific_quids" in conf else None
    dataset = get_dataset(conf, quids_to_exclude, num_examples_to_run, start_num, specific_quids)

    try_num = conf.try_num
    all_results = []
    
    if conf.dataset == "nextqa":
        reasoning_func = spatiotemporal_reasoning_nextqa
    elif conf.dataset == "videomme":
        reasoning_func = spatiotemporal_reasoning
    else:
        raise KeyError("conf.dataset error")

    for data in tqdm(dataset):

        print(f"\nProcessing: {data['quid']}")

        video_path = data["video_path"]
        # print(get_video_info(video_path))
        question = data["question"]
        options = data["options"]
        question_w_options = data["question_w_options"]
        
        result = data
        result["answers"] = []
        result["question_w_options"] = question_w_options

        # trim
        adjust_video_resolution(video_path)

        video_info = get_video_info(video_path)
        duration = video_info["duration"]

        print(question_w_options)
        
        visible_frames_all = 0
        for try_count in range(try_num):

            visible_frames = VisibleFrames(
                video_path=video_path, 
                init_sec_interval=conf.visible_frames.init_sec_interval,
                init_interval_num=conf.visible_frames.init_interval_num,
                min_interval=conf.visible_frames.min_interval,
                min_sec_interval=conf.visible_frames.min_sec_interval
            )
            
            for tool_instance in tool_instances:
                tool_instance.set_frames(visible_frames)
                tool_instance.set_video_path(video_path)

            # TODO 各种工具也可以加上一个 set_question 功能, 使得 question 不用再占据 input 的位置
            # TODO 简化，统一下面的代码
            
            if conf.try_except_mode:
                try:
                    if conf.reasoning_mode == "langgrah":
                        tool_chain_output = langgraph_reasoning(
                            input_question=question_w_options,
                            llm=tool_planner_llm,
                            tools=tools,
                            recursion_limit=conf.recursion_limit,
                            use_cache=conf.use_cache,
                            mannual_cache=mannual_cache,
                            mannual_cache_file=mannual_cache_file
                        )
                    elif conf.reasoning_mode == "st":
                        tool_chain_output = reasoning_func(
                            question=question,
                            question_w_options=question_w_options,
                            options=options,
                            llm=tool_planner_llm,
                            tools=tool_instances,
                            recursion_limit=conf.recursion_limit,
                            use_cache=conf.use_cache,
                            mannual_cache=mannual_cache,
                            mannual_cache_file=mannual_cache_file,
                            eval_llm=eval_llm,
                            eval_cache=eval_cache,
                            eval_cache_file=eval_cache_file,
                        )
                    else:
                        raise KeyError("conf.reasoning_mode error")
                except Exception as e:
                    print(f"Error: {e}")
                    tool_chain_output = "Error"
            else:
                if conf.reasoning_mode == "langgrah":
                    tool_chain_output = langgraph_reasoning(
                        input_question=question_w_options,
                        llm=tool_planner_llm,
                        tools=tools,
                        recursion_limit=conf.recursion_limit,
                        use_cache=conf.use_cache,
                        mannual_cache=mannual_cache,
                        mannual_cache_file=mannual_cache_file
                    )
                elif conf.reasoning_mode == "st":
                     tool_chain_output = reasoning_func(
                        question=question,
                        question_w_options=question_w_options,
                        options=options,
                        llm=tool_planner_llm,
                        tools=tool_instances,
                        recursion_limit=conf.recursion_limit,
                        use_cache=conf.use_cache,
                        mannual_cache=mannual_cache,
                        mannual_cache_file=mannual_cache_file,
                        eval_llm=eval_llm,
                        eval_cache=eval_cache,
                        eval_cache_file=eval_cache_file,
                    )
                else:
                    raise KeyError("conf.reasoning_mode error")
            
            if isinstance(tool_chain_output, str):
                result["answers"].append(tool_chain_output)
            elif isinstance(tool_chain_output, list):
                result["answers"].extend(tool_chain_output)
            else:
                raise ValueError("tool_chain_output error")
            
            visible_frames_all += len(visible_frames.frames)

        visible_frames_num = visible_frames_all / try_num
        result["video_info"] = video_info
        result["visible_frames_num"] = visible_frames_num
        result["visible_frames_fps"] =  visible_frames_num / duration
        all_results.append(result)

    output_file = os.path.join(conf.output_path, f"results_{timestamp}.json")
    save_to_json(all_results, output_file)
    print(f"\n{str(len(all_results))} results saved")   

    if conf.to_txt:
        sys.stdout = sys.__stdout__
        f.close()

    