import cv2
import numpy as np
from typing import List
from engine.openai import ChatOpenAI
from prompts import SELECT_FRAMES_PROMPT
from pydantic import BaseModel, Field
from pprint import pprint
import pdb

def prompts(name, description):
    
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator



class SegmentList(BaseModel):
    analysis: str
    segments: List[int]


class FrameSelector:
    def __init__(self, conf):

        self.conf = conf
        
        model_string = conf.tool.frame_selector.llm_model_name
        print(f"\nInitializing Frame Selection Tool with model: {model_string}")
        self.llm = ChatOpenAI(model_string=model_string, is_multimodal=False)    

        self.visible_frames = None
        self.video_path = None
    
    def set_video_path(self, video_path):
        self.video_path = video_path  
        
    def set_frames(self, visible_frames):
        self.visible_frames = visible_frames
    

    @prompts(
        name = "frame-extraction-tool",
        description = "Useful when you find that the currently sampled frames do not provide enough information and more frames need to be extracted from the video to answer the question."
        "The input to this tool must be a question about the video that remains unresolved. For example, 'How many children are in the video? Choose your answer from below selections: A.one, B.three, C.seven, D.two, E.five.'",
    )
    def inference(self, input):
        
        invisible_segments_list = self.visible_frames.get_invisible_segments()

        # 检查是否还有分割的余地
        if len(invisible_segments_list) == 0:
            return_message = "All video segments are already shorter than the minimum interval, so no more frames can be extracted from them."
            print(f"\nFrame Selector: {return_message}")
            return return_message
        
        select_frames_prompt = SELECT_FRAMES_PROMPT.format(
            num_frames = self.visible_frames.video_info["total_frames"],
            fps = self.visible_frames.video_info["fps"],
            visible_frames_info = self.visible_frames.get_qa_descriptions(),
            question = input,
            candidate_segment = self.visible_frames.invisible_segments_to_description(),
            max_candidate_segment_id = str(len(invisible_segments_list) - 1)
        )

        response = self.llm(select_frames_prompt, response_format=SegmentList)
        
        add_frames_indices_all = []
        min_interval = self.visible_frames.min_interval
        for segment_id in response.segments:
            segment = invisible_segments_list[segment_id]
            start_frame_idx = segment[0]
            end_frame_idx = segment[1]
            if end_frame_idx - start_frame_idx > min_interval:
                add_frames_indices = range(start_frame_idx, end_frame_idx, min_interval)
                # print("add_frames_indices: ", add_frames_indices)
                add_frames_indices_all.extend(add_frames_indices)

        add_frames_indices_all = list(set(add_frames_indices_all))
        add_frames_indices_all.sort()
        print(f"\nFrame Selector: analysis: {response.analysis}")
        print(f"Frame Selector: add_frames_indices_all: {str(add_frames_indices_all)}\n")


        self.visible_frames.add_frames(frame_indices=add_frames_indices_all)

        return_message = "A series of potentially relevant frames have been successfully extracted and added to the visible frame set. Please continue by using other tools to analyze these newly added frames."
        return return_message



        


        








    