import cv2
from PIL import Image
from typing import List
import torch
import re
import pdb
from transformers import (
    pipeline,
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model



def prompts(name, description):
    
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


def get_pil_image_list(visible_frames, target):
    pil_image_list = []
    relative_frame_idx_list = []  # 相对的 frame_idx
    frame_count = -1
    for visible_frame in visible_frames.frames:
        frame_count += 1
        
        # 如果目标是 visible_frames.qa_info 这个 dict 中的一个 key，就跳过
        if target in visible_frame.qa_info:
            continue

        rgb_image = cv2.cvtColor(visible_frame.image, cv2.COLOR_BGR2RGB)
        raw_image = Image.fromarray(rgb_image)
        pil_image_list.append(raw_image)
        relative_frame_idx_list.append(frame_count)
    return pil_image_list, relative_frame_idx_list
     
        
class ImageQA:
    def __init__(
        self,
        conf = None, 
    ):
        self.conf = conf
        self.device = conf.tool.image_qa.device
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
        
        self.model_path = conf.tool.image_qa.model_path
        print(f"Loading {self.model_path} for Image QA...\n")
        self.llava_tokenizer, self.llava_model, self.llava_image_processor, context_len = load_pretrained_model(
            model_path=self.model_path,
            model_base=None,
            model_name=get_model_name_from_path(self.model_path)
        )
        
        self.visible_frames = None
        self.video_path = None
        
        self.batch_size = conf.tool.image_qa.batch_size

    def set_frames(self, visible_frames):
        self.visible_frames = visible_frames
    
    def set_video_path(self, video_path):
        self.video_path = video_path  

    def image_qa(
        self,
        image,
        question
    ):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_image = Image.fromarray(rgb_image)
        
        args = type('Args', (), {
            "model_path": self.model_path,
            "tokenizer": self.llava_tokenizer,
            "model": self.llava_model,
            "image_processor": self.llava_image_processor,
            "query": question,
            "conv_mode": None,
            "input_pil_image": raw_image,
            "image_file": None,  # 传递处理后的图像
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        })()
        answer = eval_model(args)      
        return answer

    @prompts(
        name = "image-question-answering-tool",
        description = "Useful when you need to ask something about the frames in the video."
        "The input to this tool must be a question without options, such as 'How many children are in the video?', instead of 'How many children are in the video? A. 1 B. 2 C. 3 D. 4'."
    )
    def inference(self, input):
        # 问过同一个问题的帧就不加入
        pil_image_list, relative_frame_idx_list = get_pil_image_list(self.visible_frames, input)
        prompt_list = [input] * len(pil_image_list)
        
        print(f"\nImage QA: infer {str(len(pil_image_list))} frames...")

        outputs = []
        for i in range(0, len(prompt_list), self.batch_size):
            batch_prompts = prompt_list[i:i+self.batch_size]
            batch_images = pil_image_list[i:i+self.batch_size]
            
            args = type('Args', (), {
                "model_path": self.model_path, 
                "tokenizer": self.llava_tokenizer,
                "model": self.llava_model,
                "image_processor": self.llava_image_processor,
                "query": batch_prompts,
                "conv_mode": None,
                "input_pil_image": batch_images,
                "image_file": None,
                "sep": ",",
                "temperature": 0,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 512
            })()
            
            batch_outputs = eval_model(args)
            outputs.extend(batch_outputs)      
        
        result = "Here are the image question answering results of sampled frames:\n"
        result += f"Question: {input}\n"
        
        for relative_frame_idx, answer in zip(relative_frame_idx_list, outputs):
            
            frame = self.visible_frames.frames[relative_frame_idx]                 
            frame.qa_info[input] = answer
            result += f"Frame {frame.index} Answer: {answer}\n"

        print(result) 
        return result       


if __name__ == "__main__":
    # 加载预训练的处理器和模型
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载图像
    image_path = "/home/fsq/video_agent/ToolChainVideo/misc/car.jpg"
    image = Image.open(image_path).convert("RGB")

    # 定义问题
    question = "What is the color of the car?"
    
    
    # 处理输入
    inputs = processor(image, question, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # 推理
    output = model.generate(**inputs)

    # 解码答案
    answer = processor.decode(output[0], skip_special_tokens=True)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
