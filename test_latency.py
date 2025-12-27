import os
import time
# import openai
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

# from util import (
#     save_to_json,
#     adjust_video_resolution,
#     backup_file,
# )

from baselines.gpt4o import (
    extract_frames,
    video_qa,
)

from baselines.videollama3 import (
    video_qa
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demo")               
    parser.add_argument('--config', default="config/videomme.yaml",type=str)                           
    opt = parser.parse_args()
    conf = OmegaConf.load(opt.config)

    # 数据集
    quids_to_exclude = conf["quids_to_exclude"] if "quids_to_exclude" in conf else None
    num_examples_to_run = conf["num_examples_to_run"] if "num_examples_to_run" in conf else -1
    start_num = conf["start_num"] if "start_num" in conf else 0
    specific_quids = conf["specific_quids"] if "specific_quids" in conf else None
    dataset = get_dataset(conf, quids_to_exclude, num_examples_to_run, start_num, specific_quids)
    
    total_time = 0  # 累积处理时间
    count = 0       # 数据计数
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    print("api_key", api_key)
    print("base_url", base_url)
    
    # === 初始化 OpenAI 客户端 (新版 SDK) ===
    # client = openai.OpenAI(
    #     api_key=api_key,
    #     base_url=base_url
    # )

    for data in tqdm(dataset):
        start_time = time.time()  # 开始计时

        print(f"\n\nProcessing: {data['quid']}")

        video_path = data["video_path"]
        question = data["question"]
        options = data["options"]
        question_w_options = data["question_w_options"]


        # trim
        # adjust_video_resolution(video_path)
        
        print(video_path)
        print(question_w_options)
        # === 抽帧 ===
        # N = 384
        # N = 20
        # frames = extract_frames(video_path, N)

        # # === 问答 ===
        # answer = video_qa(client, frames, question_w_options)
        
        answer = video_qa(video_path=video_path, question=question_w_options, frames_num=180)

        print("\n=== 模型的回答 ===")
        print(answer)
        
        end_time = time.time()  # 结束计时
        elapsed_time = end_time - start_time
        total_time += elapsed_time  # 累加处理时间
        count += 1  # 增加计数
        print(f"Processing time for {data['quid']}: {elapsed_time:.2f} seconds")
    
    # 计算平均时间
    if count > 0:
        average_time = total_time / count
        print(f"\nAverage processing time: {average_time:.2f} seconds")
    else:
        print("\nNo data processed.")

        

    



    