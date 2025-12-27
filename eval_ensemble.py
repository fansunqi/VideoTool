import os
import pdb
import json
import argparse
import datetime
from tqdm import tqdm
from collections import Counter
from omegaconf import OmegaConf
from engine.openai import ChatOpenAI
from prompts import EVAL_PROMPT


def option_full_matching(answer, options):
    answer = answer.lower()
    options = [option.lower() for option in options]
    # 完整匹配（确保只有一个选项完整出现在 answer 中）
    for i, option in enumerate(options):
        if option in answer and all(opt not in answer for j, opt in enumerate(options) if j != i):
            return i, "option full matching"
    return -1, "none"

def answer_full_matching(answer, options):
    answer = answer.lower()
    options = [option.lower() for option in options]
    # 完整匹配（确保 answer 完整出现在一个选项中）
    for i, option in enumerate(options):
        if answer in option and all(answer not in opt for j, opt in enumerate(options) if j != i):
            return i, "answer full matching"
    return -1, "none"

def LLM_rephrase(answer, options, question, conf, eval_llm):
    
    # 首先构造选项的提示
    if len(options) == 5:
        option_labels = ['A', 'B', 'C', 'D', 'E']
    elif len(options) == 4:
        option_labels = ['A', 'B', 'C', 'D']
    else:
        raise ValueError("options in the data error")
    
    options_with_labels = "\n".join([f"{label}: {option}" for label, option in zip(option_labels, options)])
    
    # 创建 prompt 给 LLM
    prompt = EVAL_PROMPT.format(
        question=question,
        answer=answer,
        options_with_labels=options_with_labels
    )

    answer_rephrase = eval_llm(prompt)
    
    return answer_rephrase
    
    
def get_predicted_option(answer, options):
    """根据答案匹配正确选项"""
    
    predicted_option, match_method = option_full_matching(answer, options)
    if predicted_option != -1:
        return predicted_option, match_method
    
    predicted_option, match_method = answer_full_matching(answer, options)
    if predicted_option != -1:
        return predicted_option, match_method
    
    return -1, "none"


def get_predicted_option_with_rephrase(answer, options, question, conf, eval_llm):
    predicted_option, match_method = get_predicted_option(answer, options)
    if predicted_option == -1:
        answer_rephrase = LLM_rephrase(answer, options, question, conf, eval_llm)
        predicted_option, match_method = get_predicted_option(answer_rephrase, options)
    return predicted_option, match_method


def get_latest_file(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if 
             (os.path.isfile(os.path.join(directory, f))
              and f.endswith('.json'))]
    latest_file = sorted(files)[-1] 
    return latest_file


def main(data, output_file, conf, eval_llm):

    total_items = len(data)
    have_ans_items = 0
    correct_items = 0
    error_items = 0
    total_visible_frames_num = 0
    total_visible_frames_fps = 0

    for item in tqdm(data):
        truth = item['truth']
        options = item['options']
        question = item['question']
        
        if not isinstance(item['answers'], list) or len(item['answers']) == 0:
            error_items += 1
            continue
        
        if all(answer == "Error" for answer in item['answers']):
            error_items += 1
            continue

        answers = item['answers']
        
        predicted_options = []
        match_methods = []
        for answer in answers:
            predicted_option, match_method = get_predicted_option_with_rephrase(
                answer, options, question, conf, eval_llm
            )
            predicted_options.append(predicted_option)
            match_methods.append(match_method)
            
        item["predicted_options"] = predicted_options
        item["match_methods"] = match_methods
        
        # 去除 -1, 即判断不出来的回答
        predicted_options = [option for option in predicted_options if option != -1]
        
        # 投票确定最终预测答案, 多个选项个数一样, 随机选一个
        if predicted_options:
            option_counts = Counter(predicted_options)
            most_common_option, _ = option_counts.most_common(1)[0]
            final_predicted_option = most_common_option
            have_ans_items += 1
        else:
            final_predicted_option = None
        
        is_correct = (final_predicted_option == truth)
        
        if is_correct:
            correct_items += 1
        
        item['final_predicted_option'] = final_predicted_option
        item['is_correct'] = is_correct
        item['match_methods'] = match_methods

        if "visible_frames_num" in item:
            total_visible_frames_num += item["visible_frames_num"]
        if "visible_frames_fps" in item:
            total_visible_frames_fps += item["visible_frames_fps"]

    acc_include_no_ans = correct_items / total_items
    acc_exclude_no_ans = correct_items / have_ans_items
    avg_visible_frames_num = total_visible_frames_num / total_items
    avg_visible_frames_fps = total_visible_frames_fps / total_items

    # 输出结果
    print(f"Total items: {total_items}")
    print(f"Not Error items: {total_items - error_items}")
    print(f"Have ans items: {have_ans_items}")
    print(f"Correct items: {correct_items}")
    print(f"Acc include no ans: {acc_include_no_ans:.2%}")
    print(f"Acc exclude no ans: {acc_exclude_no_ans:.2%}")
    print(f"avg_visible_frames_num: {avg_visible_frames_num:.2f}")
    print(f"avg_visible_frames_fps: {avg_visible_frames_fps:.2f}")

    # 检查 output_file 所在的文件夹是否存在，如果不存在则创建
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存评估结果到文件
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate answers")
    parser.add_argument('--config', default="config/lvb.yaml",type=str)
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)

    input_file_list = [
        # short
        # "/home/fsq/video_agent/ToolChainVideo/output/videomme/results_20250424_121706.json",   # 1 TQA
        # "/home/fsq/video_agent/ToolChainVideo/output/videomme/results_20250424_225733.json",   # 5.1 1fps+IGQA
        # "/home/fsq/video_agent/ToolChainVideo/output/videomme/results_20250424_192740.json",   # 4.1 1fps+IQA+S
        # "/home/fsq/video_agent/ToolChainVideo/output/videomme/results_20250425_174141.json",   # 2.4 VideoQA (InternVL3-VL-2B)
        # "/home/fsq/video_agent/ToolChainVideo/output/videomme/results_20250425_193249.json"    # 2.2 VideoQA (Qwen2.5-VL-3B)
        
        # medium
        # "/home/fsq/video_agent/ToolChainVideo/output/videomme/results_20250426_121708.json",   # 1.2 VideoQA (Qwen2.5-VL-3B)
        # "/home/fsq/video_agent/ToolChainVideo/output/videomme/results_20250426_134825.json",   # 1.4 VideoQA (InternVL3-2B)
        # "/home/fsq/video_agent/ToolChainVideo/output/videomme/results_20250426_194607.json",   # 2.2 16+IGQA
        # "/home/fsq/video_agent/ToolChainVideo/output/videomme/results_20250426_193722.json",   # 3.2 16+IQA+S

        
        # long
        # "/home/fsq/video_agent/ToolChainVideo/output/videomme/results_20250426_160749.json",   # VideoQA (Qwen2.5-VL-3B)
        # "/home/fsq/video_agent/ToolChainVideo/output/videomme/results_20250426_155926.json",   # VideoQA (InternVL3-2B)
        # "/home/fsq/video_agent/ToolChainVideo/output/videomme/results_20250426_202739.json",   # 16+IGQA
        # "/home/fsq/video_agent/ToolChainVideo/output/videomme/results_20250427_113813.json",   # 16+ICL+S
        
        
        # lvb-sample-100
        "/home/fsq/video_agent/ToolChainVideo/eval/lvb/results_20250429_165807.json",
        "/home/fsq/video_agent/ToolChainVideo/eval/lvb/results_20250428_232733.json",
        "/home/fsq/video_agent/ToolChainVideo/eval/lvb/results_20250429_002333.json"
    ]
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"eval/videomme/ensemble_{timestamp}.json"

    with open(input_file_list[0], 'r') as f:
        data = json.load(f)
    data_len = len(data)
    for i in range(1, len(input_file_list)):
        extend_file = input_file_list[i]
        with open(extend_file, 'r') as f:
            extend_data = json.load(f)
        assert len(extend_data) == data_len
        for j in range(data_len):
            data[j]['answers'].extend(extend_data[j]['answers'])

    # LLM for rephrase
    eval_llm = ChatOpenAI(model_string=conf.EVAL_MODEL_NAME, is_multimodal=False)

    main(data, output_file, conf, eval_llm)

    print(f"Output saved to {output_file}.")

