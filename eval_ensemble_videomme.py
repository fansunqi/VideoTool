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

def LLM_rephrase(answer, options, question, eval_llm):
    
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
    
    # if question == "According to the video, which of the following is the main reason why people commemorate qu yuan?":
    #     pdb.set_trace()

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


def get_predicted_option_with_rephrase(answer, options, question, eval_llm):
    predicted_option, match_method = get_predicted_option(answer, options)
    if predicted_option == -1:
        answer_rephrase = LLM_rephrase(answer, options, question, eval_llm)
        predicted_option, match_method = get_predicted_option(answer_rephrase, options)
    return predicted_option, match_method


def get_latest_file(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if 
             (os.path.isfile(os.path.join(directory, f))
              and f.endswith('.json'))]
    latest_file = sorted(files)[-1] 
    return latest_file


def get_final_prediction(answers, options, question, eval_llm):
    
    predicted_options = []
    match_methods = []
    for answer in answers:
        predicted_option, match_method = get_predicted_option_with_rephrase(
            answer, options, question, eval_llm
        )
        predicted_options.append(predicted_option)
        match_methods.append(match_method)
    
    # 去除 -1, 即判断不出来的回答
    predicted_options = [option for option in predicted_options if option != -1]
    
    # 投票确定最终预测答案, 多个选项个数一样, 随机选一个
    if predicted_options:
        option_counts = Counter(predicted_options)
        most_common_option, _ = option_counts.most_common(1)[0]
        final_predicted_option = most_common_option
    else:
        final_predicted_option = None
    
    return final_predicted_option
    
    
    
def main(qwen_result, IGQA_result, output_file, eval_llm):

    total_items = len(qwen_result)
    have_ans_items = 0
    correct_items = 0
    error_items = 0
    total_visible_frames_num = 0
    total_visible_frames_fps = 0

    short_total, short_have_ans, short_correct = 0, 0, 0
    medium_total, medium_have_ans, medium_correct = 0, 0, 0
    long_total, long_have_ans, long_correct = 0, 0, 0
    
    for qwen_item, IGQA_item in zip(qwen_result, IGQA_result):
        
        uid = int(qwen_item["uid"])
        truth = qwen_item['truth']
        options = qwen_item['options']
        question = qwen_item['question']
        
        qwen_answers = qwen_item['answers']
        IGQA_answers = IGQA_item['answers']
        
        if uid <= 300:
            # short
            short_total += 1
            
            # qwen
            final_prediction = get_final_prediction(qwen_answers, options, question, eval_llm)
            
            # IGQA
            if final_prediction == None:
                final_prediction = get_final_prediction(IGQA_answers, options, question, eval_llm)
            
            # random  
            if final_prediction:
                short_have_ans += 1
            else:
                final_prediction = 0
            
            is_correct = (final_prediction == truth)
            if is_correct:
                short_correct += 1
        
        elif uid <= 600:
            # medium
            medium_total += 1
            
            # IGQA 
            final_prediction = get_final_prediction(IGQA_answers, options, question, eval_llm)
            
            if final_prediction == None:
                final_prediction = get_final_prediction(qwen_answers, options, question, eval_llm)
            
            # random
            if final_prediction:
                medium_have_ans += 1
            else:
                final_prediction = 0        
            
            is_correct = (final_prediction == truth)
            if is_correct:
                medium_correct += 1
                
        else:
            # long
            long_total += 1
            
            # IGQA 
            final_prediction = get_final_prediction(IGQA_answers, options, question, eval_llm)
            
            if final_prediction == None:
                final_prediction = get_final_prediction(qwen_answers, options, question, eval_llm)
            
            # random
            if final_prediction:
                long_have_ans += 1
            else:
                final_prediction = 0
                
            is_correct = (final_prediction == truth)
            if is_correct:
                long_correct += 1
        
        
        
        if is_correct:
            correct_items += 1      
        have_ans_items += 1
        
        
    acc_include_no_ans = correct_items / total_items
    acc_exclude_no_ans = correct_items / have_ans_items
    # avg_visible_frames_num = total_visible_frames_num / total_items
    # avg_visible_frames_fps = total_visible_frames_fps / total_items

    # 输出结果
    print(f"Total items: {total_items}")
    print(f"Not Error items: {total_items - error_items}")
    print(f"Have ans items: {have_ans_items}")
    print(f"Correct items: {correct_items}")
    print(f"Acc include no ans: {acc_include_no_ans:.2%}")
    print(f"Acc exclude no ans: {acc_exclude_no_ans:.2%}")
    # print(f"avg_visible_frames_num: {avg_visible_frames_num:.2f}")
    # print(f"avg_visible_frames_fps: {avg_visible_frames_fps:.2f}")  
    
    
    short_acc_include_no_ans = short_correct / short_total if short_total != 0 else 0.0
    medium_acc_include_no_ans = medium_correct / medium_total if medium_total != 0 else 0.0
    long_acc_include_no_ans = long_correct / long_total if long_total != 0 else 0.0
    short_acc_exclude_no_ans = short_correct / short_have_ans if short_have_ans != 0 else 0.0
    medium_acc_exclude_no_ans = medium_correct / medium_have_ans if medium_have_ans != 0 else 0.0
    long_acc_exclude_no_ans = long_correct / long_have_ans if long_have_ans != 0 else 0.0
    print(f"--short-- total: {short_total}; have_ans: {short_have_ans}, correct: {short_correct}; acc_include_no_ans: {short_acc_include_no_ans:.2%}; acc_exclude_no_ans: {short_acc_exclude_no_ans:.2%}")
    print(f"--medium-- total: {medium_total}; have_ans: {medium_have_ans}, correct: {medium_correct}; acc_include_no_ans: {medium_acc_include_no_ans:.2%}; acc_exclude_no_ans: {medium_acc_exclude_no_ans:.2%}")
    print(f"--long-- total: {long_total}; have_ans: {long_have_ans}, correct: {long_correct}; acc_include_no_ans: {long_acc_include_no_ans:.2%}; acc_exclude_no_ans: {long_acc_exclude_no_ans:.2%}")  
                
            
            
        
    
    

if __name__ == "__main__":

    qwen_result_file = "/home/fsq/video_agent/ToolChainVideo/output/videomme/results_20250503_231502.json"
    IGQA_result_file = "/home/fsq/video_agent/ToolChainVideo/output/videomme/results_20250504_223532.json"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"eval/videomme/ensemble_{timestamp}.json"

    with open(qwen_result_file, 'r') as f:
        qwen_result = json.load(f)
        
    with open(IGQA_result_file, 'r') as f:
        IGQA_result = json.load(f)


    # LLM for rephrase
    eval_llm = ChatOpenAI(model_string="gpt-4o-mini", is_multimodal=False)

    main(qwen_result, IGQA_result, output_file, eval_llm)

    print(f"Output saved to {output_file}.")

