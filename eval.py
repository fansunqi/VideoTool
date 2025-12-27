import os
import pdb
import json
import argparse
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
    elif len(options) == 3:
        option_labels = ['A', 'B', 'C']
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


def main(input_file, output_file, conf, eval_llm):

    with open(input_file, 'r') as f:
        data = json.load(f)

    total_items = len(data)
    have_ans_items = 0
    correct_items = 0
    error_items = 0
    total_visible_frames_num = 0
    total_visible_frames_fps = 0
    
    if conf.dataset == "videomme":
        short_total, short_have_ans, short_correct = 0, 0, 0
        medium_total, medium_have_ans, medium_correct = 0, 0, 0
        long_total, long_have_ans, long_correct = 0, 0, 0

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
            
        if conf.dataset == "videomme":
            uid = int(item["uid"])
            if uid <= 300:
                short_total += 1
                if predicted_options:
                    short_have_ans += 1
                if is_correct:
                    short_correct += 1
            elif uid <= 600:
                medium_total += 1
                if predicted_options:
                    medium_have_ans += 1
                if is_correct:
                    medium_correct += 1
            else:
                long_total += 1
                if predicted_options:
                    long_have_ans += 1
                if is_correct:
                    long_correct += 1
            
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
    
    if conf.dataset == "videomme":
        short_acc_include_no_ans = short_correct / short_total if short_total != 0 else 0.0
        medium_acc_include_no_ans = medium_correct / medium_total if medium_total != 0 else 0.0
        long_acc_include_no_ans = long_correct / long_total if long_total != 0 else 0.0
        short_acc_exclude_no_ans = short_correct / short_have_ans if short_have_ans != 0 else 0.0
        medium_acc_exclude_no_ans = medium_correct / medium_have_ans if medium_have_ans != 0 else 0.0
        long_acc_exclude_no_ans = long_correct / long_have_ans if long_have_ans != 0 else 0.0
        print(f"--short-- total: {short_total}; have_ans: {short_have_ans}, correct: {short_correct}; acc_include_no_ans: {short_acc_include_no_ans:.2f}; acc_exclude_no_ans: {short_acc_exclude_no_ans:.2f}")
        print(f"--medium-- total: {medium_total}; have_ans: {medium_have_ans}, correct: {medium_correct}; acc_include_no_ans: {medium_acc_include_no_ans:.2f}; acc_exclude_no_ans: {medium_acc_exclude_no_ans:.2f}")
        print(f"--long-- total: {long_total}; have_ans: {long_have_ans}, correct: {long_correct}; acc_include_no_ans: {long_acc_include_no_ans:.2f}; acc_exclude_no_ans: {long_acc_exclude_no_ans:.2f}")

    # 检查 output_file 所在的文件夹是否存在，如果不存在则创建
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存评估结果到文件
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate answers")
    parser.add_argument('--input_file', type=str, help="Path to the input JSON file")
    parser.add_argument('--output_file', type=str, help="Path to the output JSON file")
    parser.add_argument('--config', default="config/lvb.yaml",type=str)
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)

    dataset_name = conf.dataset
    if not args.input_file:
        args.input_file = get_latest_file(f'output/{dataset_name}')
    
    if not args.output_file:
        args.output_file = args.input_file.replace(f'output/{dataset_name}', f'eval/{dataset_name}')

    # LLM for rephrase
    print(f"\nInitializing eval-LLM for rephrase, model: {conf.EVAL_MODEL_NAME}")
    eval_llm = ChatOpenAI(model_string=conf.EVAL_MODEL_NAME, is_multimodal=False)

    main(args.input_file, args.output_file, conf, eval_llm)

    print(f"Output saved to {args.output_file}.")

