# Adapted from VideoTree
import os
import pdb
import json
import argparse
import pandas as pd
from pprint import pprint
from torch.utils.data import Dataset


char_to_num = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
}

def check_videomme_option(s):
    # 检查字符串长度是否大于等于 3，并且满足条件
    if len(s) >= 3 and s[0].isupper() and s[1] == '.' and s[2] == ' ' and \
        (s[-1] == '.' or s[-1] == '?' or s[-1] == '!'):
        return True
    return False


class BaseDataset(Dataset):
    def __init__(self, args, quids_to_exclude=None, num_examples_to_run=-1, start_num=0, specific_quids=None):
        '''
        num_examples_to_run < 0: run all
        '''
        self.args = args
        self.anno = self.get_anno()
        if num_examples_to_run > 0:
            self.end_num = start_num + num_examples_to_run
        else:
            self.end_num = len(self.anno)
        data = self.build()
        data = self.filter(data, quids_to_exclude, num_examples_to_run, start_num, specific_quids)
        self.data = data

    def set_ukey(self, name):
        self.ukey = name

    def filter(self, data, quids_to_exclude, num_examples_to_run, start_num, specific_quids):
        if start_num > 0:
            data = data[start_num:]
        if num_examples_to_run >= 0:
            data = data[:num_examples_to_run]
        if quids_to_exclude is not None:
            data = [el for el in data if el[self.ukey] not in quids_to_exclude]
        if specific_quids is not None:
            data = [el for el in data if el[self.ukey] in specific_quids]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class NextDataset(BaseDataset):
    def __init__(self, args, quids_to_exclude=None, num_examples_to_run=-1, start_num=0, specific_quids=None):
        self.set_ukey('quid')
        super().__init__(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run, start_num=start_num, specific_quids=specific_quids)

    def get_anno(self):
        return pd.read_csv(self.args.anno_path)  # video,frame_count,width,height,question,answer,qid,type,a0,a1,a2,a3,a4
  
    def build(self):
        print(f"\nBuilding {self.args.dataset} dataset...")
        data = []
        
        for row in self.anno.iterrows():
            if isinstance(row, tuple):
                row = row[-1]  # remove table index
            uid = str(row['video'])

            find_video = False
            for subdir in os.listdir(self.args.video_path_base):
                video_path = os.path.join(self.args.video_path_base, subdir, uid + ".mp4")
                if os.path.exists(video_path):
                    find_video = True
                    break
            
            if not find_video:
                continue

            question, truth = row['question'].capitalize(), row['answer']
            if question[-1] != '?':
                question += '?'
            qid, q_type = row['qid'], row['type']
            options = [row['a0'], row['a1'], row['a2'], row['a3'], row['a4']]
            quid = f'{uid}_{qid}'
            question_w_options = f"{question} Choose your answer from below options: A.{options[0]}, B.{options[1]}, C.{options[2]}, D.{options[3]}, E.{options[4]}."

            data.append({
                'quid': quid,
                'uid': uid,
                'qid': qid,
                'q_type': q_type,
                'question': question,
                'optionA': options[0],
                'optionB': options[1],
                'optionC': options[2],
                'optionD': options[3],
                'optionE': options[4],
                'options': options,
                'question_w_options': question_w_options,
                'truth': truth,
                'video_path': video_path,
            })
            
            if len(data) >= self.end_num:
                break
            
        return data


class VideoMMEDataset(BaseDataset):
    def __init__(self, args, quids_to_exclude=None, num_examples_to_run=-1, start_num=0, specific_quids=None):
        self.set_ukey('quid')
        super().__init__(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run, start_num=start_num, specific_quids=specific_quids)
    
    def get_anno(self):
        return pd.read_parquet(self.args.anno_path, engine='pyarrow')

    def build(self):
        print(f"\nBuilding {self.args.dataset} dataset...")
        data = []

        for row in self.anno.iterrows():
            if isinstance(row, tuple):
                row = row[-1]  # remove table index
            
            uid = row["video_id"]
            videoID = row['videoID']
            qid = row['question_id']
            q_type = row["task_type"]
            question = row['question'].capitalize()  # 首字母大写
            if question[-1] != '?':
                question += '?'
            
            options = row['options']
            new_options = []
            for option in options:
                if check_videomme_option(option):
                    new_option = option[3:-1]
                    new_options.append(new_option)

            truth = char_to_num[row['answer']]
            question_w_options = f"{question} Choose your answer from below options: A.{new_options[0]}, B.{new_options[1]}, C.{new_options[2]}, D.{new_options[3]}."

            video_path = os.path.join(self.args.video_path_base, videoID + ".mp4")
            if not os.path.exists(video_path):
                continue
            
            data.append({
                'quid': qid,
                'uid': uid,
                'qid': qid,
                'q_type': q_type,
                'question': question,
                'optionA': new_options[0],  # 去掉字母
                'optionB': new_options[1],
                'optionC': new_options[2],
                'optionD': new_options[3],
                'options': new_options,
                'question_w_options': question_w_options,
                'truth': truth,
                'video_path': video_path,
            })

            if len(data) >= self.end_num:
                break
               
        return data


class LVBDataset(BaseDataset):
    def __init__(self, args, quids_to_exclude=None, num_examples_to_run=-1, start_num=0, specific_quids=None):
        self.set_ukey('quid')
        super().__init__(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run, start_num=start_num, specific_quids=specific_quids)
    
    def get_anno(self):
        with open(self.args.anno_path, 'r') as f:
            return json.load(f)
    
    def build(self):
        print(f"\nBuilding {self.args.dataset} dataset...")
        data = []
        
        # 检查 video_id 是否重复  
        for data_item in self.anno:
            video_id = data_item['video_id']
            seen_video_ids = set()
            if video_id in seen_video_ids:
                print(f"Duplicate video_id found: {video_id}")
                raise ValueError(f"Duplicate video_id found: {video_id}")
            else:
                seen_video_ids.add(video_id)
        
        for data_item in self.anno:
            uid = data_item['video_id']
            question = data_item["question_wo_referring_query"].capitalize()
            
            # options = []
            # for candidate in data_item['candidates']:
            #     if candidate.endwith(','):
            #         options.append(candidate[:-1])
            #     else:
            #         options.append(candidate)
            
            options = data_item['candidates']
            
            if len(options) == 5:
                question_w_options = f"{question} Choose your answer from below options: A.{options[0]}, B.{options[1]}, C.{options[2]}, D.{options[3]}, E.{options[4]}."
            elif len(options) == 4:
                question_w_options = f"{question} Choose your answer from below options: A.{options[0]}, B.{options[1]}, C.{options[2]}, D.{options[3]}."
            elif len(options) == 3:
                question_w_options = f"{question} Choose your answer from below options: A.{options[0]}, B.{options[1]}, C.{options[2]}."
            else:
                raise ValueError(f"Invalid number of options: {len(options)}. Expected 3 or 4.")
                
            
            video_path = os.path.join(self.args.video_path_base, 'videos', data_item['video_path'])
            
            subtitle_file = data_item['subtitle_path']
            subtitle_path = os.path.join(self.args.video_path_base, 'subtitles', subtitle_file)
            
            truth = data_item["correct_choice"]
            
            q_type = data_item["question_category"]
            
            formatted_data_item = {
                'quid': uid,
                'uid': uid,
                'qid': uid,
                'q_type': q_type,
                'question': question,
                'optionA': options[0],
                'optionB': options[1],
                'optionC': options[2],
                'options': options,
                'question_w_options': question_w_options,
                'truth': truth,
                'video_path': video_path,
                'subtitle_path': subtitle_path,
            }
            
            if len(options) == 4:
                formatted_data_item['optionD'] = options[3]
            elif len(options) == 5:
                formatted_data_item['optionD'] = options[3]
                formatted_data_item['optionE'] = options[4]
            
            data.append(formatted_data_item)
            
            if len(data) >= self.end_num:
                break
        
        return data


def get_dataset(args, quids_to_exclude=None, num_examples_to_run=-1, start_num=0, specific_quids=None):
    if args.dataset == 'nextqa':
        return NextDataset(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run, start_num=start_num, specific_quids=specific_quids)
    elif args.dataset == 'videomme':
        return VideoMMEDataset(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run, start_num=start_num, specific_quids=specific_quids)
    elif args.dataset == 'lvb':  # longvideobench
        return LVBDataset(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run, start_num=start_num, specific_quids=specific_quids)
    else:
        raise ValueError(f"Dataset {args.dataset} not found")



def parse_args():
    parser = argparse.ArgumentParser(description="Dataset script")

    # parser.add_argument('--dataset', type=str, default="nextqa", help='Name of the dataset to use')
    # parser.add_argument('--video_path_base', type=str, default="/share_data/NExT-QA/NExTVideo")
    # parser.add_argument('--anno_path', type=str, default="/share_data/NExT-QA/dataset/nextqa/val.csv", help='Path to the annotation file')
    # parser.add_argument('--num_examples_to_run', type=int, default=100)

    # parser.add_argument('--dataset', type=str, default="videomme", help='Name of the dataset to use')
    # parser.add_argument('--video_path_base', type=str, default="/hf_home/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/data/")
    # parser.add_argument('--anno_path', type=str, default="/hf_home/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/videomme/test-00000-of-00001.parquet", help='Path to the annotation file')
    # parser.add_argument('--num_examples_to_run', type=int, default=100)
    
    parser.add_argument('--dataset', type=str, default="lvb", help='Name of the dataset to use')
    parser.add_argument('--video_path_base', type=str, default="/mnt/Shared_03/fsq/LongVideoBench")
    parser.add_argument('--anno_path', type=str, default="/mnt/Shared_03/fsq/LongVideoBench/lvb_val.json", help='Path to the annotation file')
    parser.add_argument('--num_examples_to_run', type=int, default=100)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = get_dataset(args, num_examples_to_run=args.num_examples_to_run)
    print(len(dataset))
    for data in dataset:
        pprint(data)
        pdb.set_trace()
