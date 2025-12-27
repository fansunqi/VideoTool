import cv2
import json
import torch
from typing import List, Optional
from dataclasses import dataclass, field


def get_video_info(video_path):
    """
    获取视频的基本信息，包括总帧数、帧率和时长
    
    参数:
        video_path: 视频文件路径
    
    返回:
        dict: 包含视频信息的字典
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    # 释放视频捕获对象
    cap.release()
    
    return {
        "total_frames": total_frames,
        "fps": fps,
        "duration": duration  # 单位：秒
    }


@dataclass
class Frame:
    """表示视频中的一帧"""
    index: int  # 帧在视频中的索引
    timestamp: float  # 帧的时间戳（秒）
    image: cv2.Mat = None  # 帧的图像数据
    description: Optional[str] = ""  # 帧的文字描述
    qa_info: dict = field(default_factory=dict)  # 帧的问答信息


class VisibleFrames:
    """管理一个视频中的可见帧"""
    def __init__(self, 
                video_path, 
                init_sec_interval=None,
                init_interval_num=None,
                min_interval=None,
                min_sec_interval=1,
                subtitle_path=None):
        self.frames: List[Frame] = [] 
        self.video_path = video_path                
        self.video_info = get_video_info(video_path)
        if init_sec_interval != None:
            fps = self.video_info["fps"]
            init_interval = int(init_sec_interval * fps)
            self.add_frames(video_stride=init_interval)
        elif init_interval_num != None:
            total_frames = self.video_info["total_frames"]
            if total_frames <= init_interval_num:
                frame_indices = list(range(total_frames))
            else:
                step = (total_frames - 1) / (init_interval_num - 1)
                frame_indices = [round(i * step) for i in range(init_interval_num)]
            self.add_frames(frame_indices=frame_indices)

        if min_interval:
            self.min_interval = min_interval
        else:
            self.min_interval = int(min_sec_interval * self.video_info['fps'])
        
        if subtitle_path:
            with open(subtitle_path, 'r', encoding='utf-8') as f:
                self.subtitles = json.load(f)
        

    def add_frames(self, 
                   frame_indices=None, 
                   video_stride=None):
        """添加新的可见帧"""

        # 确定要读取的帧索引
        total_frames = self.video_info["total_frames"]
        if frame_indices is None:
            if video_stride is None or video_stride <= 0:
                # 如果没有指定stride或stride无效，则读取所有帧
                frame_indices = list(range(total_frames))
            else:
                # 根据stride抽帧
                frame_indices = list(range(0, total_frames, video_stride))

        print(f"\nVisible Frames: add {len(list(frame_indices))} frames to visible frames: {str(list(frame_indices))}")
        
        # 每次都是重新读取视频文件
        cap = cv2.VideoCapture(self.video_path)

        for frame_idx in frame_indices:
            # 检查帧索引是否有效
            if frame_idx < 0 or frame_idx >= total_frames:
                print(f"Warning: Frame index {frame_idx} is out of range [0, {total_frames-1}]")
                continue
            
            # 检查帧是否已存在
            if any(frame.index == frame_idx for frame in self.frames):
                print(f"Frame {frame_idx} already exists, skipping...")
                continue

            # 设置要读取的帧位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # 读取指定帧
            ret, frame = cap.read()
            
            # 检查是否成功读取帧
            if ret:
                timestamp = frame_idx / self.video_info['fps']
                self.frames.append(Frame(
                    index=frame_idx,
                    timestamp=timestamp,
                    image=frame
                ))
            else:
                print(f"Warning: Could not read frame {frame_idx}")

        self.frames.sort(key=lambda x: x.timestamp)
    
    def remove_all_frames(self):
        print("!! remove all frames")
        self.frames = []
        

    def get_images_rgb_tchw(self, frames_num) -> torch.Tensor:
        """获取所有可见帧的图像数据
        
        返回:
            (T, C, H, W), torch.uint8, 0-255, RGB
        """
        all_images = []
        cur_frames_num = len(self.frames)

        if cur_frames_num < frames_num:
            duplicate_num = int(frames_num // cur_frames_num)
            padding_num = frames_num - cur_frames_num * duplicate_num
            for frame in self.frames:
                # TODO 是否转为 RGB 格式
                image_rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
                for i in range(duplicate_num):
                    all_images.append(image_rgb)
            for i in range(padding_num):
                image_rgb = cv2.cvtColor(self.frames[-1].image, cv2.COLOR_BGR2RGB)
                all_images.append(image_rgb)
        else:
            # 计算需要跳过的帧数
            step = int(cur_frames_num // frames_num)
            for i in range(frames_num):
                # 计算当前帧的索引
                frame_idx = int(i * step)
                # 获取对应帧并转换为RGB格式
                image_rgb = cv2.cvtColor(self.frames[frame_idx].image, cv2.COLOR_BGR2RGB)
                all_images.append(image_rgb)
        
        all_images = torch.stack([torch.from_numpy(img).permute(2, 0, 1) for img in all_images])
        return all_images
    
    def get_qa_descriptions(self) -> str:
        """获取所有可见帧的问答信息"""
        if not self.frames:
            return "No visible frames."
        
        qa_info_all = {}
        for frame in self.frames:
            for q, a in frame.qa_info.items():
                if q in qa_info_all:
                    qa_info_all[q][frame.index] = a
                else:
                    qa_info_all[q] = {frame.index: a}
        
        if len(qa_info_all) == 0:
            return "No QA information available."

        result = "Here are the image question answering results of sampled frames:\n"
        for q, many_a in qa_info_all.items():
            result += f"Question: {q}\n"
            for f_idx, a in many_a.items():
                result += f"Frame {f_idx} Answer: {a}\n"  
        
        return result
    
    def get_frame_descriptions(self) -> str:
        """获取所有可见帧的文字描述"""
        if not self.frames:
            return "No visible frames."
        
        descriptions = []
        for frame in self.frames:
            desc = f"Frame {frame.index}"
            if frame.description:
                desc += f":\n{frame.description}"
            descriptions.append(desc)
        
        return "\n".join(descriptions)
    
    def get_frame_count(self) -> int:
        """获取可见帧数量"""
        return len(self.frames)
    
    def get_time_range(self) -> tuple:
        """获取可见帧的时间范围"""
        if not self.frames:
            return (0, 0)
        return (self.frames[0].timestamp, self.frames[-1].timestamp)
    
    def get_frame_at_time(self, timestamp: float) -> Optional[Frame]:
        """获取指定时间点最接近的帧"""
        if not self.frames:
            return None
        return min(self.frames, key=lambda x: abs(x.timestamp - timestamp))
    
    def get_frame_indices(self) -> tuple:
        """获取所有可见帧的索引集合
        
        返回:
            set: 所有可见帧的索引集合
        """
        if not self.frames:
            return set()
        return set(sorted(frame.index for frame in self.frames))

    def get_invisible_segments(self) -> List[tuple]:
        if not self.frames or not self.video_info:
            return []
        
        # 获取所有可见帧的索引并排序
        visible_indices = sorted(frame.index for frame in self.frames)
        total_frames = self.video_info['total_frames']

        invisible_segments = []
        
        # 1. 检查视频开始到第一个可见帧之间的片段
        if visible_indices[0] - 0 > self.min_interval:
            invisible_segments.append((0, visible_indices[0]))
        
        # 2. 检查相邻可见帧之间的片段
        for i in range(len(visible_indices) - 1):
            current_idx = visible_indices[i]
            next_idx = visible_indices[i + 1]
            if next_idx - current_idx > self.min_interval:  # 如果相邻可见帧之间有间隔
                invisible_segments.append((current_idx + 1, next_idx))
        
        # 3. 检查最后一个可见帧到视频结束之间的片段
        if visible_indices[-1] < total_frames - self.min_interval:
            invisible_segments.append((visible_indices[-1] + 1, total_frames))
        
        return invisible_segments

    def invisible_segments_to_description(self):
        invisible_segments = self.get_invisible_segments()
        segments_description = ""
        # 从 1 开始进行枚举
        for i, (start, end) in enumerate(invisible_segments):
            segments_description += f"{i}: {start}-{end}\n"
        return segments_description


if __name__ == "__main__":
    video_path = "/share_data/NExT-QA/NExTVideo/1164/3238737531.mp4"
    video_stride = 30  # 设置抽帧间隔
    
    # 创建可见帧管理器
    visible_frames = VisibleFrames(video_path=video_path, init_video_stride=video_stride)
    
    # 打印视频信息
    print(visible_frames.video_info)
    
    # 打印可见帧信息
    print(f"可见帧数量: {visible_frames.get_frame_count()}")
    start_idx, end_idx = visible_frames.get_frame_indices()
    print(f"帧索引范围: {start_idx} - {end_idx}")
    print("\n可见帧描述:")
    print(visible_frames.get_frame_descriptions())

    segments_dict = visible_frames.invisible_segments_to_description()
    print(segments_dict)


    