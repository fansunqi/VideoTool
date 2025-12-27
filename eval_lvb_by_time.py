import json
import cv2
from collections import defaultdict

def calculate_video_duration(video_path):
    """
    计算视频的时长（秒）
    :param video_path: 视频文件路径
    :return: 视频时长（秒）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if fps == 0:  # 防止除以零
        return 0
    return frame_count / fps

def classify_duration(duration):
    """
    根据视频时长分类
    :param duration: 视频时长（秒）
    :return: 分类标签
    """
    if 8 <= duration < 15:
        return "8s-15s"
    elif 15 <= duration < 60:
        return "15s-60s"
    elif 60 <= duration < 180:
        return "60s-180s"
    elif 180 <= duration < 600:
        return "180s-600s"
    elif 600 <= duration < 900:
        return "600s-900s"
    elif 900 <= duration < 3600:
        return "900s-3600s"
    else:
        return "其他"

def main(json_file):
    """
    主函数，读取 JSON 文件并统计正确率
    :param json_file: JSON 文件路径
    """
    # 读取 JSON 文件
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 初始化统计数据
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    # 遍历 JSON 数据
    for item in data:
        video_path = item.get("video_path")
        is_correct = item.get("is_correct", False)
        
        # 计算视频时长
        try:
            duration = calculate_video_duration(video_path)
        except Exception as e:
            print(f"无法处理视频 {video_path}: {e}")
            continue
        
        # 分类视频时长
        category = classify_duration(duration)
        
        # 更新统计数据
        stats[category]["total"] += 1
        if is_correct:
            stats[category]["correct"] += 1
    
    # 计算并打印正确率
    print("按时长范围统计正确率：")
    for category, values in stats.items():
        total = values["total"]
        correct = values["correct"]
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"{category}: 数量 {total} 正确率 {accuracy:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    # JSON 文件路径
    # json_file = "/home/fsq/video_agent/ToolChainVideo/eval/lvb/results_20250429_232827.json"
    json_file = "/home/fsq/video_agent/ToolChainVideo/eval/lvb/results_20250513_174854.json"
    
    # 调用主函数
    main(json_file)