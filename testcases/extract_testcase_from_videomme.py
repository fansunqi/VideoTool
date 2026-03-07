"""
从 VideoMME 数据集中提取一个测例（视频 + 问题 + 选项 + 答案），
放置到 testcases/ 目录下。

用法:
    python testcases/extract_testcase_from_videomme.py

输出:
    testcases/testcase.json   — 测例的文本信息
    testcases/<videoID>.mp4   — 配套视频文件（符号链接）
"""

import os
import json
import shutil
import pandas as pd

# ── 路径配置 ──────────────────────────────────────────────
PARQUET_PATH = "/mnt/Shared_05_disk/fsq/VideoMME/test-00000-of-00001.parquet"
VIDEO_DIR = "/mnt/Shared_05_disk/fsq/VideoMME/data"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# ── 选取条件：优先选最小的 short 视频 ─────────────────────
TARGET_VIDEO_ID = "004"       # 3.6 MB，human evolution 主题
TARGET_QUESTION_ID = "004-1"  # 第一个问题


def check_videomme_option(s: str) -> bool:
    """检查选项格式是否为 'X. xxxxx.' """
    if len(s) >= 3 and s[0].isupper() and s[1] == '.' and s[2] == ' ':
        return True
    return False


def strip_option_prefix(option: str) -> str:
    """去掉 'A. ' 前缀和末尾句号: 'A. Apples.' -> 'Apples' """
    if check_videomme_option(option):
        body = option[3:]           # 去掉 'A. '
        if body and body[-1] in '.?!':
            body = body[:-1]        # 去掉末尾标点
        return body
    return option


def main():
    # 1. 读取 parquet
    df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")

    # 2. 筛选目标行
    row = df[df["question_id"] == TARGET_QUESTION_ID]
    if row.empty:
        raise ValueError(f"未找到 question_id={TARGET_QUESTION_ID}")
    row = row.iloc[0]

    video_id = row["video_id"]
    video_youtube_id = row["videoID"]
    question = row["question"].capitalize()
    if question and question[-1] != "?":
        question += "?"

    raw_options = list(row["options"])  # numpy array -> list
    options_clean = [strip_option_prefix(o) for o in raw_options]

    answer_letter = row["answer"]         # e.g. 'B'
    task_type = row["task_type"]
    duration = row["duration"]
    domain = row["domain"]
    sub_category = row["sub_category"]

    # 构造 question_w_options（与 dataset.py 中 VideoMMEDataset 一致）
    question_w_options = (
        f"{question} Choose your answer from below options: "
        f"A.{options_clean[0]}, B.{options_clean[1]}, "
        f"C.{options_clean[2]}, D.{options_clean[3]}."
    )

    # 3. 复制 / 链接视频到 testcases/
    src_video = os.path.join(VIDEO_DIR, f"{video_youtube_id}.mp4")
    dst_video = os.path.join(OUTPUT_DIR, f"{video_youtube_id}.mp4")

    if not os.path.exists(src_video):
        raise FileNotFoundError(f"视频文件不存在: {src_video}")

    if os.path.exists(dst_video):
        os.remove(dst_video)
    # 使用符号链接，避免浪费磁盘空间
    os.symlink(src_video, dst_video)
    print(f"视频链接: {dst_video} -> {src_video}")

    # 4. 构造测例 JSON
    testcase = {
        "video_id": video_id,
        "videoID": video_youtube_id,
        "question_id": TARGET_QUESTION_ID,
        "task_type": task_type,
        "duration": duration,
        "domain": domain,
        "sub_category": sub_category,
        "question": question,
        "options_raw": raw_options,
        "options": options_clean,
        "answer": answer_letter,
        "question_w_options": question_w_options,
        "video_path": dst_video,
    }

    # 5. 保存 JSON
    json_path = os.path.join(OUTPUT_DIR, "testcase.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(testcase, f, indent=2, ensure_ascii=False)
    print(f"测例已保存: {json_path}")

    # 6. 打印摘要
    print("\n" + "=" * 60)
    print(f"  Video ID  : {video_id} ({video_youtube_id})")
    print(f"  Duration  : {duration}")
    print(f"  Domain    : {domain} / {sub_category}")
    print(f"  Task Type : {task_type}")
    print(f"  Question  : {question}")
    for i, (raw, clean) in enumerate(zip(raw_options, options_clean)):
        marker = " ✓" if chr(ord("A") + i) == answer_letter else ""
        print(f"  Option {chr(ord('A') + i)} : {clean}{marker}")
    print(f"  Answer    : {answer_letter}")
    print("=" * 60)


if __name__ == "__main__":
    main()
