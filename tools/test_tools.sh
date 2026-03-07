#!/bin/bash
# 工具单元测试脚本
# 用法: cd /home/fsq/video_agent/VideoTool && bash tools/test_tools.sh [tool_name]
# 如果不指定 tool_name，则列出所有可测试的工具
# 所有测试使用 testcases/testcase.json + testcases/HwnB8aCn8yE.mp4 作为测试数据

set -e

TOOL=${1:-""}

run_test() {
    local name=$1
    local module=$2
    echo ""
    echo "============================================================"
    echo "Testing: $name"
    echo "============================================================"
    python -m $module
    echo ""
    echo "$name test done."
    echo "============================================================"
}

# ---- 无需 GPU / 轻量级工具 ----
if [ "$TOOL" == "frame_selector" ] || [ "$TOOL" == "all_light" ]; then
    run_test "FrameSelector" "tools.frame_selector"
fi

if [ "$TOOL" == "image_grid_qa" ] || [ "$TOOL" == "all_light" ]; then
    run_test "ImageGridQA" "tools.image_grid_qa"
fi

if [ "$TOOL" == "image_grid_select" ] || [ "$TOOL" == "all_light" ]; then
    run_test "ImageGridSelect" "tools.image_grid_select"
fi

if [ "$TOOL" == "patch_zoomer" ] || [ "$TOOL" == "all_light" ]; then
    run_test "PatchZoomer" "tools.patch_zoomer"
fi

if [ "$TOOL" == "summarizer" ] || [ "$TOOL" == "all_light" ]; then
    run_test "Summarizer" "tools.summarizer"
fi

# ---- 需要 GPU / 重量级模型工具 ----
if [ "$TOOL" == "image_captioner" ]; then
    run_test "ImageCaptioner (BLIP)" "tools.image_captioner"
fi

if [ "$TOOL" == "image_captioner_llava" ]; then
    run_test "ImageCaptionerLLaVA" "tools.image_captioner_llava"
fi

if [ "$TOOL" == "image_qa" ]; then
    run_test "ImageQA (LLaVA)" "tools.image_qa"
fi

if [ "$TOOL" == "yolo_tracker" ]; then
    run_test "YOLOTracker" "tools.yolo_tracker"
fi

# ---- 需要 Grounded-Video-LLM 权重 ----
if [ "$TOOL" == "temporal_grounding" ]; then
    run_test "TemporalGrounding" "tools.temporal_grounding"
fi

if [ "$TOOL" == "temporal_qa" ]; then
    run_test "TemporalQA" "tools.temporal_qa"
fi

if [ "$TOOL" == "temporal_referring" ]; then
    run_test "TemporalReferring" "tools.temporal_referring"
fi

# ---- 需要 Qwen / InternVL 模型 ----
if [ "$TOOL" == "video_qa" ]; then
    run_test "VideoQA (Qwen2.5-VL)" "tools.video_qa"
fi

if [ "$TOOL" == "video_qa_internvl" ]; then
    run_test "VideoQAInternVL" "tools.video_qa_internvl"
fi

# ---- 帮助信息 ----
if [ "$TOOL" == "" ]; then
    echo ""
    echo "用法: bash tools/test_tools.sh <tool_name>"
    echo ""
    echo "可测试的工具列表:"
    echo "  --- 轻量级 (仅需 OpenAI API) ---"
    echo "  frame_selector      - 帧选择工具 (LLM)"
    echo "  image_grid_qa       - 网格图问答 (VLM API)"
    echo "  image_grid_select   - 网格图片段选择 (VLM API)"
    echo "  patch_zoomer        - 区域放大 (VLM API)"
    echo "  summarizer          - 信息汇总 (LLM)"
    echo "  all_light           - 运行以上所有轻量级工具测试"
    echo ""
    echo "  --- 需要 GPU 模型 ---"
    echo "  image_captioner     - BLIP 图像描述 (需 GPU)"
    echo "  image_captioner_llava - LLaVA 图像描述 (需 GPU + LLaVA 权重)"
    echo "  image_qa            - LLaVA 图像问答 (需 GPU + LLaVA 权重)"
    echo "  yolo_tracker        - YOLO 目标跟踪 (需 GPU + YOLOE 权重)"
    echo ""
    echo "  --- 需要 Grounded-Video-LLM 权重 ---"
    echo "  temporal_grounding  - 时间定位 (需 GPU + Grounded-Video-LLM)"
    echo "  temporal_qa         - 时序问答 (需 GPU + Grounded-Video-LLM)"
    echo "  temporal_referring  - 时间参照推理 (需 GPU + Grounded-Video-LLM)"
    echo ""
    echo "  --- 需要 Qwen / InternVL 模型 ---"
    echo "  video_qa            - Qwen2.5-VL 视频问答 (需 GPU + Qwen 权重)"
    echo "  video_qa_internvl   - InternVL 视频问答 (需 GPU + InternVL 权重)"
    echo ""
    echo "示例:"
    echo "  bash tools/test_tools.sh frame_selector"
    echo "  bash tools/test_tools.sh all_light"
fi
