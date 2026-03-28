"""
STAR (SpatioTemporal Alternating Reasoning) 专用 Prompt 模板和 Pydantic 模型。
"""

from pydantic import BaseModel, Field
from typing import Literal


# ============================================================
# Pydantic 结构化输出模型
# ============================================================

class PlannerDecision(BaseModel):
    """Planner LLM 的结构化输出：选择下一步使用哪个工具"""
    reasoning: str = Field(description="Analysis of current state and reasoning for tool selection")
    tool_name: str = Field(description="Name of the selected tool (must be one of the available tools)")
    tool_input: str = Field(description="Input string to pass to the selected tool")
    info_sufficient: bool = Field(description="Whether the collected information is already sufficient to answer the question. If True, the system will proceed to generate the final answer instead of calling the selected tool.")


# ============================================================
# 工具分类映射
# ============================================================

TOOL_CATEGORY_MAP: dict[str, Literal["temporal", "spatial", "generalist"]] = {
    # 时序工具 (Temporal)
    "FrameSelector": "temporal",
    "TemporalGrounding": "temporal",
    "TemporalQA": "temporal",
    "TemporalReferring": "temporal",
    "ImageGridSelect": "temporal",   # 在时间维度上聚焦片段，归为时序
    # 空间工具 (Spatial)
    "ImageQA": "spatial",
    "ImageCaptioner": "spatial",
    "ImageCaptionerLLaVA": "spatial",
    "ImageGridQA": "spatial",
    "PatchZoomer": "spatial",
    "YOLOTracker": "spatial",
    # 通用工具 (Generalist)
    "Summarizer": "generalist",
    "VideoQA": "generalist",
    "VideoQAInternVL": "generalist",
}


def get_tool_category(tool_class_name: str) -> str:
    """获取工具类名对应的类别"""
    return TOOL_CATEGORY_MAP.get(tool_class_name, "unknown")


# ============================================================
# Planner System Prompt
# ============================================================

STAR_PLANNER_SYSTEM_PROMPT = """You are an expert AI planner for video question answering. Your job is to select the most appropriate tool to gather information from a video in order to answer a given question.

You follow the STAR (SpatioTemporal Alternating Reasoning) framework:
- You alternate between temporal tools (which operate on the time dimension, e.g., selecting frames, temporal grounding) and spatial tools (which analyze individual frames, e.g., image QA, captioning, object detection).
- This alternating pattern ensures comprehensive video understanding by combining WHEN something happens with WHAT is happening.

Key principles:
1. Each iteration, you MUST select a tool from the available tools listed below (these have already been filtered by the alternating constraint).
2. Analyze what information has been collected so far and what is still missing.
3. Choose the tool that will provide the most useful additional information.
4. If the collected information is sufficient to answer the question, set info_sufficient=True.
5. Provide a clear, specific input for the selected tool (usually the question or a relevant sub-question).

CRITICAL RULES:
6. After a temporal tool (e.g., frame extraction) has been used, newly extracted frames have NOT been analyzed yet. You MUST use a spatial tool to analyze these new frames BEFORE concluding that information is sufficient. NEVER set info_sufficient=True immediately after a temporal tool.
7. Frame indices reflect temporal order in the video: lower frame indices correspond to earlier moments, higher frame indices correspond to later moments. When the question asks about "earliest", "first", or "beginning", pay special attention to information from frames with the LOWEST indices.

IMPORTANT STRATEGY FOR SPATIAL TOOLS (ImageQA, ImageCaptioner, etc.):
8. When using spatial tools like ImageQA, ask **descriptive/factual questions** about what is visible in each frame, NOT the overall question directly. Spatial tools analyze each frame independently — a single frame cannot determine global/comparative answers (e.g., "what is the earliest/latest", "how many total", "what happens most often").
   - BAD example: "What is the earliest stage of human evolution in the video?" (each frame will claim its own content is the answer)
   - GOOD example: "What species name, text, or label is displayed in this frame?" or "What event or action is shown in this frame?"
   - GOOD example: "What time period or date is mentioned in this frame?" or "What numbers or statistics are visible?"
9. Break the overall question into descriptive sub-questions that extract factual information from each frame. The Summarizer will later synthesize all frame-level facts to answer the original question.

AVOIDING REDUNDANT QUERIES:
10. Do NOT repeatedly use the same tool_input across iterations. If you have already asked a question to ImageQA, use a DIFFERENT sub-question next time to gather complementary information (e.g., first ask about text/labels, then about time periods, then about visual details).
11. Review the "QA Information" section carefully — if frames already have answers for a particular question, asking the same question again will yield no new information.
"""


# ============================================================
# Planner User Prompt Template
# ============================================================

STAR_PLANNER_USER_PROMPT = """## Question
{question}

## Video Information
- Total frames: {total_frames}
- Duration: {duration:.1f} seconds
- FPS: {fps:.1f}
- Currently visible frames: {num_visible_frames}

## Collected Information So Far

### Frame Descriptions
{frame_descriptions}

### QA Information
{qa_descriptions}

## Previous Tool Calls
{tool_history_description}

## Iteration Status
- Current iteration: {current_iteration} / {max_iterations}
- Previous tool type: {last_tool_type}

## Available Tools (filtered by alternating constraint)
{available_tools_description}

## Instructions
Based on the question and the information collected so far, select the best tool to gather more information, or indicate that the information is sufficient.
- If this is iteration {current_iteration} of {max_iterations} and you still need more info, prioritize the most impactful tool.
- If the collected information is clearly sufficient to answer the question, set info_sufficient to True.
- IMPORTANT: Do NOT reuse the same tool_input as any previous call. If you need to use the same tool again, formulate a DIFFERENT sub-question to extract complementary information.
"""


# ============================================================
# Generalist（最终回答）System Prompt
# ============================================================

STAR_GENERALIST_SYSTEM_PROMPT = """You are an expert AI assistant for video question answering. Based on the collected frame information from a video, provide an accurate and concise answer to the question.

IMPORTANT: Each frame was analyzed independently, so individual frame answers may contain incorrect global claims (e.g., each frame might claim its content is "the earliest" or "the most important"). You must reason across ALL frames to determine the correct answer, using frame indices as temporal ordering (lower index = earlier in video)."""


STAR_GENERALIST_USER_PROMPT = """Regarding a given video, based on the frame information to answer the following question as best you can.

IMPORTANT TEMPORAL REASONING RULES:
1. Frame indices indicate temporal position — lower indices appear EARLIER, higher indices appear LATER in the video.
2. Each frame's answer describes ONLY what is visible in that single frame. Do NOT trust individual frame claims about global properties (e.g., "this is the earliest"). Instead, compare information across ALL frames.
3. When different frames mention different time periods, compare the actual values to determine temporal order.

Frame Information: 
{frame_information}

Question:
{question}
"""
