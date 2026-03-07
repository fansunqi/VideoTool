"""
STAR (SpatioTemporal Alternating Reasoning) 推理核心逻辑。

使用 LangGraph StateGraph 实现时空交替推理：
- 维护可见帧字典 (VisibleFrames)
- LLM Planner 在每一步根据交替约束选择工具
- 强制时序/空间工具交替调用
- 迭代直到信息充足或达到上限
- 最终由 Generalist 工具生成答案
"""

import pdb
from typing import TypedDict, Literal, Optional, List, Any

from langgraph.graph import StateGraph, START, END

from engine.openai import ChatOpenAI
from star_prompts import (
    TOOL_CATEGORY_MAP,
    get_tool_category,
    PlannerDecision,
    STAR_PLANNER_SYSTEM_PROMPT,
    STAR_PLANNER_USER_PROMPT,
    STAR_GENERALIST_SYSTEM_PROMPT,
    STAR_GENERALIST_USER_PROMPT,
)


# ============================================================
# State 定义
# ============================================================

class STARState(TypedDict):
    """STAR 推理状态"""
    question: str                           # 用户问题
    question_w_options: str                 # 带选项的问题（用于最终回答）
    last_tool_type: str                     # "temporal" / "spatial" / "none"
    iteration_count: int                    # 当前迭代次数
    max_iterations: int                     # 最大迭代次数
    should_end: bool                        # 是否应终止
    final_answer: str                       # 最终答案
    tool_history: list                      # 工具调用历史 [{tool_name, tool_input, tool_output, tool_type}]
    selected_tool_name: str                 # Planner 选定的工具名（节点间传递）
    selected_tool_input: str                # Planner 选定的工具输入（节点间传递）


# ============================================================
# 辅助函数
# ============================================================

def _get_tool_instance_by_name(tool_instances: list, tool_name: str):
    """通过工具的 inference.name 查找工具实例"""
    for inst in tool_instances:
        if hasattr(inst, 'inference') and hasattr(inst.inference, 'name'):
            if inst.inference.name == tool_name:
                return inst
    return None


def _get_tool_class_name(tool_instance) -> str:
    """获取工具实例的类名"""
    return type(tool_instance).__name__


def _get_available_tools_for_type(
    tool_instances: list, 
    required_type: str
) -> list:
    """
    根据所需的工具类型过滤工具列表。
    
    Args:
        tool_instances: 所有工具实例列表
        required_type: "temporal" / "spatial" / "any"（首次调用时两类都可选）
    
    Returns:
        符合条件的工具实例列表（排除 generalist）
    """
    result = []
    for inst in tool_instances:
        cls_name = _get_tool_class_name(inst)
        category = get_tool_category(cls_name)
        
        # 跳过 generalist 工具（只在最终阶段使用）
        if category == "generalist":
            continue
        
        if required_type == "any":
            result.append(inst)
        elif category == required_type:
            result.append(inst)
    
    return result


def _format_tools_description(tool_instances: list) -> str:
    """将工具实例列表格式化为描述文本"""
    if not tool_instances:
        return "No tools available."
    
    lines = []
    for inst in tool_instances:
        cls_name = _get_tool_class_name(inst)
        category = get_tool_category(cls_name)
        if hasattr(inst, 'inference') and hasattr(inst.inference, 'name'):
            name = inst.inference.name
            desc = getattr(inst.inference, 'description', 'No description')
            lines.append(f"- **{name}** (type: {category}): {desc}")
    
    return "\n".join(lines) if lines else "No tools available."


def _get_next_required_type(last_tool_type: str) -> str:
    """
    根据上一步的工具类型，确定下一步应使用的工具类型。
    
    交替约束：
    - 上一步是 temporal → 这一步必须是 spatial
    - 上一步是 spatial → 这一步必须是 temporal
    - 首次 (none) → 两类都可选
    """
    if last_tool_type == "temporal":
        return "spatial"
    elif last_tool_type == "spatial":
        return "temporal"
    else:  # "none" — 首次调用
        return "any"


# ============================================================
# 构建 STAR 推理图
# ============================================================

def build_star_graph(
    tool_instances: list,
    visible_frames,
    planner_llm: ChatOpenAI,
    generalist_llm: ChatOpenAI,
) -> StateGraph:
    """
    构建 STAR LangGraph StateGraph。
    
    Nodes:
        - planner: LLM 根据交替约束选择工具
        - tool_executor: 执行选定的工具
        - generalist: 生成最终答案
    
    Edges:
        START -> planner
        planner -> (conditional) -> tool_executor / generalist
        tool_executor -> planner
        generalist -> END
    """

    # ----------------------------------------------------------
    # Planner Node
    # ----------------------------------------------------------
    def planner_node(state: STARState) -> dict:
        """LLM Planner：根据交替约束选择工具"""
        
        last_tool_type = state["last_tool_type"]
        iteration_count = state["iteration_count"]
        max_iterations = state["max_iterations"]
        question = state["question"]
        
        # 确定可用工具类型
        required_type = _get_next_required_type(last_tool_type)
        available_tools = _get_available_tools_for_type(tool_instances, required_type)
        
        # 如果没有可用工具（例如只配了一种类型的工具），放宽约束
        if not available_tools:
            print(f"\n[STAR Planner] No {required_type} tools available, relaxing constraint to 'any'")
            available_tools = _get_available_tools_for_type(tool_instances, "any")
        
        if not available_tools:
            print(f"\n[STAR Planner] No non-generalist tools available, ending iteration")
            return {"should_end": True}
        
        # 获取当前信息
        frame_descriptions = visible_frames.get_frame_descriptions()
        qa_descriptions = visible_frames.get_qa_descriptions()
        video_info = visible_frames.video_info
        
        # 构造 Planner Prompt
        user_prompt = STAR_PLANNER_USER_PROMPT.format(
            question=question,
            total_frames=video_info["total_frames"],
            duration=video_info["duration"],
            fps=video_info["fps"],
            num_visible_frames=visible_frames.get_frame_count(),
            frame_descriptions=frame_descriptions,
            qa_descriptions=qa_descriptions,
            current_iteration=iteration_count + 1,
            max_iterations=max_iterations,
            last_tool_type=last_tool_type,
            available_tools_description=_format_tools_description(available_tools),
        )
        
        print(f"\n{'='*60}")
        print(f"[STAR Planner] Iteration {iteration_count + 1}/{max_iterations}")
        print(f"[STAR Planner] Last tool type: {last_tool_type}")
        print(f"[STAR Planner] Required type: {required_type}")
        print(f"[STAR Planner] Available tools: {[inst.inference.name for inst in available_tools]}")
        
        # 调用 LLM
        decision = planner_llm.generate(
            user_prompt,
            system_prompt=STAR_PLANNER_SYSTEM_PROMPT,
            response_format=PlannerDecision,
        )
        
        if isinstance(decision, PlannerDecision):
            print(f"[STAR Planner] Decision: tool={decision.tool_name}, info_sufficient={decision.info_sufficient}")
            print(f"[STAR Planner] Reasoning: {decision.reasoning}")
            print(f"[STAR Planner] Tool input: {decision.tool_input}")
            
            # 如果 LLM 判断信息已充足
            if decision.info_sufficient:
                return {"should_end": True}
            
            # 验证选择的工具是否在可用列表中
            available_names = [inst.inference.name for inst in available_tools]
            if decision.tool_name not in available_names:
                print(f"[STAR Planner] Warning: selected tool '{decision.tool_name}' not in available tools, using first available")
                decision.tool_name = available_names[0]
            
            return {
                "should_end": False,
                "selected_tool_name": decision.tool_name,
                "selected_tool_input": decision.tool_input,
            }
        else:
            # LLM 返回错误或非结构化结果
            print(f"[STAR Planner] Error: unexpected LLM response: {decision}")
            return {"should_end": True}

    # ----------------------------------------------------------
    # Tool Executor Node
    # ----------------------------------------------------------
    def tool_executor_node(state: STARState) -> dict:
        """执行 Planner 选定的工具"""
        
        tool_name = state.get("selected_tool_name", "")
        tool_input = state.get("selected_tool_input", "")
        
        if not tool_name:
            print("[STAR ToolExec] No tool selected, skipping")
            return {}
        
        # 查找工具实例
        tool_instance = _get_tool_instance_by_name(tool_instances, tool_name)
        if tool_instance is None:
            print(f"[STAR ToolExec] Tool '{tool_name}' not found!")
            return {}
        
        cls_name = _get_tool_class_name(tool_instance)
        tool_type = get_tool_category(cls_name)
        
        print(f"\n[STAR ToolExec] Executing: {tool_name} (type: {tool_type})")
        print(f"[STAR ToolExec] Input: {tool_input}")
        
        # 执行工具
        try:
            tool_output = tool_instance.inference(input=tool_input)
        except Exception as e:
            print(f"[STAR ToolExec] Error executing {tool_name}: {e}")
            tool_output = f"Error: {str(e)}"
        
        print(f"[STAR ToolExec] Output: {str(tool_output)[:500]}")
        
        # 更新工具调用历史
        tool_history = list(state.get("tool_history", []))
        tool_history.append({
            "iteration": state["iteration_count"] + 1,
            "tool_name": tool_name,
            "tool_class": cls_name,
            "tool_type": tool_type,
            "tool_input": tool_input,
            "tool_output": str(tool_output)[:1000],
        })
        
        return {
            "last_tool_type": tool_type,
            "iteration_count": state["iteration_count"] + 1,
            "tool_history": tool_history,
        }

    # ----------------------------------------------------------
    # Generalist Node (最终回答)
    # ----------------------------------------------------------
    def generalist_node(state: STARState) -> dict:
        """使用 Summarizer 或 LLM 生成最终答案"""
        
        question = state.get("question_w_options", state["question"])
        
        print(f"\n{'='*60}")
        print(f"[STAR Generalist] Generating final answer...")
        print(f"[STAR Generalist] Total iterations: {state['iteration_count']}")
        print(f"[STAR Generalist] Tool history: {[h['tool_name'] for h in state.get('tool_history', [])]}")
        
        # 优先尝试使用 Summarizer 工具实例
        summarizer = _get_tool_instance_by_name(tool_instances, "summarization-tool")
        
        if summarizer is not None:
            print("[STAR Generalist] Using Summarizer tool")
            try:
                answer = summarizer.inference(input=question)
                return {"final_answer": answer}
            except Exception as e:
                print(f"[STAR Generalist] Summarizer failed: {e}, falling back to LLM")
        
        # Fallback: 直接用 LLM 汇总
        # 获取帧信息
        frame_info = visible_frames.get_qa_descriptions()
        if frame_info == "No QA information available.":
            frame_info = visible_frames.get_frame_descriptions()
        
        user_prompt = STAR_GENERALIST_USER_PROMPT.format(
            frame_information=frame_info,
            question=question,
        )
        
        answer = generalist_llm.generate(
            user_prompt,
            system_prompt=STAR_GENERALIST_SYSTEM_PROMPT,
        )
        
        print(f"[STAR Generalist] Answer: {answer}")
        return {"final_answer": str(answer)}

    # ----------------------------------------------------------
    # 条件路由
    # ----------------------------------------------------------
    def should_continue(state: STARState) -> str:
        """判断是继续迭代还是生成最终答案"""
        
        # 如果 Planner 标记为结束
        if state.get("should_end", False):
            print(f"[STAR Router] -> generalist (planner decided to end)")
            return "generalist"
        
        # 如果达到最大迭代次数
        if state["iteration_count"] >= state["max_iterations"]:
            print(f"[STAR Router] -> generalist (max iterations reached: {state['iteration_count']})")
            return "generalist"
        
        # 继续迭代
        print(f"[STAR Router] -> tool_executor")
        return "tool_executor"

    # ----------------------------------------------------------
    # 组装 StateGraph
    # ----------------------------------------------------------
    graph = StateGraph(STARState)
    
    # 添加节点
    graph.add_node("planner", planner_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("generalist", generalist_node)
    
    # 添加边
    graph.add_edge(START, "planner")
    
    # Planner -> 条件路由
    graph.add_conditional_edges(
        "planner",
        should_continue,
        {
            "tool_executor": "tool_executor",
            "generalist": "generalist",
        }
    )
    
    # Tool Executor -> 回到 Planner
    graph.add_edge("tool_executor", "planner")
    
    # Generalist -> END
    graph.add_edge("generalist", END)
    
    return graph


# ============================================================
# 顶层入口函数
# ============================================================

def star_reasoning(
    question: str,
    question_w_options: str,
    tool_instances: list,
    visible_frames,
    conf,
    planner_llm: ChatOpenAI = None,
    generalist_llm: ChatOpenAI = None,
) -> str:
    """
    STAR 推理入口函数。
    
    Args:
        question: 问题文本（不含选项）
        question_w_options: 带选项的问题文本
        tool_instances: 工具实例列表（已设置 visible_frames 和 video_path）
        visible_frames: VisibleFrames 实例（已初始化采样帧）
        conf: OmegaConf 配置
        planner_llm: Planner 使用的 LLM（如果 None 则从 conf 创建）
        generalist_llm: Generalist 使用的 LLM（如果 None 则从 conf 创建）
    
    Returns:
        最终答案字符串
    """
    
    # 获取 STAR 配置
    star_conf = conf.get("star", {})
    max_iterations = star_conf.get("max_iterations", 6)
    planner_model = star_conf.get("planner_model", "gpt-4o-mini")
    generalist_model = star_conf.get("generalist_model", "gpt-4o-mini")
    
    # 初始化 LLM
    if planner_llm is None:
        planner_llm = ChatOpenAI(
            model_string=planner_model,
            is_multimodal=False,
            enable_cache=True,
        )
    
    if generalist_llm is None:
        generalist_llm = ChatOpenAI(
            model_string=generalist_model,
            is_multimodal=False,
            enable_cache=True,
        )
    
    print(f"\n{'='*60}")
    print(f"[STAR] Starting STAR reasoning")
    print(f"[STAR] Question: {question}")
    print(f"[STAR] Max iterations: {max_iterations}")
    print(f"[STAR] Planner model: {planner_model}")
    print(f"[STAR] Generalist model: {generalist_model}")
    print(f"[STAR] Available tools: {[type(t).__name__ for t in tool_instances]}")
    print(f"[STAR] Initial visible frames: {visible_frames.get_frame_count()}")
    print(f"{'='*60}")
    
    # 构建图
    graph = build_star_graph(
        tool_instances=tool_instances,
        visible_frames=visible_frames,
        planner_llm=planner_llm,
        generalist_llm=generalist_llm,
    )
    
    # 编译图
    app = graph.compile()
    
    # 初始状态
    initial_state: STARState = {
        "question": question,
        "question_w_options": question_w_options,
        "last_tool_type": "none",
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "should_end": False,
        "final_answer": "",
        "tool_history": [],
        "selected_tool_name": "",
        "selected_tool_input": "",
    }
    
    # 执行图
    final_state = app.invoke(initial_state)
    
    answer = final_state.get("final_answer", "")
    
    print(f"\n{'='*60}")
    print(f"[STAR] Reasoning complete")
    print(f"[STAR] Total iterations: {final_state['iteration_count']}")
    print(f"[STAR] Tool history: {[h['tool_name'] for h in final_state.get('tool_history', [])]}")
    print(f"[STAR] Final answer: {answer}")
    print(f"{'='*60}")
    
    return answer
