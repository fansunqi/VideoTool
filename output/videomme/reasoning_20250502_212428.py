import pdb

from prompts import (
    QUERY_PREFIX,
    TOOLS_RULE,
    ASSISTANT_ROLE,
)

from util import load_cache, save_cache

from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.prebuilt import create_react_agent

from tools.yolo_tracker import YOLOTracker
from tools.image_captioner import ImageCaptioner
from tools.frame_selector import FrameSelector
from tools.image_qa import ImageQA
from tools.temporal_grounding import TemporalGrounding
from tools.image_grid_qa import ImageGridQA
from tools.summarizer import Summarizer
from tools.patch_zoomer import PatchZoomer
from tools.temporal_qa import TemporalQA
from tools.video_qa import VideoQA
from tools.image_captioner_llava import ImageCaptionerLLaVA
from tools.image_grid_select import ImageGridSelect

from eval import get_predicted_option_with_rephrase


def identify_tools(tool_class, tools):
    for tool in tools:
        if isinstance(tool, tool_class):
            return tool


def spatiotemporal_reasoning(
    question,
    question_w_options,
    options,
    tools, 
    frames_count,
    eval_llm=None,
):

    # image_grid_qa = identify_tools(ImageGridQA, tools)
    # image_grid_select = identify_tools(ImageGridSelect, tools)
    
    image_qa = identify_tools(ImageQA, tools)
    summarizer = identify_tools(Summarizer, tools)
    
    image_qa.inference(input=question)
    summarizer_output = summarizer.inference(input=question_w_options)
    
    output = summarizer_output
    print(f"\nToolChainOutput: {output}") 
    return output
    
    
    

def spatiotemporal_reasoning_videomme(
    question,
    question_w_options,
    options,
    tools, 
    frames_count,
    eval_llm=None,
):

    image_grid_qa = identify_tools(ImageGridQA, tools)
    image_grid_select = identify_tools(ImageGridSelect, tools)
    
    try_count = 0
    try_max = image_grid_qa.conf.reasoning_try_max
    while try_count < try_max:
        try_count += 1
        not_last_turn = (try_count < try_max)
        
        image_grid_qa_output = image_grid_qa.inference(input=question_w_options)
        print(f"\nimage_grid_qa_output: {image_grid_qa_output}")    
        output = image_grid_qa_output
      
        if not_last_turn:
            image_grid_qa_pred, _ = get_predicted_option_with_rephrase(
                image_grid_qa_output, options, question, image_grid_qa.conf, eval_llm
            )
        
            if image_grid_qa_pred == -1:
                print("\nusing image grid select")
                image_grid_select.inference(input=question_w_options)
                frames_count = frames_count.union(image_grid_select.visible_frames.get_frame_indices())
            else:
                break
    
    print(f"\nToolChainOutput: {output}") 
    return output
    
    
    
    
    
    
def spatiotemporal_reasoning_nextqa(
    question,
    question_w_options,
    options,
    tools, 
    frames_count,
    eval_llm=None,
):

    temporal_grounding = identify_tools(TemporalGrounding, tools)
    patch_zoomer = identify_tools(PatchZoomer, tools)
    image_grid_qa = identify_tools(ImageGridQA, tools)  
    image_qa = identify_tools(ImageQA, tools)
    summarizer = identify_tools(Summarizer, tools)
    temporal_qa = identify_tools(TemporalQA, tools)
    frame_selector = identify_tools(FrameSelector, tools)
     
    # 1. T: temporal grounding
    temporal_grounding.inference(input=question)

    # 2. S: patch zoomer 对所有 visible_frames 都进行 zoom in
    patch_zoomer.inference(input=question)

    # 3. image grid qa
    image_grid_qa_output = image_grid_qa.inference(input=question_w_options)
    image_grid_qa_pred, _ = get_predicted_option_with_rephrase(
        image_grid_qa_output, options, question, image_grid_qa.conf, eval_llm
    )

    # 4. image qa LLaVA
    image_qa.inference(input=question)
    summarizer_output = summarizer.inference(input=question_w_options)
    summarizer_pred, _ = get_predicted_option_with_rephrase(
        summarizer_output, options, question, summarizer.conf, eval_llm
    )

    # 5. temporal qa
    temporal_qa_output = temporal_qa.inference(input=question_w_options)
    temporal_qa_pred, _ = get_predicted_option_with_rephrase(
        temporal_qa_output, options, question, temporal_qa.conf, eval_llm
    )

    if (image_grid_qa_pred == -1) or (summarizer_pred == -1) or (temporal_qa_pred == -1) or \
        (image_grid_qa_pred != summarizer_pred) or (image_grid_qa_pred != temporal_qa_pred) or (summarizer_pred != temporal_qa_pred):
        # 有一个方法不确定，进行 frame_selector

        invisible_segments_list = frame_selector.visible_frames.get_invisible_segments()

        # 检查是否还有分割的余地
        if len(invisible_segments_list) > 0:
            print("\nFrame Selector inferencing...")
            frame_selector.inference(input=question)

            # 3. image grid qa
            image_grid_qa_output = image_grid_qa.inference(input=question_w_options)

            # 4. image qa LLaVA
            image_qa.inference(input=question)
            summarizer_output = summarizer.inference(input=question_w_options)

            # 5. temporal qa
            temporal_qa_output = temporal_qa.inference(input=question_w_options)


    output = [image_grid_qa_output, summarizer_output, temporal_qa_output]

    print(f"\nToolChainOutput: {output}") 
    return output





def langgraph_reasoning( 
    input_question, 
    llm, 
    tools,
    recursion_limit=24,  
    use_cache=True,
    mannual_cache=None,
    mannual_cache_file=None
):

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ASSISTANT_ROLE),
            ("placeholder", "{messages}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    
    def _modify_state_messages(state: AgentState):
        return prompt.invoke({"messages": state["messages"]}).to_messages()
    
    tool_planner = create_react_agent(llm, tools, state_modifier=_modify_state_messages)
    
    query = QUERY_PREFIX + input_question + '\n\n' + TOOLS_RULE
    
    if use_cache and (query in mannual_cache):
        print("\nCache hit!")
        steps = mannual_cache[query]
    else:
        print("\nCache miss. Calling API...")
        steps = []
    
        for step in tool_planner.stream(
            {"messages": [("human", query)]}, 
            {"recursion_limit": recursion_limit},
                stream_mode="values"):
            
            step_message = step["messages"][-1]

            if isinstance(step_message, tuple):
                print(step_message)
            else:
                step_message.pretty_print()
        
            steps.append(step)
        
        save_cache(mannual_cache, query, steps, mannual_cache_file)    
 
    try:
        output = steps[-1]["messages"][-1].content
    except:
        output = None
    
    print(f"\nToolChainOutput: {output}") 
    return output