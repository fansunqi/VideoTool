from engine.openai import ChatOpenAI
from prompts import QUERY_PREFIX_DES, QUERY_PREFIX_INFO

def prompts(name, description):
    
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator

class Summarizer:
    def __init__(
        self,
        conf = None, 
    ):
        self.conf = conf
        self.visible_frames = None
        self.video_path = None

        model_string = conf.tool.summarizer.llm_model_name
        print(f"\nInitializing Summarizer Tool with model: {model_string}")
        self.llm_engine = ChatOpenAI(
            model_string=model_string, 
            is_multimodal=False,
            enable_cache=conf.tool.summarizer.use_cache
        )
        
        self.mode = conf.tool.summarizer.mode
    
    def set_frames(self, visible_frames):
        self.visible_frames = visible_frames
    
    def set_video_path(self, video_path):
        self.video_path = video_path  

    @prompts(
        name = "summarization-tool",
        description = "Useful when you want to summarize the infomation of all visible frames and find the answer."
        "The input to this tool must be a question without options, such as 'How many children are in the video?', instead of 'How many children are in the video? A. 1 B. 2 C. 3 D. 4'."
    )
    def inference(self, input):
        
        if self.mode == "by_caption":
            info = self.visible_frames.get_frame_descriptions()
        elif self.mode == "by_qa":
            info = self.visible_frames.get_qa_descriptions()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        input_prompt = QUERY_PREFIX_INFO.format(
            frame_information = info,
            question = input,
        )

        print("\nSummarizer Input Prompt: ", input_prompt)

        output = self.llm_engine(input_prompt)

        print("\nSummarizer Output Answer: ", output)

        return output