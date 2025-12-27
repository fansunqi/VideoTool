from langchain_openai import ChatOpenAI
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
        
        self.visible_frames = None
        self.video_path = None

        self.llm = ChatOpenAI(
            api_key = conf.openai.GPT_API_KEY,
            model = conf.openai.GPT_MODEL_NAME,
            temperature = 0,
            base_url = conf.openai.PROXY
        )
    
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

        all_frames_descriptions = self.visible_frames.get_frame_descriptions()

        # input_prompt = QUERY_PREFIX_DES.format(
        #     frame_caption = all_frames_descriptions,
        #     question = input,
        # )

        input_prompt = QUERY_PREFIX_INFO.format(
            frame_information = all_frames_descriptions,
            question = input,
        )

        print("\nSummarizer Input Prompt: ", input_prompt)

        output = self.llm.invoke(input_prompt)

        print("\nSummarizer Output Answer: ", output.content)

        return output.content