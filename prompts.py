QUERY_PREFIX = """Question: """

ASSISTANT_ROLE = """You are an AI assistant for video analysis. Regarding a given video, you will receive information from sampled frames, use tools to extract additional information, and answer the following question as accurately as possible.
"""

TOOLS_RULE = """Please strictly follow the rules below while using the tools:
Rule 1: Do not use the frame-extraction-tool as the first tool. 
Rule 2: If, after using other tools, you still do not have enough information to provide a clear answer, you must use the frame-extraction-tool to extract more frames from the video.
Rule 3: After using the frame-extraction-tool, please continue with other tools to analyze the extracted frames."
Rule 4: The same tool should not be invoked consecutively.
"""


SELECT_FRAMES_PROMPT = """Given a video that has {num_frames} frames, the frames are decoded at {fps} fps. 

Given the following information of sampled frames in the video:
```
{visible_frames_info}
```

To answer the following question: 
``` 
{question}
```

However, the information in the initial sampled frames is not suffient. Our goal is to identify additional frames that contain crucial information necessary for answering the question. These frames should not only address the query directly but should also complement the insights gleaned from the descriptions of the initial frames.

To achieve this, we will:

1. List the uninformed video segments between sampled frames in the format 'segment_id': 'start_frame_index'-'end_frame_index': 
```
{candidate_segment}
```

2. Determine which segments are likely to contain frames that are most relevant to the question. These frames should capture key visual elements, such as objects, humans, interactions, actions, and scenes, that are supportive to answer the question.

Return the selected video segments in json format:
<analysis>: Describe the known information of sampled frames first. Then analyze the question and select the most relevant uninformed video segment.
<segments>: List of segment ids of selected video segments, i.e., list of integers in [0, {max_candidate_segment_id}].
"""


QUERY_PREFIX_DES = """Regarding a given video, based on the following frame caption to answer the following question as best you can.

Frame Caption: 
{frame_caption}

Question:
{question}
"""


QUERY_PREFIX_INFO = """Regarding a given video, based on the frame information to answer the following question as best you can.

Frame Information: 
{frame_information}

Question:
{question}
"""


PATCH_ZOOMER_PROMPT = """Analyze this image to identify the most relevant region(s) for answering the question:

Question: {question}

The image is divided into 5 regions:
- (A) Top-left quarter
- (B) Top-right quarter
- (C) Bottom-left quarter
- (D) Bottom-right quarter
- (E) Center region (1/4 size, overlapping middle section)

Instructions:
1. First describe what you see in each of the five regions.
2. Then select the most relevant region(s) to answer the question.
3. Choose only the minimum necessary regions - avoid selecting redundant areas that show the same content. For example, if one patch contains the entire object(s), do not select another patch that only shows a part of the same object(s).


Response in json format:
<analysis>: Describe the image and five patches first. Then analyze the question and select the most relevant patch or list of patches.
<patch>: List of letters (A-E)
"""

EVAL_PROMPT = """Given the following question and answer, determine which option matches the provided answer.
If the answer matches exactly one of the options, return that option in full, e.g. 'A. one' ; if it matches none or more than one option, return 'not matched'.

Question: {question}
Answer: {answer}
Options: {options_with_labels}

The matched option is:"""


# EVAL_PROMPT = """Given the following question and answer, determine which option matches the provided answer.
# If the answer matches one of the options, return that option in full, e.g. 'A. one' ; if it matches none or more than one option, return 'not matched'.

# Question: {question}
# Answer: {answer}
# Options: {options_with_labels}

# The matched option is:"""


IMAGE_GRID_SELECT_PROMPT = """Analyze this image sequence to identify the most relevant video segment(s) for answering the question:

Question: {question}

The image sequence is {grid_num} sampled frames from a video. The number in the circle indicates the frame index.

Instructions:
1. First describe what you see in each of the {grid_num} frames.
2. Then select the most relevant video segment(s) to answer the question.
3. Choose only the minimum necessary segments.

Response in json format:
<analysis>: Describe the {grid_num} frames first. Then analyze the question and select the most relevant segment(s).
<start>: the start frame index of the selected segment(s), an integer in [1, {grid_num}]
<end>: the end frame index of the selected segment(s), an integer in [start+1, {grid_num}]
"""

IMAGE_GRID_QA_PROMPT = \
"""I will show you an image sequence of {grid_num} sampled frames from a video. I have annotated the images with numbered circles. Based on the image sequence, try to answer this question: 
{question}"""


IMAGE_GRID_QA_PROMPT_ANALYSIS = \
"""Analyze this image sequence to answer the question:

Question: {question}

The image sequence is {grid_num} sampled frames from a video arranged in temporal order. The number in the circle indicates the frame index.

Instructions:
1. First describe and analyze what you see in each of the {grid_num} frames.
2. Then answer the question based on your descriptions and analysis.

Response in json format:
<analysis>: Describe the {grid_num} frames first. Then analyze the question.
<answer>: Answer to the question.
"""



IMAGE_GRID_QA_PROMPT_SUBTITLE = \
"""I will show you an image sequence of {grid_num} sampled frames from a video. I have annotated the images with numbered circles. 

The total duration of this video is {duration} seconds. The sampled frames are taken at the following timestamps:
{frame_timestamps}

The subtitles of this video, organized by timestamp, are as follows:
{subtitle_desp}

Based on the image sequence and the subtitles, try to answer this question: 
{question}"""