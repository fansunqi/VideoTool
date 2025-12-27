# VideoTool

Keep Updating...

[![GitHub license](https://img.shields.io/badge/License-MIT-green.svg?logo=github)](https://lbesson.mit-license.org/)
[![Arxiv](https://img.shields.io/badge/arXiv-2512.10359-B31B1B.svg?logo=arxiv)](https://arxiv.org/abs/2512.10359)

This repository is the official implementation of [Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task](https://arxiv.org/abs/2512.10359) (NeurIPS 2025 main track).

## News and Todo 🗓️

- [ ] Release all video tools and test scripts

- [ ] Release toolChain algorithm (STAR)

- [ ] Release evaluating scripts

## Setup and Configuration 🛠️

1. Clone the repository 📦:
   ```python
   git clone git@github.com:fansunqi/VideoTool.git
   cd ToolChainVideo
   ```
2. Create a virtual environment 🧹 and install the dependencies 🧑‍🍳:
   ```python
   conda create -n videotool python=3.9
   conda activate videotool
   pip install -r requirements.txt
   ```
3. Set up your API key 🗝️ in `config/*.yaml`:
     ```python
     openai:
       GPT_API_KEY: "put your openai api key here"
       PROXY: "put your openai base url here"
     ```

5. Bulid related projects 🧩:
    ```python
    mkdir projects
    cd projects
    ```
   - **Download [Grounded-Video-LLM](https://github.com/WHB139426/Grounded-Video-LLM) for temporal grounding and temporal QA**

        ```python
        git clone git@github.com:WHB139426/Grounded-Video-LLM.git
        ```
   - **Build [LLaVA](https://github.com/haotian-liu/LLaVA) for image QA**

     ```python
     git clone git@github.com:fansunqi/LLaVA.git
     cd LLaVA
     pip install -e .
     cd ..
     ```


## Tools

Thanks to the authors of these open-source projects for providing excellent projects.

Temporal Tools:
- Frame Selector
    + select frames of interest based on current information, driven by LLM.
- Temporal Grounding
    + Grounded-Video-LLM-7B: https://github.com/WHB139426/Grounded-Video-LLM
- Temporal Refering
    + Grounded-Video-LLM-7B: https://github.com/WHB139426/Grounded-Video-LLM
- Temporal QA
    + Grounded-Video-LLM-7B: https://github.com/WHB139426/Grounded-Video-LLM

Spatial Tools:
- Object Tracking 
    + YOLO by ultralytics: https://github.com/ultralytics/ultralytics
- Image Captioning
    + BLIP: https://huggingface.co/docs/transformers/model_doc/blip
- Image QA
    + BLIP: https://huggingface.co/docs/transformers/model_doc/blip
    + LLaVA: https://github.com/haotian-liu/LLaVA

Generalist Solution:
- Image Grid QA
    + Image Grid QA driven by GPT-4o: https://github.com/microsoft/VLM-Video-Action-Localization
- Video QA
    + Qwen-VL-2.5-7B: https://github.com/QwenLM/Qwen2.5-VL


## Download Datasets
- NeXT-QA：
  ```
  git clone git@github.com:doc-doc/NExT-QA.git
  ```
  specify your data path in ```config/nextqa.yaml```
