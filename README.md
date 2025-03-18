
# <img src="assets/icon.png" alt="icon" style="width:35px;margin-bottom:-8px;"/> MAT: <ins>M</ins>ulti-modal <ins>A</ins>gent <ins>T</ins>uning  **🔥 ICLR 2025 (Spotlight)**

![authors](assets/authors.png) 

🚀 **VLM-Powered Agent for Intelligent Tool Orchestration** | 🔧 **Open-source Framework for Multi-modal AI**

<p align="center">
  <a href="https://github.com/mat-agent/MAT-Agent/stargazers">
    <img src="https://img.shields.io/github/stars/mat-agent/MAT-Agent?style=social&label=Stars" alt="GitHub Stars">
  </a>
  <a href="https://arxiv.org/abs/2412.15606">
    <img src="https://img.shields.io/badge/arXiv-2412.15606-b31b1b?logo=arxiv" alt="arXiv Paper">
  </a>
</p>

<p align="center">
  🤗 <a href="https://huggingface.co/datasets/PengxiangLi/MAT">Dataset</a> • 
  🌐 <a href="https://mat-agent.github.io/">Webpage</a> • 
  📄 <a href="https://arxiv.org/pdf/2412.15606">Paper</a>  
<!--   🚦 <a href="#quickstart">Quick Start</a> -->
</p>

<!-- ![MAT Architecture](assets/framework.jpg)  -->
<!-- *✨ MAT Framework Overview - Seamless Integration of Vision, Language, and Tools* -->

## 🌟 Features
- 🎯 **Precision Tools** - 10+ integrated tools (SAM 2.0, Web Search, OCR, etc.) 
- 🧠 **Multi-modal Memory** - Context-aware task execution
- 🔄 **Self-Correction** - Auto-verification mechanism for reliable outputs
- 🌍 **Real-World Ready** - Google Search API integration ets

### Performance Highlights (Baseline: Minicpm-V-8.5B)
| Model        | GAIA-L1 | GAIA-L2 | GAIA-L3 | GTA   |
|--------------|---------|---------|---------|-------|
| Baseline     |13.21%   | 5.81%   | 0.00%   | 33.97% |
| **MAT**      | 🚀26.42% | 🚀11.63% | 🚀3.84% | 🚀52.56% |
# 🛠️ Setup

## Install environment
```bash
conda create -n tongagent python=3.10
conda activate tongagent

pip install -r requirements.txt
```

If you want to generate data by yourself, install the following environment.
```bash
pip install -r requirements_generation.txt
```
## Dowanload dataset
You can use `git lfs` or `huggingface-cli` to download the dataset we used in paper from [HF dataset](https://huggingface.co/datasets/PengxiangLi/MAT). **Images related to training is zipped in a file called files.zip**.


## Image source

The image captions and caption embeddings can be downloaded via the following link:
[Google Drive](https://drive.google.com/drive/folders/1Ek6qfmhcaTd7zTEQcBvELh6i7unVhTrk?usp=sharing).

Please follow [ShareGPT4V](https://sharegpt4v.github.io/) to organize the image source as follows:
```none

├── ...
├── image_source
│   ├── llava
│   │   ├── llava_pretrain
│   │   │   ├── images
│   ├── coco
│   │   ├── train2017
│   ├── sam
│   │   ├── images
│   ├── web-celebrity
│   │   ├── images
│   ├── web-landmark
│   │   ├── images
│   ├── wikiart
│   ├── share_textvqa
│   │   ├── images
│   ├── chatqa
│   │   ├── train
│   │   │   ├── png

```

## Download model parameters for vision tools
You only need to download SAM 2 manually. For other models, `transformers` will do downloading for you.

Put the folder `model_checkpoints` in your repo's root so that you have something like
```
main.py
model_checkpoints/sam2_checkpoints
model_checkpoints/sam2_configs
```
You can download the model checkpoints and configs by scripts from from the official repo.
* [sam2_checkpoints](https://github.com/facebookresearch/sam2/blob/main/checkpoints/download_ckpts.sh)
* [sam2_configs](https://github.com/facebookresearch/sam2?tab=readme-ov-file#sam-21-checkpoints)

### Setup Google Customized Search
This project using Google Customized Search to search the web. You need to set the `cx` and `key` in `configs/agent_config.yaml`. You will find the `cx` and `key` in the `search_engine` section.
```yaml
search_engine:
  -
    cx: # enter your cx here
    key: # enter your key here
```
To obtain this key, check the official API documentation[here](https://console.cloud.google.com/apis/api/customsearch.googleapis.com). It has a rate-limit 100 query per day for free user 10k query per day for paid user.


# Execute with closed-source api
## Setup
First, you need to set the api key and endpoint in `configs/agent_config.yaml`. The config file looks like this:
```yaml
tonggpt:
  model_name:  gpt-4o-2024-08-06
  region: eastus
  api_key: # enter your api key here
  open_ai_client_type: openai # or azure
  endpoint: # only for azure, you need to specify the endpoint you are using

agent_controller:
  engine_type: tonggpt # use minicpm, qwen if you want to use other models
```
We use GPT on Azure and provide a simple alternative for you to use original OpenAI client.

## Download benchmark dataset
You can download the GTA dataset from [GTA Link](https://github.com/open-compass/GTA/releases/download/v0.1.0/gta_dataset.zip), and revise your dataset path `data/gta_dataset/dataset.json` in `examples/gta/main.py` if you put it in some other path.

You can download the GAIA dataset from [GAIA Link](https://huggingface.co/datasets/gaia-benchmark/GAIA). Or running evaluation script will automatically download the dataset from HF.

## Run

Run in command line manner with arbitrary prompt.
```bash
python main.py --prompt 'Can you edit the image to turn him into cyborg? Image path: tests/data/draw.jpg.'
```

See results runing on GAIA set
```bash
python examples/gaia/main.py
```

See results runing on GTA set
```bash
python examples/gta/main.py
```

# Experiments
## MiniCPM-V
Refer to official repo [OpenBMB/MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) for environment setup. Since Qwen-VL might have different version than MiniCPM-V, you should consider using a new conda environment.

To train the model, enter the directory and run the script:
```bash
cd experiments/CPM-FT

# for training a model for GAIA dataset
bash slurm_jobs/job_lora_5_gaia_1206.sh

# for training a model for GTA dataset
bash slurm_jobs/job_lora_5_gta_with_verifier.sh
```
Check this scripts for assign data path. It should takes 4 hours on 8X A100 for 50K dataset per epoch.

## Qwen-VL
Refer to official repo [Qwen-VL](https://github.com/QwenLM/Qwen2-VL) for environment setup.


After setup the environment, you can run the script convert dataset from MiniCPM-V to Qwen-VL format:
```bash
cd experiments/Qwen-VL

python scripts/convert_dataset_v2.py
```
Then you can run the script to train the model:
```bash
bash slurm_jobs/train_gaia.sh
bash slurm_jobs/train_gta.sh
```

## Evaluation
To evaluate the model, first modify the `configs/agent_config.yaml` to set the model path. Then run the script:
```bash
export RUN_MODE=eval

# for GAIA dataset
python examples/gaia/main.py --engine minicpm --lora-path experiments/CPM-FT/output/cpm_v2_6_7904295_2024_12_10_23_05/ --data-name 2023_level1 --split validation

python examples/gaia/main.py --engine minicpm --lora-path experiments/CPM-FT/output/cpm_v2_6_7904295_2024_12_10_23_05/ --data-name 2023_level2 --split validation

python examples/gaia/main.py --engine minicpm --lora-path experiments/CPM-FT/output/cpm_v2_6_7904295_2024_12_10_23_05/ --data-name 2023_level3 --split validation
# for GTA dataset
python examples/gta/main.py --engine minicpm --lora-path experiments/CPM-FT/output/cpm_v2_6_7904295_2024_12_10_23_05/
```
`cpm_v2_6_7904295_2024_12_10_23_05` is the model path. The training script automatically saves the model to that path. We use SLURM in our cluster such that the path consists of the job id and the time of the job. You should check the training script for the exact path.

Both benchmarks will output the results in `.cache` folder. You should use `eval.py` to get the metric we reported in the paper.

```bash
python examples/gaia/eval.py --data-path .cache/qa_cache/validation/minicpm/experiments/CPM-FT/output/cpm_v2_6_7904295_2024_12_10_23_05/2023_level1.db

python examples/gta/eval.py --folder .cache/gta/cpm_v2_6_7904295_2024_12_10_23_05/
```

## Data Generation
Run in command line manner. 
```bash
bash data_generation.sh
```

## 🌟 Star History
[![Star History Chart](https://api.star-history.com/svg?repos=mat-agent/MAT-Agent&type=Date)](https://star-history.com/#mat-agent/MAT-Agent&Date)

# Acknowledgement
Thanks for their brilliant contributions to the community! Here are the codebases we built upon.

Our agent is based on the wonderful Huggingface Agent framework.
* https://huggingface.co/docs/transformers/v4.47.1/en/main_classes/agent#transformers.ReactCodeAgent

Our agent design is inspired by the following works:
* https://github.com/aymeric-roucher/GAIA
* https://github.com/Ag2S1/Sibyl-System

Model training and inference code:
* https://github.com/OpenBMB/MiniCPM-V
* https://github.com/QwenLM/Qwen2-VL

# Citation
If you find our work helpful, please consider cite our paper 📝 and star us ⭐️！

```bib
@inproceedings{gao2025multi,
      title={Multi-modal Agent Tuning: Building a VLM-Driven Agent for Efficient Tool Usage},
      author={Gao, Zhi and Zhang, Bofei and Li, Pengxiang and Ma, Xiaojian and Yuan, Tao and Fan, Yue and Wu, Yuwei and Jia, Yunde and Zhu, Song-Chun and Li, Qing},
      booktitle={The Thirteenth International Conference on Learning Representations(ICLR)},
      year=2025
}
```
