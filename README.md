# MBPP (Mostly Basic Python Problems) MCTS Generation

## Overview

The **MBPP-MCTS** (Mostly Basic Python Problems - Monte Carlo Tree Search) data generation process is designed to facilitate the training of a process reward model (ER-PRM), as described by Hanning Zhang et al. (2024) for MBPP data. This data is essential for fine-tuning large language models (LLMs) through advanced techniques such as **RAFT** (Hanze Dong et al., 2024) and **RLHF** (Long Ouyang et al., 2022; Yuxi Xie et al., 2024).

### Key Distinctions

Unlike traditional question-and-answer datasets that emphasize reasoning—such as GSM8K or MATH (Karl Cobbe et al., 2021)—MBPP lacks a clearly defined reasoning process. This is primarily due to the difficulty in breaking down Python code into distinct reasoning steps, complicating the definition of process rewards (Liangchen Luo et al., 2021).

In **SRA-MCTS** (Self-driven Reasoning Augmentation with Monte Carlo Tree Search for Code Generation — Bin et al., 2024), a similar data generation approach was investigated. One proposed solution involves prompting the LLM to articulate a step-by-step reasoning process before providing the code solution, thereby enforcing reasoning. While SRA-MCTS presents a feasible approach, it primarily focuses on online training (David Silver et al., 2017). Our method combines this concept with the innovative idea of **EP-PRM**, enabling the construction of an offline MCTS data generation process that enhances the reusability of generated data (Huaijie Wang et al., 2024).

## Data Generation Process

All main code for the MBPP-MCTS data generation is contained within the `MBPP` folder. The data generation process consists of the following steps:

1. **Right Solution Reasoning**: The LLM is prompted to construct and explain the reasoning based on the provided question and solution (see `generat_reasoning_from_train_data.py`). The reasoning with the highest likelihood of achieving the correct solution is identified.

2. **Completion Rollout**: The LLM generates completion rollouts for each step derived from the right solution reasoning (see `mc_reward_data_mbpp.py`).

3. **Score Calculation**: The LLM implements code according to the sampled reasoning to generate a solution for the given question. This process, termed "backpropagation" in MCTS terminology, assigns a score to each reasoning step based on its likelihood of leading to the correct solution (see `calculate_score_mbpp.py`).

### Results

Our findings demonstrate that utilizing reasoning steps derived from correct answers significantly enhances the likelihood of generating accurate code solutions. As we progress to reasoning generated solely from questions (without reference to the solutions), we observe a gradual decrease in scores.

## References

- Hanning Zhang, Pengcheng Wang, Shizhe Diao, Yong Lin, Rui Pan, Hanze Dong, Dylan Zhang, Pavlo Molchanov, Tong Zhang. *Entropy-Regularized Process Reward Model*. arXiv preprint arXiv:2412.11006.
- Yuxi Xie, Anirudh Goyal, Wenyue Zheng, Min-Yen Kan, Timothy P. Lillicrap, Kenji Kawaguchi, Michael Shieh. *Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning*. arXiv:2405.00451.
- Long Ouyang, et al. *Training language models to follow instructions with human feedback*. arXiv:2203.02155.
- Jacob Austin, et al. *Program Synthesis with Large Language Models*. arXiv:2108.07732.
- Peiyi Wang, et al. *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations*. arXiv:2312.08935.
- Hanze Dong, et al. *RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment*. arXiv:2304.06767.
- Liangchen Luo, et al. *Improve Mathematical Reasoning in Language Models by Automated Process Supervision*. arXiv:2406.06592.
- Bin Xu, et al. *SRA-MCTS: Self-driven Reasoning Augmentation with Monte Carlo Tree Search for Code Generation*. arXiv:2411.11053.
- Huaijie Wang, Shibo Hao, Hanze Dong, Shenao Zhang, Yilin Bao, Ziran Yang, Yi Wu. *Offline Reinforcement Learning for LLM Multi-Step Reasoning*. arXiv:2412.16145.
- David Silver, et al. *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*. arXiv:1712.01815.

### Original code
We borrow the environment setting from `https://github.com/RLHFlow/Online-RLHF`
This was originated from https://github.com/hanningzhang/ER-PRM
## Environment Setup

- clone and create a conda environment
```
conda create -n math python=3.10.9 --yes
conda activate math

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout d17fd7cd3b71c6a7bf7af34d8dc73135bb7ea8e9
pip3 install torch==2.1.2 torchvision torchaudio
python -m pip install .
pip install https://github.com/vllm-project/vllm/releases/download/v0.4.0/vllm-0.4.0-cp310-cp310-manylinux1_x86_64.whl
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install accelerate==0.27.2
pip install datasets deepspeed wandb stanza fraction jsonlines spacy
pip install --upgrade trl
conda install ninja

pip install wandb

export SSL_CERT_DIR='/etc/ssl/certs'
export REQUESTS_CA_BUNDLE='/etc/ssl/certs/ca-certificates.crt'
