<div align="center">
<h1> 🛠️ ToolBeHonest: A Multi-level Hallucination Diagnostic Benchmark for Tool-Augmented Large Language Models
 </h1>
</div>

<div align="center">

![Data License](https://img.shields.io/badge/Data%20License-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10.0-blue.svg)
</div>

<div align="center">
  <!-- <a href="#model">Model</a> • -->
  🏆 <a href="https://toolbehonest.github.io/">Leaderboard</a> |
  📃 <a href="https://arxiv.org/abs/xxxx">Paper</a> |
  📚 <a href="https://huggingface.co/datasets/Joelzhang/ToolBeHonest">Data</a> |
  📜 <a href="https://github.com/ToolBeHonest/ToolBeHonest/blob/main/LICENSE">License</a>
</div>

## 📝 Introduction

ToolBeHonest aims at diagnosing **hallucination issues** in large language models (LLMs) that are augmented with tools for real-world applications. We utilize a comprehensive diagnostic approach to assess LLMs' hallucinations through multi-level diagnostic processes and various toolset scenarios.

ToolBeHonest includes **700 manually annotated evaluation samples** across seven distinct tasks, representing scenarios where tools might be missing, potentially available, or have limited functionality. The benchmark spans tasks such as solvability detection, solution planning, and missing-tool analysis, ensuring a broad and deep evaluation.

### Diagnostic Framework and Scope

- **In-depth Multi-Level Diagnostic Process**:
  - **Solvability Detection**: Evaluates whether the problem can be solved with the given toolset.
  - **Solution Planning**: Requires LLMs to propose a sequential plan to address the task.
  - **Missing-Tool Analysis**: Identifies the necessary tools and their functionalities for unsolvable tasks.

- **In-breadth Hallucination-inducing Scenarios**:
  - **Missing Necessary Tools**: Tests the LLM's performance when key tools are unavailable, inducing the LLM to attempt using non-existent tools. It evaluates the LLM's strategy and reliability when essential tools are absent.
  - **Induced Use of Potential Tools**: Examines the LLM's ability to incorrectly utilize potential tools that should be avoided. For example, even though operating system environments may be available, they should not be used by the LLM, as their usage could lead to serious consequences.
  - **Misuse of Limited Functionality Tools**: In this scenario, LLMs are induced to directly use tools with limited capabilities for task execution, testing the LLM's handling capacity and recognition of real-world limitations when tools are insufficient.

<div align="center">
<img src="./assets/main_graph.png" style="width: 100%;height: 100%">
</div>

## 🎉 What's New

- **[2024.06.30]** 📣 ToolBeHonest Benchmark is released.

## 📄 Table of Contents

<details>
<summary>
Click to expand the table of contents
</summary>

- [📝 Introduction](#-introduction)
  - [Diagnostic Framework and Scope](#diagnostic-framework-and-scope)
- [🎉 What's New](#-whats-new)
- [📄 Table of Contents](#-table-of-contents)
- [🔧 Setup Environment](#-setup-environment)
- [📚 Data](#-data)
- [📊 Evaluate Benchmark](#-evaluate-benchmark)
  - [1. Inference](#1-inference)
  - [2. Evaluation](#2-evaluation)
- [🔁 Reproduction](#-reproduction)
- [📖 Citation](#-citation)

</details>

## 🔧 Setup Environment

```shell
conda create -n toolbh python=3.10
conda activate toolbh
pip3 install -r requirements.txt
```

## 📚 Data

You can download the whole evaluation data through the following command from huggingface:

```shell
cd toolbh
mkdir data
cd data

wget https://huggingface.co/datasets/Joelzhang/ToolBeHonest/resolve/main/test_en.json

# We also provide the Chinese version of the data.
wget https://huggingface.co/datasets/Joelzhang/ToolBeHonest/resolve/main/test_zh.json
```

## 📊 Evaluate Benchmark

### 1. Inference

Example script for `gemini-1.5-pro`:

```shell
cd toolbh

# Replace "--api_key your_api_key" to your own Google AI Studio APIKey
bash scripts/infer_gemini.sh
```

Example script for `gpt-4o-2024-05-13`:

```shell
cd toolbh

# Replace "--api_key your_api_key" to your own Openai APIKey
bash scripts/infer_openi.sh
```

### 2. Evaluation

Example script for evaluating the result of `gpt-4o-2024-05-13`:

```shell
cd toolbh

bash scripts/eval_results_single.sh
```

Example script for evaluating the all the results:

```shell
cd toolbh

bash scripts/eval_results_dictory.sh
```

Evaluation process will provide 2 evaluation results:

1. Eval_results: `(model)_(embedding_model)_hard.json` provide the detailed scores in samples-level for each subtasks for each level of the `model` using the `embedding_model`.
2. Table_results:

   - `Analysis_table_(model)_(embedding_model)_hard.txt`: Provide the detailed number of five error types for each hallucination-inducing scenario for Level-2 and Level-3 for `model` using the `embedding_model`.
   - `Scenario_table_(model)_(embedding_model)_hard.txt`: Provide the in-breadth hallucination-inducing scenario-leve l score and overall score for `model` using the `embedding_model`.
   - `Subtask_table_(model)_(embedding_model)_hard.txt`: Provide the score for each subtask for `model` using the `embedding_model`.

## 🔁 Reproduction

If you want to reproduce the results shown in the paper, download the reproduction results [`20240609_reproduction_results.tgz`](https://drive.google.com/file/d/11k_U-8BnBxqApU4C5xheAzOvmsNX5MLW/view?usp=sharing) and the reproduction embeddings[`20240609_reproduction_embedding.tgz`](https://drive.google.com/file/d/1WOxmZDEru5KtYwstiDSJeeeDuJWczZHc/view?usp=sharing), and run the following command.

```shell
cd toolbh
mkdir results
cd results

# Put the 20240609_reproduction_results.tgz here, and unzip it.
tar -zxvf 20240609_reproduction_results.tgz

cd ..
mkdir tools_emb

# Put the 20240609_reproduction_embedding.tgz here, and unzip it.
tar -zxvf 20240609_reproduction_embedding.tgz

cd ..

bash scripts/eval_results_reproduction.sh
```

The evaluation results will be saved in `toolbh/results/eval_results` and `toolbh/results/table_results`.

The overall results for each model can be found in `toolbh/results/eval_results/Evaluation_results_0609_(model)_(embedding_model)_hard.json`, `group_results` key contains the overall scores.

## 📖 Citation

If you find this repository useful, please consider giving star and citing our paper:

```plaintext
@article{zhang2024toolbehonestmultilevelhallucinationdiagnostic,
  title={ToolBeHonest: A Multi-level Hallucination Diagnostic Benchmark for Tool-Augmented Large Language Models},
  author={Yuxiang Zhang and Jing Chen and Junjie Wang and Yaxin Liu and Cheng Yang and Chufan Shi and Xinyu Zhu and Zihao Lin and Hanwen Wan and Yujiu Yang and Tetsuya Sakai and Tian Feng and Hayato Yamana},
  journal={arXiv preprint arXiv:2406.20015},
  year={2024},
  url={https://arxiv.org/abs/2406.20015},
}
```
