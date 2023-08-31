# LLMeBench: A Flexible Framework for Accelerating LLMs Benchmarking

This repository contains code for the [LLMeBench framework](https://youtu.be/FkQn4UjYA0s?feature=shared) (described in <a href="https://arxiv.org/abs/2308.04945" target="_blank">this paper</a>). The framework currently supports evaluation of a variety of NLP tasks using OpenAI's GPT and BLOOM models; it can be seamlessly customized for any NLP task, LLM model and dataset, regardless of language.

<p align="center">
<picture>
<img alt = "The architecture of the LLMeBench framework." src="https://github.com/qcri/LLMeBench/assets/3918663/f1b927ea-fb7f-4dc6-b654-7c141f596067" width="400" height="140"/>
</picture>
</p>

## Overview
<p align="center">
<picture>
<img alt = "Summary and examples of the 53 datasets, 31 tasks, 3 models and metrics currently implemented and
validated in LLMeBench." src="https://github.com/qcri/LLMeBench/assets/3918663/a9b926c0-8a10-4334-84b2-ad0b4e3e5ceb" width="470" height="140"/>
</picture>
</p>

- Currently supports 31 [tasks](llmebench/tasks) featuring 3 [models](llmebench/models). Rigorously tested with 53 [datasets](llmebench/datasets) associated with 11 languages.
- Easly extendible to new models accessible through APIs.
- Extensive caching capabilities, to avoid costly API re-calls for repeated experiments.
- Supports zero- and few-shot learning paradigms.
- Open-source.

## Quick Start!
1. [Install](https://github.com/qcri/LLMeBench/tree/readme_update1#installation) LLMeBench.
2. [Get the data](https://github.com/qcri/LLMeBench/tree/readme_update1#get-the-benchmark-data).
3. Evaluate!
   
   To evaluate a randome baseline performance for one task (e.g., Sentiment analysis) and one dataset, you need to run what we refer to as an "asset" that specifies: dataset, model and task to evaluate as follows:
```bash
python -m llmebench --filter '*ArSAS_Random*' assets/benchmark_v1/sentiment/ results/ 
```
where `ArSAS` is the dataset name, `Random` is the model name and `assets/benchmark_v1/sentiment/` is the directory where the asset for the sentiment analysis task can be found. 

## Installation
*pip package to be made available soon!*

Clone this repository:
```bash
git clone https://github.com/qcri/LLMeBench.git
cd LLMeBench
```

Create and activate virtual environment:
```bash
python -m venv .envs/llmebench
source .envs/llmebench/bin/activate
```

Install the dependencies and benchmarking package:
```bash
pip install -e '.[dev,fewshot]'
```

## Get the benchmark data
Download the benchmark from [here](https://neurox.qcri.org/projects/llmebench/arabic_llm_benchmark_data.zip), and unzip it into the `Arabic_LLM_Benchmark` folder. After this process, there should be a `data` directory inside the top-level folder of the repository, with roughly the following contents:

```bash
$ ls data/
MT
STS
XNLI
demography
factuality_disinformation_harmful_content
sentiment_emotion_others
sequence_tagging_ner_pos_etc
speech
```

## Usage
To run the benchmark,

```bash
python -m llmebench --filter '*benchmarking_asset*' --limit <k> --n_shots <n> --ignore_cache <benchmark-dir> <results-dir> 
```

#### Parameters
- `--filter '*benchmarking_asset*'`: **(Optional)** This flag indicates specific tasks in the benchmark to run. The framework will run a wildecard search using '*benchmarking_asset*'. If not set, the framework will run the entire benchmark.
- `--limit <k>`: **(Optional)** Specify the number of samples from input data to run through the pipeline, to allow effecient testing.
- `--n_shots <n>`: **(Optional)** If defined, framework will expect a few shot asset and will run the few shots learning paradigm, setting `n` as the number of shots.
- `--ignore_cache`: **(Optional)** A flag to ignore loading and saving intermediate model responses from/to cache. 
- `<benchmark-dir>`: Path of directory where the benchmarking assets to run can be found.
- `<results-dir>`: Path of directory where to save output results, along with intermediate cached values.
- You might need to also define environment variables such as `AZURE_API_URL` and `AZURE_API_KEY` depending on the benchmark you are running. This can be done by either:
   - `export AZURE_API_KEY="..."` _before_ running the above command, or
   - prepending `AZURE_API_URL="..." AZURE_API_KEY="..."` to the above command.

#### Outputs format
- `<results-dir>`: The framework will create a sub-folder per benchmarking asset in this directory. A sub-folder will contain:
  - **_n.json_**: A file per dataset sample, where *n* indicates sample order in the dataset input file. This file contains input sample, full prompt sent to the model, full model response, and the model output after post-processing as defined in the asset file.
  - **_summary.jsonl_**: Lists all input samples that successfuly ran through the pipeline, and for each, the raw model prediction, and the post-processed model prediction.
  -  **_summary_failed.jsonl_**: Lists all input samples that didn't get a successful response from the model, in addition to output model's reason behind failure.
  -  **_results.json_**: Contains a summary on number of processed and failed input samples, and evaluation results.

#### Caching
The framework provides caching (if `--ignore_cache` isn't passed), to enable the following: 
- Allowing users to bypass making API calls for items that have already been successfully processed.
- Enhancing the post-processing of the models’ output, as post-processing can be performed repeatedly without having to call the API every time. 

#### Running Few Shot Assets
The framework has some preliminary support to automatically select `n` examples _per test sample_ based on a maximal marginal relevance-based approach (using [langchain's implementation](https://python.langchain.com/docs/modules/model_io/prompts/example_selectors/mmr)). This will be expanded in the future to have more few shot example selection mechanisms (e.g Random, Class based etc.).

To run few shot assets, supply the `--n_shots <n>` option to the benchmarking script. This is set to 0 by default and will run only zero shot assets. If `--n_shots` is > zero, only few shot assets are run.

## Tutorial
It is possible to extend the framework by at least one of the following components. Details on implementing each can be found in [Tutorial README file]():
- Model
- Task
- Dataset
- Asset

## Citation
Please cite our paper when referring to this framework!
```
@misc{dalvi2023llmebench,
      title={LLMeBench: A Flexible Framework for Accelerating LLMs Benchmarking}, 
      author={Fahim Dalvi and Maram Hasanain and Sabri Boughorbel and Basel Mousi and Samir Abdaljalil and Nizi Nazar and Ahmed Abdelali and Shammur Absar Chowdhury and Hamdy Mubarak and Ahmed Ali and Majd Hawasly and Nadir Durrani and Firoj Alam},
      year={2023},
      eprint={2308.04945},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
