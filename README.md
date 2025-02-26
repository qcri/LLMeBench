# LLMeBench: A Flexible Framework for Accelerating LLMs Benchmarking

This repository contains code for the LLMeBench framework (described in <a href="https://aclanthology.org/2024.eacl-demo.23/" target="_blank">this paper</a>). The framework currently supports evaluation of a variety of NLP tasks using **three** model providers: OpenAI (e.g., [GPT](https://platform.openai.com/docs/guides/gpt)), [HuggingFace Inference API](https://huggingface.co/docs/api-inference/), and Petals (e.g., [BLOOMZ](https://huggingface.co/bigscience/bloomz)); it can be seamlessly customized for any NLP task, LLM model and dataset, regardless of language.

<!---"https://github.com/qcri/LLMeBench/assets/3918663/15d989e0-edc7-489a-ba3b-36184a715383"--->

<p align="center">
<picture>
<img alt = "The architecture of the LLMeBench framework." src="https://github.com/qcri/LLMeBench/assets/3918663/7f7a0da8-cd73-49d5-90d6-e5c62781b5c3" width="400" height="250"/>
</picture>
</p>

## Recent Updates
- 20st January, 2025 -- New assets added. Updated versions for openai, anthropic and sentence_transformers.
- 21st July, 2024 -- Multimodal capabilities have been added. Assets now include support for GPT-4 (OpenAI) and Sonet (Anthropic).

## Overview
<p align="center">
<picture>
<img alt = "Summary and examples of the 53 datasets, 31 tasks, 3 model providers and metrics currently implemented and
validated in LLMeBench." src="https://github.com/qcri/LLMeBench/assets/3918663/8a0ddf60-5d2f-4e8c-a7d9-de37cdeac104" width="510" height="160"/>
</picture>
</p>

Developing **LLMeBench** is an ongoing effort and it will be continuously expanded. Currently, the framework features the following:
- Supports 34 [tasks](llmebench/tasks) featuring 7 [model providers](llmebench/models). Tested with 66 [datasets](llmebench/datasets) associated with 16 languages, resulting in **800 [benchmarking assets](assets/)** ready to run.
- Support for **text, speech, and multimodality**
- Easily extensible to new models accessible through APIs.
- Extensive caching capabilities, to avoid costly API re-calls for repeated experiments.
- Supports zero- and few-shot learning paradigms.
- On-the-fly datasets download and dataset caching.
- Open-source.

## Quick Start!
1. Install LLMeBench: `pip install 'llmebench[fewshot]'`
2. Download the current assets: `python -m llmebench assets download`. This will fetch assets and place them in the current working directory.
3. Download one of the dataset, e.g. ArSAS. `python -m llmebench data download ArSAS`. This will download the data to the current working directory inside the `data` folder.
4. Evaluate!

   For example, to evaluate the performance of a random baseline for Sentiment analysis on [ArSAS dataset](https://github.com/qcri/LLMeBench/blob/main/llmebench/datasets/ArSAS.py), you can run:
   ```bash
   python -m llmebench --filter 'sentiment/ArSAS_Random*' assets/ results/
   ```
   which uses the [ArSAS_random "asset"](assets/ar/sentiment_emotion_others/sentiment/ArSAS_random.py): a file that specifies the dataset, model and task to evaluate. Here, `ArSAS_Random` is the asset name referring to the `ArSAS` dataset name and the `Random` model, and `assets/ar/sentiment_emotion_others/sentiment/` is the directory where the benchmarking asset for the sentiment analysis task on Arabic ArSAS dataset can be found. Results will be saved in a directory called `results`.

## Get the Benchmark Data
In addition to supporting the user to implement their own LLM evaluation and benchmarking experiments, the framework comes equipped with benchmarking assets over a large variety of datasets and NLP tasks. To benchmark models on the same datasets, the framework *automatically* downloads the datasets when possible. Manually downloading them (for example to explore the data before running any assets) can be done as follows:

```bash
python -m llmebench data download <DatasetName>
```

**_Voilà! all ready to start evaluation..._**

**Note:** Some datasets and associated assets are implemented in LLMeBench but the dataset files can't be re-distributed, it is the responsibility of the framework user to acquire them from their original sources. The metadata for each `Dataset` includes a link to the primary page for the dataset, which can be used to obtain the data. The data should be downloaded and present in a folder under `data/<DatasetName>`, where `<DatasetName>` is the same as implementation under `llmebench.datasets`. For instance, the `ADIDataset` should have it's data under `data/ADI/`.

**Disclaimer:** The datasets associated with the current version of LLMeBench are either existing datasets or processed versions of them. We refer users to the original license accompanying each dataset as provided in the metadata for [each dataset script](https://github.com/qcri/LLMeBench/tree/main/llmebench/datasets). It is our understanding that these licenses allow for datasets use and redistribution for research or non-commercial purposes .

## Usage
To run the benchmark,

```bash
python -m llmebench --filter '*benchmarking_asset*' --limit <k> --n_shots <n> --ignore_cache <benchmark-dir> <results-dir>
```

#### Parameters
- `--filter '*benchmarking_asset*'`: **(Optional)** This flag indicates specific tasks in the benchmark to run. The framework will run a wildcard search using '*benchmarking_asset*' in the assets directory specified by `<benchmark-dir>`. If not set, the framework will run the entire benchmark.
- `--limit <k>`: **(Optional)** Specify the number of samples from input data to run through the pipeline, to allow efficient testing. If not set, all the samples in a dataset will be evaluated.
- `--n_shots <n>`: **(Optional)** If defined, the framework will expect a few-shot asset and will run the few-shot learning paradigm, with `n` as the number of shots. If not set, zero-shot will be assumed.
- `--ignore_cache`: **(Optional)** A flag to ignore loading and saving intermediate model responses from/to cache.
- `<benchmark-dir>`: Path of the directory where the benchmarking assets can be found.
- `<results-dir>`: Path of the directory where to save output results, along with intermediate cached values.
- You might need to also define environment variables (like access tokens and API urls, e.g. `AZURE_API_URL` and `AZURE_API_KEY`) depending on the benchmark you are running. This can be done by either:
   - `export AZURE_API_KEY="..."` _before_ running the above command, or
   - prepending `AZURE_API_URL="..." AZURE_API_KEY="..."` to the above command.
   - supplying a dotenv file using the `--env` flag. Sample dotenv files are provided in the `env/` folder
   - Each [model provider's](llmebench/models) documentation specifies what environment variables are expected at runtime.

#### Outputs Format
`<results-dir>`: This folder will contain the outputs resulting from running assets. It follows this structure:
- **all_results.json**: A file that presents summarized output of all assets that were run where `<results-dir>` was specified as the output directory.
- The framework will create a sub-folder per benchmarking asset in this directory. A sub-folder will contain:
  - **_n.json_**: A file per dataset sample, where *n* indicates sample order in the dataset input file. This file contains input sample, full prompt sent to the model, full model response, and the model output after post-processing as defined in the asset file.
  - **_summary.jsonl_**: Lists all input samples, and for each, a summarized model prediction, and the post-processed model prediction.
  -  **_summary_failed.jsonl_**: Lists all input samples that didn't get a successful response from the model, in addition to output model's reason behind failure.
  -  **_results.json_**: Contains a summary on number of processed and failed input samples, and evaluation results.
- For few shot experiments, all results are stored in a sub-folder named like **_3_shot_**, where the number signifies the number of few shots samples provided in that particular experiment

[jq](https://jqlang.github.io/jq/) is a helpful command line utility to analyze the resulting json files. The simplest usage is `jq . summary.jsonl`, which will print a summary of all samples and model responses in a readable form.

#### Caching
The framework provides caching (if `--ignore_cache` isn't passed), to enable the following:
- Allowing users to bypass making API calls for items that have already been successfully processed.
- Enhancing the post-processing of the models’ output, as post-processing can be performed repeatedly without having to call the API every time.

#### Running Few Shot Assets
The framework has some preliminary support to automatically select `n` examples _per test sample_ based on a maximal marginal relevance-based approach (using [langchain's implementation](https://python.langchain.com/docs/modules/model_io/prompts/example_selectors/mmr)). This will be expanded in the future to have more few shot example selection mechanisms (e.g Random, Class based etc.).

To run few shot assets, supply the `--n_shots <n>` option to the benchmarking script. This is set to 0 by default and will run only zero shot assets. If `--n_shots` is > zero, only few shot assets are run.

## Tutorial
The [tutorials directory](docs/tutorials/) provides tutorials on the following: updating an existing asset, advanced usage commands to run different benchmarking use cases, and extending the framework by at least one of these components:
- Model Provider
- Task
- Dataset
- Asset

## Citation
Please cite our papers when referring to this framework:

```
@inproceedings{abdelali-2024-larabench,
  title = "{{LAraBench}: Benchmarking Arabic AI with Large Language Models}",
  author ={Ahmed Abdelali and Hamdy Mubarak and Shammur Absar Chowdhury and Maram Hasanain and Basel Mousi and Sabri Boughorbel and Samir Abdaljalil and Yassine El Kheir and Daniel Izham and Fahim Dalvi and Majd Hawasly and Nizi Nazar and Yousseif Elshahawy and Ahmed Ali and Nadir Durrani and Natasa Milic-Frayling and Firoj Alam},
  booktitle = {Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers},
  month = mar,
  year = {2024},
  address = {Malta},
  publisher = {Association for Computational Linguistics},
}

@article{dalvi2023llmebench,
      title={{LLMeBench}: A Flexible Framework for Accelerating LLMs Benchmarking},
      author={Fahim Dalvi and Maram Hasanain and Sabri Boughorbel and Basel Mousi and Samir Abdaljalil and Nizi Nazar and Ahmed Abdelali and Shammur Absar Chowdhury and Hamdy Mubarak and Ahmed Ali and Majd Hawasly and Nadir Durrani and Firoj Alam},
      booktitle = {Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations},
      month = mar,
      year = {2024},
      address = {Malta},
      publisher = {Association for Computational Linguistics},
}

```

Please consider citing the following papers if you use the assets derived from them.

```
@inproceedings{kmainasi2024native,
  title={Native vs non-native language prompting: A comparative analysis},
  author={Kmainasi, Mohamed Bayan and Khan, Rakif and Shahroor, Ali Ezzat and Bendou, Boushra and Hasanain, Maram and Alam, Firoj},
  booktitle={International Conference on Web Information Systems Engineering},
  pages={406--420},
  year={2024},
  organization={Springer}
}

@article{hasan2024nativqa,
  title={{NativQA}: Multilingual culturally-aligned natural query for {LLMs}},
  author={Hasan, Md Arid and Hasanain, Maram and Ahmad, Fatema and Laskar, Sahinur Rahman and Upadhyay, Sunaya and Sukhadia, Vrunda N and Kutlu, Mucahid and Chowdhury, Shammur Absar and Alam, Firoj},
  journal={arXiv preprint arXiv:2407.09823},
  year={2024}
}
```
