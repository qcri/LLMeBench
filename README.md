# Arabic LLM Benchmark

## Installation
*pip package to me made available soon!*

Clone this repository:
```bash
git clone https://github.com/qcri/Arabic_LLM_Benchmark.git
cd Arabic_LLM_Benchmark
```

Create a virtual environment:
```bash
python -m venv .envs/arabic_llm_benchmark
source .envs/arabic_llm_benchmark/bin/activate
```

Install the dependencies and benchmarking package:
```bash
pip install -e '.[dev]'
```

## Get the benchmark data
Download the benchmark from [here](https://neurox.qcri.org/projects/arabic_llm_benchmark/arabic_llm_benchmark_data.zip), and unzip in into the `Arabic_LLM_Benchmark` folder. After this process, there should be a `data` directory inside the top-level folder of the repository, with roughly the following contents:

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

## Running the benchmark
A sample benchmark is available in `assets/benchmark_v1`. To run the benchmark,

```bash
python -m arabic_llm_benchmark <benchmark-dir> <results-dir>
```

where `<benchmark-dir>` can point to `assets/benchmark_v1` for example. The
actual results will be saved in `<results-dir>`, along with intermediate cached
values.

## Adding a new task
The framework expects a given run to have a `*.py` file with three functions:

```python
def config():
	# This function returns a dictionary with the dataset, task and model the
	# current run is targeting along with arguments for each of these, as well
	# as a path to the dataset itself.

def prompt(input_sample):
	# This function receives an input_sample and pre-processes it into the
	# expected input for the model being uses. For instance, GPTModel expects
	# its input to be a dictionary with two keys, ``system_message`` and a list
	# of ``messages`` with the ``sender`` and ``text`` in each message.
	# See the documentation linked with the available models for exact specifications

def post_process(response):
	# This function takes the output from the model, and post-processes it to
	# extract the actual prediction. The framework expects this function to
	# return one of the labels (or None if the model output cannot be parsed
	# into a label). The output of the function is matched with the gold label
	# in a task's evaluation function.
```

Once also needs to implement the dataset loader under
`arabic_llm_benchmark/datasets` if not already present, and the same for the
a task handler under `arabic_llm_benchmark/tasks`.