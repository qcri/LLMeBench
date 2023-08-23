# LLMeBench: A Flexible Framework for Accelerating LLMs Benchmarking

This repository contains code for the LLMeBench framework (described in [this paper](https://arxiv.org/abs/2308.04945)). The framework currently supports evaluation of a variety of NLP tasks using OpenAI's GPT and BLOOM models; it can be seamlessly customized for any NLP task, model and dataset, regardless of language. 

<p align="center">
<picture>
<img alt = "The architecture of the LLMeBench framework." src="https://github.com/qcri/LLMeBench/assets/3918663/14aebcc4-80ed-4b90-b20d-72b08ba4c909" width="400" height="140"/>
</picture>
</p>

#### Summary and examples of the datasets, tasks, models and metrics currently implemented
<p align="center">
<picture>
<img alt = "Summary and examples of the 53 datasets, 31 tasks, 3 models and metrics currently implemented and
validated in LLMeBench." src="https://github.com/qcri/LLMeBench/assets/3918663/72c8b59e-5f43-4437-b493-5a52690e5a10" width="470" height="160"/>
</picture>
</p>

## Installation
*pip package to be made available soon!*

Clone this repository:
```bash
git clone https://github.com/qcri/LLMeBench.git
cd LLMeBench
```

Create a virtual environment:
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

## Running the benchmark
A sample benchmark is available in `assets/benchmark_v1`. To run the benchmark,

```bash
python -m llmebench --filter '*benchmarking_asset*' <benchmark-dir> <results-dir> --filter 
```
#### Parameters
- `--filter '*benchmarking_asset*'`: This flag indicates specific tasks in the benchmark to run. The framework will run a wildecard search using '*benchmarking_asset*'.
- `<benchmark-dir>`: Path of directory where the benchmarking asset to run can be found.
- `<results-dir>`: Path of directory where to save output results, along with intermediate cached values.
- You might need to also define environment variables such as `AZURE_API_URL` and `AZURE_API_KEY` depending on the benchmark you are running. This can be done by either:
   - `export AZURE_API_KEY="..."` _before_ running the above command, or
   - prepending `AZURE_API_URL="..." AZURE_API_KEY="..."` to the above command.

#### Outputs format
- `<results-dir>`:
- 
## Adding a new task
Before adding a new task, make sure you have the latest changes:

```bash
git pull
```

Create a new branch for your task
```bash
git checkout -b feat/sarcasm_task
```

### Dataset
Check if the dataset used by your task already has an implementation in `llmebench/datasets`. If not, implement a new dataset module (e.g. `llmebench/datasets/SemEval23.py`), which implements a class (e.g. `SemEval23Dataset`) which subclasses `DatasetBase`. See an existing dataset module for inspiration. Each new dataset class requires implementing three functions:

```python
class NewDataset(DatasetBase):
	def __init__(self, custom_param_1, custom_param_2, **kwargs):
		# custom_param_1/2 are passed from `dataset_args` in the benchmark
		# config
		...
		super(NewDataset, self).__init__(**kwargs)

	def citation():
		# This function returns a string with the bib entry for the dataset

	def load_data(self, data_path):
		# This function loads the data and _must_ return a list of
		# dictionaries, where each dictionary has atleast two keys
		#   "input": this will be sent to the prompt generator
		#   "label": this will be used for evaluation
```

Once the `Dataset` is implemented, export it in `llmebench/datasets/__init__.py`.

### Task
Check if the task you are adding to the benchmark already has an implementation in `llmebench/tasks`. If not, implement a new dataset module (e.g. `llmebench/tasks/Sarcasm.py`), which implements a class (e.g. `SarcasmTask`) which subclasses `TaskBase`. See an existing task module for inspiration. Each new task class requires implementing two functions:

```python
class NewTask(TaskBase):
	def __init__(self, custom_param_1, custom_param_2, **kwargs):
		# custom_param_1/2 are passed from `task_args` in the benchmark
		# config
		...
		super(NewTask, self).__init__(**kwargs)

	def evaluate(self, true_labels, predicted_labels):
		# This function gets two lists, the `true_labels` from the
		# dataset loader, and `predicted_labels` from the
		# post_process function
```

Once the `Task` is implemented, export it in `llmebench/tasks/__init__.py`.

### Model
Next, check if the model you are trying to run the benchmark for has an implementation in `llmebench/models`. If not, implement a new model module (e.g. `llmebench/models/QARiB.py`), which implements a class (e.g. `QARiBModel`) which subclasses `ModelBase`. See an existing model module for inspiration. Each new model class requires implementing two functions:

```python
class NewModel(TaskBase):
	def __init__(self, custom_param_1, custom_param_2, **kwargs):
		# custom_param_1/2 are passed from `model_args` in the benchmark
		# config
		...
		super(NewModel, self).__init__(**kwargs)

	def prompt(self, **kwargs):
		# This function gets the pre-processed input and must
		# run the actual model and return model outputs
```

Once the `Model` is implemented, export it in `llmebench/models/__init__.py`.

### Benchmark Asset
Now that the Dataset, Task and Model are defined, the framework expects a given benchmark asset (e.g. "ArabGender" dataset, "GenderClassification" task, "GPT" model and "ZeroShot" prompting setting) to have a `*.py` file with three functions:

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

### Testing
The benchmarking module allows one to run a specific asset instead of the entire benchmark using the `--filter` option. It is also a good idea to use the `--limit` option to limit the tests to few (e.g. 5 samples). Sample command below:

```bash
python -m llmebench --filter 'demography/gender/AraGend_ChatGPT_ZeroShot' --limit 5 --ignore_cache <benchmark-dir> <results-dir>
```

Make sure to also run `scripts/run_tests.sh` before submitting your code, and once you are ready, you can commit your changes locally and push them to a remote branch:

```bash
git push origin feat/sarcasm_task
```

and open a _pull request_ by going to the [repository webpage](https://github.com/qcri/Arabic_LLM_Benchmark)

### Creating Few Shot Assets
The framework has some preliminary support to automatically select N examples per test sample based on sentence similarity (using langchain's implementation). This will be expanded in the future to have more few shot example selection mechanism (e.g Random, Class based etc.). For now, a config needs to have the following keys to enable the few shot pipeline:

```python
"general_args": {
        "data_path": "...",
        # ...other general args
        "fewshot": {
            "train_data_path": "... path to train data ...",
        },
    },
```

and the prompt function needs to accept two parameters:

```python
def prompt(input_sample, examples):
	# "examples" will contain the few shots samples selected
	# for this particular test sample
```

To run the actual few shot assets, supply the `--n_shots <n>` option to the benchmarking script. This is set to 0 by default and will run only zero shot assets. If `--nshots` is set to something greater than zero, only few shot assets are run.
