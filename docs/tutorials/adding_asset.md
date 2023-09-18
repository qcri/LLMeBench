<!---# Defining a new Benchmark Asset ([See Demo](https://youtu.be/j6sA5u7LHYM?feature=shared))-->
# Defining a new Benchmark Asset

Given a Dataset, Task and Model provider, the framework expects a given benchmark asset (e.g. "ArabGendDataset", "DemographyGenderTask", "OpenAIModel" and "ZeroShot" prompting setting) to have a `*.py` file with four functions:

```python
def metadata():
	# This function returns a dictionary with the "author", "model" and "description"
	# keys with string values. An optional "scores" key can also be included with
	# a dictionary value of `<metric>: <score>` pairs
def config():
	# This function returns a dictionary with the dataset, task and model provider
	# the current run is targeting along with arguments for each of these, as well
	# as a path to the dataset itself. The structure is as follows:
	# {
	# 	"dataset": llmebench.dataset.*Dataset,
	# 	"dataset_args": {}, # Optional, arguments passed to Dataset constructor
	# 	"task": llmebench.task.*Task,
	# 	"task_args": {}, # Optional, arguments passed to Task constructor
	# 	"model": llmebench.model.*Model,
	# 	"model_args": {}, # Optional, arguments passed to Model constructor
	# 	"general_args": { # Optional to define splits, fewshot settings
	# 		"test_split": "test-gold" # Optional, "test" will be picked from the dataset automatically if available
	# 		"fewshot": {
	# 			"train_split": "dev" # Optional, "train" will be picked from the dataset automatically if available
	# 			"deduplicate": False # Optional, enable/disable train/test deduplication
	# 		}
	# 	}
	# }

def prompt(input_sample):
	# This function receives an input_sample and pre-processes it into the
	# expected input for the model being used. For instance, OpenAIModel expects
	# its input to be a dictionary with two keys, ``system_message`` and a list
	# of ``messages`` with the ``sender`` and ``text`` in each message.
	# See the documentation linked with the available model providers for exact specifications

def post_process(response):
	# This function takes the output from the model, and post-processes it to extract
	# the actual prediction. The framework expects this function to return a valied prediction
	# in the expected format (or None if the model output cannot be parsed). The output of
	# the function is matched with the gold label in a task's evaluation function.
```

**Note:** When defining the prompt for an asset, the prompt structure must follow the structure required by the target model. See the documentation linked with the [available models](https://github.com/qcri/LLMeBench/tree/main/llmebench/models) for exact specifications.

## Creating Few Shot Assets
The framework has some preliminary support to **automatically** select `n` examples _per test sample_ based on a maximal marginal relevance-based approach (using [langchain's implementation](https://python.langchain.com/docs/modules/model_io/prompts/example_selectors/mmr)). This will be expanded in the future to have more few shot example selection mechanisms (e.g Random, Class based etc.). To define a few shot asset, we start from the same approach of [implementing an asset](#benchmark-asset), however, the config needs to be extended with the following keys to enable the few shot pipeline:

```python
"general_args": {
    "test_split": "... test split to use ..." # Optional, "test" will be picked from the dataset automatically if available
    # ...other general args
    "fewshot": {
        "train_split": "... train split to use ...", # Optional, "train" will be picked from the dataset automatically if available
        "deduplicate": False, # Optional parameter. Default is True
    },
},
```

and the prompt function needs to accept two parameters, and should return the full prompt including few shots:

```python
def prompt(input_sample, examples):
	# "examples" will contain the few shots samples selected
	# for this particular test sample
	# this function should also handle creating the prompt including few shots
```

**Note:** in case of few shots assets, the framework default behavior is to deduplicate the training examples, from which few shots are being extracted, against the evaluation dataset, based on sample IDs. To enable this functionality:
  1) `load_data` in the dataset to be used should also define `"input_id"` per input sample (See: [Adding Dataset](docs/tutorials/adding_dataset.md))
  2)  `"deduplicate"` shouldn't be passed in `"fewshot": { ` or it should be set to True.
