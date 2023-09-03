# Tutorial: Extending LLMeBench
It is possible to extend the framework by at least one of the following components:
- [Dataset](#adding-dataset)
- [Task](#adding-task)
- [Model](#adding-model)
- [Defining a new asset](#benchmark-asset)
  - [Creating Few Shot Assets](#creating-few-shot-assets)


## Adding Dataset ([See Demo](https://youtu.be/_sO2PhKhKGA?feature=shared))
#### 
Check if the dataset used by your task already has an implementation in `llmebench/datasets`. If not, implement a new dataset module (e.g. `llmebench/datasets/SemEval23.py`), which implements a class (e.g. `SemEval23Dataset`) which subclasses `DatasetBase`. See [existing dataset modules](llmebench/datasets) for inspiration. Each new dataset class requires implementing four functions:

```python
class NewDataset(DatasetBase):
	def __init__(self, custom_param_1, custom_param_2, **kwargs):
		# custom_param_1/2 are passed from `dataset_args` in the benchmark
		# config
		...
		super(NewDataset, self).__init__(**kwargs)

	def metadata():
		# This method should return a dictionary that defines metadata describing
		# the dataset, like: citation or reference, download_url, language, etc.

	def get_data_sample(self):
		# This method should return a dictionary that represents the structure of
		# a single sample of the data for the purpose of testing and viewing
		# of NewDataset representation
	
	def load_data(self, data_path):
		# This function loads the data and must return a list of
		# dictionaries, where each dictionary has at least two keys:
		#   "input": this will be sent to the prompt generator
		#   "label": this will be used for evaluation
		#   "input_id": this optional key will be used for deduplication
```

**Note:** in case of few shots assets, the framework provides the functionality of deduplicating the training examples, from which few shots are being extracted, against the evaluatin dataset, based on sample IDs. To enable this functionality, `load_data` should also define `"input_id"` per input sample.

**Once the `Dataset` is implemented, export it in `llmebench/datasets/__init__.py`.**

## Adding Task 
#### ([See Demo](https://youtu.be/TN1bpWBpSTU?feature=shared))
Check if the task you are adding to the benchmark already has an implementation in `llmebench/tasks`. If not, implement a new task module (e.g. `llmebench/tasks/Sarcasm.py`), which implements a class (e.g. `SarcasmTask`) that subclasses `TaskBase`. See [existing task modules](llmebench/tasks) for inspiration. Each new task class requires implementing two functions:

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
		# post_process function. 
		# The framework expects this function to handle cases when
		# a predicted label is None. A suggested solution is assigning
		# a random prediction. Thus, it offers functions for random
		# prediction assigment to samples with None predictions
		# that should be called here. 
```

**Note:** In some cases, the model mightn't return a valid prediction for a given input sample, leading to `None` as the returned prediction. The framework expects the `evaluate` function to handle these cases. A suggested solution is assigning a random prediction. Thus, the framework offers functions for random prediction assigment (e.g., for classification and regression tasks) in the parent class [`TaskBase`](llmebench/tasks/task_base.py), that should be called in the `evaluate` function. 

**Once the `Task` is implemented, export it in `llmebench/tasks/__init__.py`.**

## Adding Model 
#### ([See Demo](https://youtu.be/J5H-BD8HQsk?feature=shared))
Next, check if the model you are trying to run the benchmark for has an implementation in `llmebench/models`. If not, implement a new model module (e.g. `llmebench/models/QARiB.py`), which implements a class (e.g. `QARiBModel`) which subclasses `ModelBase`. See an existing model module for inspiration. Each new model class requires implementing three functions:

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

    	def summarize_response(self, response):
        	# This function gets the full model response and must return the 
		# part of the model response that contains the answer to the prompt
```

Once the `Model` is implemented, export it in `llmebench/models/__init__.py`.

### Benchmark Asset 
#### ([See Demo](https://youtu.be/j6sA5u7LHYM?feature=shared))
Now that the Dataset, Task and Model are defined, the framework expects a given benchmark asset (e.g. "ArabGender" dataset, "GenderClassification" task, "GPT" model and "ZeroShot" prompting setting) to have a `*.py` file with three functions:

```python
def config():
	# This function returns a dictionary with the dataset, task and model the
	# current run is targeting along with arguments for each of these, as well
	# as a path to the dataset itself.

def prompt(input_sample):
	# This function receives an input_sample and pre-processes it into the
	# expected input for the model being used. For instance, GPTModel expects
	# its input to be a dictionary with two keys, ``system_message`` and a list
	# of ``messages`` with the ``sender`` and ``text`` in each message.
	# See the documentation linked with the available models for exact specifications

def post_process(response):
	# This function takes the output from the model, and post-processes it to extract
	# the actual prediction. The framework expects this function to return a valied prediction
	# in the expected format (or None if the model output cannot be parsed). The output of 
	# the function is matched with the gold label in a task's evaluation function.
```


### Creating Few Shot Assets
The framework has some preliminary support to **automatically** select `n` examples _per test sample_ based on a maximal marginal relevance-based approach (using [langchain's implementation](https://python.langchain.com/docs/modules/model_io/prompts/example_selectors/mmr)). This will be expanded in the future to have more few shot example selection mechanisms (e.g Random, Class based etc.). To define a few shot asset, we start from the same approach of [implementing an asset](#benchmark-asset), however, the config needs to be extended with the following keys to enable the few shot pipeline:

```python
"general_args": {
        "data_path": "...",
        # ...other general args
        "fewshot": {
            "train_data_path": "... path to train data ...",
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

**Note:** in case of few shots assets, the framework default behavior is to deduplicate the training examples, from which few shots are being extracted, against the evaluatin dataset, based on sample IDs. To enable this functionality: 
  1) `load_data` in the dataset to be used should also define `"input_id"` per input sample (See: [Adding Dataset](#adding-dataset))
  2)  `"deduplicate"` shouldn't be passed in `"fewshot": { ` or it should be set to True. 
