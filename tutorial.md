# Tutorial: Extending LLMeBench
It is possible to extend the framework by at least one of the following components:
- [Dataset](#adding-dataset)
- [Task](#adding-dataset)
- [Model](#adding-dataset)
- [Defining a new asset](#benchmark-asset)
  - [Creating Few Shot Assets](#creating-few-shot-assets)


## Adding Dataset
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

## Adding Task
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

## Adding Model
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


### Creating Few Shot Assets
The framework has some preliminary support to automatically select `<n>` examples per test sample based on sentence similarity (using langchain's implementation). This will be expanded in the future to have more few shot example selection mechanism (e.g Random, Class based etc.). For now, a config needs to have the following keys to enable the few shot pipeline:

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
