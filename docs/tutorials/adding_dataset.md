<!----# Adding Dataset ([See Demo](https://youtu.be/_sO2PhKhKGA?feature=shared)) --->
# Adding Dataset

Check if the dataset used by your task already has an implementation in `llmebench/datasets`. If not, implement a new dataset module (e.g. `llmebench/datasets/SemEval23.py`), which implements a class (e.g. `SemEval23Dataset`) which subclasses `DatasetBase`. See [existing dataset modules](https://github.com/qcri/LLMeBench/tree/main/llmebench/datasets) for inspiration. Each new dataset class requires implementing four functions:

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

**Notes:** 
- In case of few shots assets, the framework provides the functionality of deduplicating the training examples, from which few shots are being extracted, against the evaluatin dataset, based on sample IDs. To enable this functionality, `load_data` should also define `"input_id"` per input sample.
- Further details on how to implement each function for a dataset can be found [here](https://github.com/qcri/LLMeBench/blob/main/llmebench/datasets/dataset_base.py).

**Once the `Dataset` is implemented, export it in `llmebench/datasets/__init__.py`.**
