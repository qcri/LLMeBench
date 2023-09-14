# Adding Model Provider ([See Demo](https://youtu.be/J5H-BD8HQsk?feature=shared))
Next, check if the model provider you are trying to run the benchmark for has an implementation in `llmebench/models`. If not, implement a new model provider module (e.g. `llmebench/models/FastChat.py`), which implements a class (e.g. `FastChatModelBase`) which subclasses `ModelBase`. See an existing model providers module for inspiration. Each new model class requires implementing three functions:

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
