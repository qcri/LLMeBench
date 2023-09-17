<!---# Adding Task ([See Demo](https://youtu.be/TN1bpWBpSTU?feature=shared)) -->
# Adding Task
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
