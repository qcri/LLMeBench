from abc import ABC, abstractmethod

class TaskBase(ABC):
	def __init__(self, dataset, **kwargs):
		self.dataset = dataset

	def load_data(self, data_path):
		return self.dataset.load_data(data_path)

	@abstractmethod
	def evaluate(self, true_labels, predicted_labels):
		pass