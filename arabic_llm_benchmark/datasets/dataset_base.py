from abc import ABC, abstractmethod

class DatasetBase(ABC):
	def __init__(self, **kwargs):
		pass

	@abstractmethod
	def load_data(self, data_path, no_labels=False):
		pass
