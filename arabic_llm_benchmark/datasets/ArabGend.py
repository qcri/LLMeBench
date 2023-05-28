from arabic_llm_benchmark.datasets.dataset_base import DatasetBase

class ArabGendDataset(DatasetBase):
	def __init__(self):
		pass

	def load_data(self, data_path, no_labels=False):
		# TODO: modify to iterator
		data = []
		with open(data_path, "r") as fp:
			for line in fp:
				label, name = line.strip().split("\t")
				data.append((name, label[-1]))

		return data