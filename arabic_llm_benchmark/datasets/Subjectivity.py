from arabic_llm_benchmark.datasets.dataset_base import DatasetBase
import pandas as pd

class Subjectivity(DatasetBase):
	def __init__(self):
		pass

	def load_data(self, data_path, no_labels=False):
		data = []
		# with open(data_path, "r") as fp:
		raw_data = pd.read_csv(data_path, sep='\t')
		for index, row in raw_data.iterrows():
			text = row["tweet_text"]
			label= str(row["class_label"])
			data.append((text, label))
		return data