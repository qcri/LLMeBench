from llmebench.datasets.dataset_base import DatasetBase


class ArSASSentimentDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ArSASSentimentDataset, self).__init__(**kwargs)

    def citation(self):
        return """@inproceedings{Elmadany2018ArSASA,
            title={ArSAS : An Arabic Speech-Act and Sentiment Corpus of Tweets},
            author={AbdelRahim Elmadany and Hamdy Mubarak and Walid Magdy},
            year={2018}
        }"""

    def get_data_sample(self):
        return {"input": "Tweet", "label": "Positive"}

    def load_data(self, data_path, no_labels=False):
        # TODO: modify to iterator
        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                text, label = line.strip().split("\t")
                data.append({"input": text, "label": label, "line_number": line_idx})

        return data
