from llmebench.datasets.HuggingFaceDatasetBase import HuggingFaceDatasetBase
from llmebench.tasks import TaskType


class SST2Dataset(HuggingFaceDatasetBase):
    def __init__(self, **kwargs):
        super(SST2Dataset, self).__init__(huggingface_model_name="sst2", **kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "en",
            "citation": """@@inproceedings{socher-etal-2013-recursive,
			    title = "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank",
			    author = "Socher, Richard  and
			      Perelygin, Alex  and
			      Wu, Jean  and
			      Chuang, Jason  and
			      Manning, Christopher D.  and
			      Ng, Andrew  and
			      Potts, Christopher",
			    booktitle = "Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing",
			    month = oct,
			    year = "2013",
			    address = "Seattle, Washington, USA",
			    publisher = "Association for Computational Linguistics",
			    url = "https://www.aclweb.org/anthology/D13-1170",
			    pages = "1631--1642",
			}""",
            "link": "https://huggingface.co/datasets/sst2",
            "license": "Research Purpose Only",
            "splits": {
                "test": "test",
                "train": "train",
                "dev": "validation",
            },
            "task_type": TaskType.Classification,
            "class_labels": ["0", "1"],
        }

    @staticmethod
    def get_data_sample():
        return {"input": "Bad Movie", "label": "0"}

    def load_data(self, data_split, no_labels=False):
        data = []
        for sample in self.dataset[data_split]:
            data.append(
                {
                    "input": sample["sentence"],
                    "label": str(sample["label"]),
                    "input_id": sample["idx"],
                }
            )

        return data
