from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class SST2(DatasetBase):
    def __init__(self, **kwargs):
        super(SST2, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "en",
            "citation": """@inproceedings{socher-etal-2013-recursive,
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
            }
            """,
            "link": "https://huggingface.co/datasets/sst2",
            "license": "Unknown",
            "splits": {
                "test": "validation.tsv",
                "train": "train.tsv",
            },
            "task_type": TaskType.Classification,
            "class_labels": ["0", "1"],
            #'0': negative
            # '1': positive
        }

    @staticmethod
    def get_data_sample():
        return {"input": "uneasy mishmash of styles and genres", "label": "negative"}

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)
        data = []
        label_dict = {0: "negative", 1: "positive"}
        with open(data_path, "r") as fp:
            next(fp)
            for line_idx, line in enumerate(fp):
                fields = line.split("\t")
                input_id = fields[0]
                text = fields[1].strip()
                label_id = int(fields[2].strip())
                if text == "" or label_id == "":
                    continue
                label = label_dict[label_id]
                data.append(
                    {
                        "input": text,
                        "label": label,
                        "input_id": input_id,
                    }
                )

        return data
