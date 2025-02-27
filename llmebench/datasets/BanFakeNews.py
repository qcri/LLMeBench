from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class BanFakeNewsDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(BanFakeNewsDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "bn",
            "citation": """@article{hossain2020banfakenews,
                          title={Banfakenews: A dataset for detecting fake news in bangla},
                          author={Hossain, Md Zobaer and Rahman, Md Ashraful and Islam, Md Saiful and Kar, Sudipta},
                          journal={arXiv preprint arXiv:2004.08789},
                          year={2020}
                        }""",
            "link": "https://github.com/Rowan1224/FakeNews",
            "license": "CC BY-NC-SA 2.0",
            "splits": {
                "test": "bn_fake_test.tsv",
                "train": "bn_fake_train.tsv",
            },
            "task_type": TaskType.Classification,
            "class_labels": ["True", "Clickbaits", "Satire", "Fake"],
        }

    @staticmethod
    def get_data_sample():
        return {"input": "News", "label": "Fake", "id": 1}

    def load_data(self, data_path):
        data_path = self.resolve_path(data_path)

        data = []
        with open(data_path, "r") as fp:
            next(fp)
            for line_idx, line in enumerate(fp):
                id, headline, content, label = line.strip().split("\t")
                label = label.capitalize()
                data.append(
                    {
                        "input": headline + "\n" + content,
                        "label": label,
                        "id": id,
                    }
                )

        return data
