from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class BanglaSentimentDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(BanglaSentimentDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "bn",
            "citation": """@article{alam2021review,
                title={A Review of Bangla Natural Language Processing Tasks and the Utility of Transformer Models},
                author={Alam, Firoj and Hasan, Md Arid and Alam, Tanvir and Khan, Akib and Tajrin, Janntatul and Khan, Naira and Chowdhury, Shammur Absar},
                journal={arXiv preprint arXiv:2107.03844},
                year={2021}
            }        
            @inproceedings{iccit2020Arid,
                Author = {Md. Arid Hasan and Jannatul Tajrin and Shammur Absar Chowdhury and Firoj Alam},
                Booktitle = {23rd International Conference on Computer and Information Technology (ICCIT)},
                Month = {December},
                Title = {Sentiment Classification in Bangla Textual Content: A Comparative Study},
                Year = {2020},
                url={https://github.com/banglanlp/bangla-sentiment-classification},
            }""",
            "link": "https://github.com/banglanlp/bangla-sentiment-classification",
            "license": "CC BY-NC-SA 2.0",
            "splits": {
                "test": "bn_all_test.tsv",
                "train": "bn_all_train.tsv",
            },
            "task_type": TaskType.Classification,
            "class_labels": ["Positive", "Negative", "Neutral"],
        }

    @staticmethod
    def get_data_sample():
        return {"input": "Tweet", "label": "Positive", "line_number": 0}

    def load_data(self, data_path):
        data_path = self.resolve_path(data_path)

        data = []
        with open(data_path, "r") as fp:
            next(fp)
            for line_idx, line in enumerate(fp):
                id, text, label = line.strip().split("\t")
                label = label.capitalize()
                data.append({"input": text, "label": label, "line_number": id})

        return data
