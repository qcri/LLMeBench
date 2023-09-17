import pandas as pd

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class CT22CheckworthinessDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(CT22CheckworthinessDataset, self).__init__(**kwargs)

    @staticmethod
    def get_data_sample():
        return {
            "input": "some tweet",
            "label": "1",
            "input_id": 0,
            "line_number": 0,
        }

    @staticmethod
    def metadata():
        return {
            "language": ["ar", "bg", "nl", "en", "es", "tr"],
            "citation": """@inproceedings{nakov2022overview,
                  title={Overview of the clef--2022 checkthat! lab on fighting the covid-19 infodemic and fake news detection},
                  author={Nakov, Preslav and Barr{\\'o}n-Cede{\\~n}o, Alberto and da San Martino, Giovanni and Alam, Firoj and Stru{\\ss}, Julia Maria and Mandl, Thomas and M{\\'\\i}guez, Rub{\\'e}n and Caselli, Tommaso and Kutlu, Mucahid and Zaghouani, Wajdi and others},
                  booktitle={Experimental IR Meets Multilinguality, Multimodality, and Interaction: 13th International Conference of the CLEF Association, CLEF 2022, Bologna, Italy, September 5--8, 2022, Proceedings},
                  pages={495--520},
                  year={2022},
                  organization={Springer}
            }""",
            "link": "https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab",
            "license": "Research Purpose Only",
            "splits": {
                "ar": {
                    "test": "arabic/CT22_arabic_1A_checkworthy_test_gold.tsv",
                    "train": "arabic/CT22_arabic_1A_checkworthy_train.tsv",
                },
                "bg": {
                    "test": "bulgarian/CT22_bulgarian_1A_checkworthy_test_gold.tsv",
                    "train": "bulgarian/CT22_bulgarian_1A_checkworthy_train.tsv",
                },
                "en": {
                    "test": "english/CT22_english_1A_checkworthy_test_gold.tsv",
                    "train": "english/CT22_english_1A_checkworthy_train.tsv",
                },
                "es": {
                    "test": "spanish/CT22_spanish_1A_checkworthy_test_gold.tsv",
                    "train": "spanish/CT22_spanish_1A_checkworthy_train.tsv",
                },
                "nl": {
                    "test": "dutch/CT22_dutch_1A_checkworthy_test_gold.tsv",
                    "train": "dutch/CT22_dutch_1A_checkworthy_train.tsv",
                },
                "tr": {
                    "test": "turkish/CT22_turkish_1A_checkworthy_test_gold.tsv",
                    "train": "turkish/CT22_turkish_1A_checkworthy_train.tsv",
                },
            },
            "task_type": TaskType.Classification,
            "class_labels": ["0", "1"],
        }

    def load_data(self, data_path):
        data_path = self.resolve_path(data_path)

        data = []
        raw_data = pd.read_csv(data_path, sep="\t", dtype={"tweet_id": object})
        for index, row in raw_data.iterrows():
            text = row["tweet_text"]
            label = str(row["class_label"])
            tweet_id = str(row["tweet_id"])
            data.append(
                {
                    "input": text,
                    "label": label,
                    "input_id": tweet_id,
                    "line_number": index,
                }
            )
        return data
