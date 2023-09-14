import pandas as pd

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class CT22CheckworthinessDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(CT22CheckworthinessDataset, self).__init__(**kwargs)

    def get_data_sample(self):
        return {
            "input": "some tweet",
            "label": "1",
            "input_id": 0,
            "line_number": 0,
        }

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
                    "test": "data/factuality_disinformation_harmful_content/checkworthyness/arabic/CT22_arabic_1A_checkworthy_test_gold.tsv",
                    "train": "data/factuality_disinformation_harmful_content/checkworthyness/arabic/CT22_arabic_1A_checkworthy_train.tsv",
                },
                "bg": {
                    "test": "data/factuality_disinformation_harmful_content/checkworthyness/bulgarian/CT22_bulgarian_1A_checkworthy_test_gold.tsv",
                    "train": "data/factuality_disinformation_harmful_content/checkworthyness/bulgarian/CT22_bulgarian_1A_checkworthy_train.tsv",
                },
                "en": {
                    "test": "data/factuality_disinformation_harmful_content/checkworthyness/english/CT22_english_1A_checkworthy_test_gold.tsv",
                    "train": "data/factuality_disinformation_harmful_content/checkworthyness/english/CT22_english_1A_checkworthy_train.tsv",
                },
                "es": {
                    "test": "data/factuality_disinformation_harmful_content/checkworthyness/spanish/CT22_spanish_1A_checkworthy_test_gold.tsv",
                    "train": "data/factuality_disinformation_harmful_content/checkworthyness/spanish/CT22_spanish_1A_checkworthy_train.tsv",
                },
                "nl": {
                    "test": "data/factuality_disinformation_harmful_content/checkworthyness/dutch/CT22_dutch_1A_checkworthy_test_gold.tsv",
                    "train": "data/factuality_disinformation_harmful_content/checkworthyness/dutch/CT22_dutch_1A_checkworthy_train.tsv",
                },
                "tr": {
                    "test": "data/factuality_disinformation_harmful_content/checkworthyness/turkish/CT22_turkish_1A_checkworthy_test_gold.tsv",
                    "train": "data/factuality_disinformation_harmful_content/checkworthyness/turkish/CT22_turkish_1A_checkworthy_train.tsv",
                },
            },
            "task_type": TaskType.Classification,
            "class_labels": ["0", "1"],
        }

    def load_data(self, data_path):
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
