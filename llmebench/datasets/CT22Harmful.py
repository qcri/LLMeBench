from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class CT22HarmfulDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(CT22HarmfulDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": ["ar", "bg", "nl", "en", "tr"],
            "citation": """@inproceedings{nakov2022overview,
                title={Overview of the CLEF-2022 CheckThat! lab task 1 on identifying relevant claims in tweets},
                author={Nakov, Preslav and Barr{\\'o}n-Cede{\\~n}o, Alberto and Da San Martino, Giovanni and Alam, Firoj and Kutlu, Mucahid and Zaghouani, Wajdi and Li, Chengkai and Shaar, Shaden and Mubarak, Hamdy and Nikolov, Alex},
                 year={2022},
                booktitle={Proceedings of the Working Notes of CLEF 2022 - Conference and Labs of the Evaluation Forum}
            }""",
            "link": "https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab",
            "license": "Research Purpose Only",
            "splits": {
                "ar": {
                    "test": "CT22_arabic_1C_harmful_test_gold.tsv",
                    "train": "CT22_arabic_1C_harmful_train.tsv",
                },
            },
            "task_type": TaskType.Classification,
            "class_labels": ["0", "1"],
        }

    @staticmethod
    def get_data_sample():
        return {"input": "Tweet", "label": "1"}

    def load_data(self, data_path):
        data_path = self.resolve_path(data_path)

        formatted_data = []

        with open(data_path, "r", encoding="utf-8") as in_file:
            next(in_file)
            for index, line in enumerate(in_file):
                tweet = [str(s.strip()) for s in line.split("\t")]

                text = tweet[3]
                label = tweet[4]
                twt_id = tweet[1]

                formatted_data.append(
                    {
                        "input": text,
                        "label": label,
                        "line_number": index,
                        "input_id": twt_id,
                    }
                )

        return formatted_data
