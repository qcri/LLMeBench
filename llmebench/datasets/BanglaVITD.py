from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class BanglaVITDDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(BanglaVITDDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "bn",
            "citation": """@inproceedings{SahaAndJunaed,
                  title = "Vio-Lens: A Novel Dataset of Annotated Social Network Posts Leading to Different Forms of Communal Violence and its Evaluation",
                  author= "Saha, Sourav and Junaed, Jahedul Alam  and Saleki, Maryam and Sharma, Arnab Sen and Rifat, Mohammad Rashidujjaman and Rahout, Mohamed and Ahmed, Syed Ishtiaque and Mohammad, Nabeel and Amin, Mohammad Ruhul",
                  booktitle =  "Proceedings of the 1st International Workshop on Bangla Language Processing (BLP-2023)",
                  month = "Dec",
                  year = "2023",
                  publisher = "Association for Computational Linguistics",
                  address = "Singapore",
                }    
                @inproceedings{blp2023-overview-task1,
                  title = "BLP-2023 Task 1: Violence Inciting Text Detection (VITD)",
                  author= "Saha, Sourav and Junaed, Jahedul Alam and Saleki, Maryam and Rahouti, Mohamed and Mohammed, Nabeel and Amin, Mohammad Ruhul",
                  booktitle =  "Proceedings of the 1st International Workshop on Bangla Language Processing (BLP-2023)",
                  month = "Dec",
                  year = "2023",
                  publisher = "Association for Computational Linguistics",
                  address = "Singapore",
                }""",
            "link": "https://github.com/blp-workshop/blp_task1",
            "license": "CC BY-NC-SA 2.0",
            "splits": {
                "test": "test.tsv",
                "train": "train.tsv",
            },
            "task_type": TaskType.Classification,
            "class_labels": ["Direct Violence", "Passive Violence", "Non-Violence"],
        }

    @staticmethod
    def get_data_sample():
        return {"input": "text", "label": "Direct Violence", "line_number": 0}

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
