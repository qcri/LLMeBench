from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class OSACT4SubtaskADataset(DatasetBase):
    def __init__(self, **kwargs):
        super(OSACT4SubtaskADataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """
            @inproceedings{zampieri-etal-2020-semeval,
                title = "{S}em{E}val-2020 Task 12: Multilingual Offensive Language Identification in Social Media ({O}ffens{E}val 2020)",
                author = {Zampieri, Marcos  and
                  Nakov, Preslav  and
                  Rosenthal, Sara  and
                  Atanasova, Pepa  and
                  Karadzhov, Georgi  and
                  Mubarak, Hamdy  and
                  Derczynski, Leon  and
                  Pitenis, Zeses  and
                  {\\c{C}}{\\"o}ltekin, {\\c{C}}a{\\u{g}}r{\\i}},
                booktitle = "Proceedings of the Fourteenth Workshop on Semantic Evaluation",
                month = dec,
                year = "2020",
                address = "Barcelona (online)",
                publisher = "International Committee for Computational Linguistics",
                url = "https://aclanthology.org/2020.semeval-1.188",
                doi = "10.18653/v1/2020.semeval-1.188",
                pages = "1425--1447",
                abstract = "We present the results and the main findings of SemEval-2020 Task 12 on Multilingual Offensive Language Identification in Social Media (OffensEval-2020). The task included three subtasks corresponding to the hierarchical taxonomy of the OLID schema from OffensEval-2019, and it was offered in five languages: Arabic, Danish, English, Greek, and Turkish. OffensEval-2020 was one of the most popular tasks at SemEval-2020, attracting a large number of participants across all subtasks and languages: a total of 528 teams signed up to participate in the task, 145 teams submitted official runs on the test data, and 70 teams submitted system description papers.",
            }
            """,
            "link": "https://edinburghnlp.inf.ed.ac.uk/workshops/OSACT4/",
            "license": "CC BY 4.0",
            "splits": {
                "test": "OSACT2020-sharedTask-test-tweets-labels.txt",
                "train": "OSACT2020-sharedTask-train_OFF.txt",
            },
            "task_type": TaskType.Classification,
            "class_labels": ["OFF", "NOT_OFF"],
        }

    @staticmethod
    def get_data_sample():
        return {"input": "@USER يلا يا خوخة يا مهزئة ع دراستك", "label": "OFF"}

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)

        # Format: text \t offensive_label
        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                splits = line.strip().split("\t")
                if len(splits) < 2:
                    continue
                label = splits[1].strip()
                data.append(
                    {"input": splits[0], "label": label, "line_number": line_idx}
                )

        return data
