from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class OSACT4SubtaskBDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(OSACT4SubtaskBDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{mubarak2020overview,
                title={Overview of OSACT4 Arabic offensive language detection shared task},
                author={Mubarak, Hamdy and Darwish, Kareem and Magdy, Walid and Elsayed, Tamer and Al-Khalifa, Hend},
                booktitle={Proceedings of the 4th Workshop on open-source arabic corpora and processing tools, with a shared task on offensive language detection},
                pages={48--52},
                year={2020}
            }""",
            "link": "https://edinburghnlp.inf.ed.ac.uk/workshops/OSACT4/",
            "license": "CC BY 4.0",
            "splits": {
                "test": "OSACT2020-sharedTask-test-tweets-labels.txt",
                "train": "OSACT2020-sharedTask-train_HS.txt",
            },
            "task_type": TaskType.Classification,
            "class_labels": ["HS", "NOT_HS"],
        }

    @staticmethod
    def get_data_sample():
        return {"input": "ايه اللي انت بتقوله ده يا اوروبي يا متخلف", "label": "HS"}

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)

        # Format: text \t hatespeech_label
        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                if len(line.split("\t")) == 2:
                    text, label = line.split("\t")
                else:
                    text, label = line.split("\t")[:2]
                label = label.strip()
                data.append({"input": text, "label": label, "line_number": line_idx})

        return data
