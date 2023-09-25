from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class QADIDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(QADIDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{abdelali2021qadi,
                title={{QADI}: Arabic dialect identification in the wild},
                author={Abdelali, Ahmed and Mubarak, Hamdy and Samih, Younes and Hassan, Sabit and Darwish, Kareem},
                booktitle={Proceedings of the Sixth Arabic Natural Language Processing Workshop},
                pages={1--10},
                year={2021}
            }""",
            "link": "https://alt.qcri.org/resources/qadi/",
            "license": "Apache License, Version 2.0",
            "splits": {"test": "QADI_test-PalestinePS-corrected.txt"},
            "task_type": TaskType.Classification,
            "class_labels": [
                "EG",
                "DZ",
                "SD",
                "YE",
                "SY",
                "TN",
                "AE",
                "JO",
                "LY",
                "PS",
                "OM",
                "LB",
                "KW",
                "QA",
                "BH",
                "MSA",
                "SA",
                "IQ",
                "MA",
            ],
        }

    @staticmethod
    def get_data_sample():
        return {"input": "طب ماتمشي هو حد ماسك فيك", "label": "EG"}

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)

        # Format: dialect_id_label \t text
        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                fields = line.split()
                label = fields[0]
                label = label.replace("__label__", "")

                text = ""
                for j in range(1, len(fields)):
                    text += f"{fields[j]} "

                data.append({"input": text, "label": label, "line_number": line_idx})

        return data
