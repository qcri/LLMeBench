from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class QCRIDialectalArabicSegmentationDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(QCRIDialectalArabicSegmentationDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{samih-etal-2017-learning,
                title = "Learning from Relatives: Unified Dialectal {A}rabic Segmentation",
                author = "Samih, Younes  and
                  Eldesouki, Mohamed  and
                  Attia, Mohammed  and
                  Darwish, Kareem  and
                  Abdelali, Ahmed  and
                  Mubarak, Hamdy  and
                  Kallmeyer, Laura",
                booktitle = "Proceedings of the 21st Conference on Computational Natural Language Learning ({C}o{NLL} 2017)",
                month = aug,
                year = "2017",
                address = "Vancouver, Canada",
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/K17-1043",
                doi = "10.18653/v1/K17-1043",
                pages = "432--441"
            }""",
            "link": "https://alt.qcri.org/resources/da_resources/",
            "license": "Apache License, Version 2.0",
            "splits": {
                "glf.data_5": {
                    "dev": "glf.seg/glf.data_5.dev.src.sent",
                    "test": "glf.seg/glf.data_5.test.src.sent",
                },
                "lev.data_5": {
                    "dev": "lev.seg/lev.data_5.dev.src.sent",
                    "test": "lev.seg/lev.data_5.test.src.sent",
                },
                "egy.data_5": {
                    "dev": "egy.seg/egy.data_5.dev.src.sent",
                    "test": "egy.seg/egy.data_5.test.src.sent",
                },
                "mgr.data_5": {
                    "dev": "mgr.seg/mgr.data_5.dev.src.sent",
                    "test": "mgr.seg/mgr.data_5.test.src.sent",
                },
                "default": ["glf.data_5", "lev.data_5", "egy.data_5", "mgr.data_5"],
            },
            "task_type": TaskType.Other,
        }

    @staticmethod
    def get_data_sample():
        return {
            "input": "Original sentence",
            "label": "Sentence with segmented words",
        }

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)

        data = []

        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                data.append(
                    {
                        "input": line.replace("+", "").strip(),
                        "label": line.strip(),
                        "line_number": line_idx,
                    }
                )

        return data
