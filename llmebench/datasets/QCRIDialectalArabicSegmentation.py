from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class QCRIDialectalArabicSegmentationDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(QCRIDialectalArabicSegmentationDataset, self).__init__(**kwargs)

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
                "dev": [
                    "data/sequence_tagging_ner_pos_etc/segmentation/glf.seg/glf.data_5.dev.src.sent",
                    "data/sequence_tagging_ner_pos_etc/segmentation/lev.seg/lev.data_5.dev.src.sent",
                    "data/sequence_tagging_ner_pos_etc/segmentation/egy.seg/egy.data_5.dev.src.sent",
                    "data/sequence_tagging_ner_pos_etc/segmentation/mgr.seg/mgr.data_5.dev.src.sent",
                ],
                "test": [
                    "data/sequence_tagging_ner_pos_etc/segmentation/glf.seg/glf.data_5.test.src.sent",
                    "data/sequence_tagging_ner_pos_etc/segmentation/lev.seg/lev.data_5.test.src.sent",
                    "data/sequence_tagging_ner_pos_etc/segmentation/egy.seg/egy.data_5.test.src.sent",
                    "data/sequence_tagging_ner_pos_etc/segmentation/mgr.seg/mgr.data_5.test.src.sent",
                ],
            },
            "task_type": TaskType.Other,
        }

    def get_data_sample(self):
        return {
            "input": "Original sentence",
            "label": "Sentence with segmented words",
        }

    def load_data(self, data_path, no_labels=False):
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
