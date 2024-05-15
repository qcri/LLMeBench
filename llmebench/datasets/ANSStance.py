from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class ANSStanceDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ANSStanceDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{khouja-2020-stance,
                title = "Stance Prediction and Claim Verification: An {A}rabic Perspective",
                author = "Khouja, Jude",
                booktitle = "Proceedings of the Third Workshop on Fact Extraction and VERification (FEVER)",
                month = jul,
                year = "2020",
                address = "Online",
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/2020.fever-1.2",
                doi = "10.18653/v1/2020.fever-1.2",
                pages = "8--17",
                abstract = "This work explores the application of textual entailment in news claim verification and stance prediction using a new corpus in Arabic. The publicly available corpus comes in two perspectives: a version consisting of 4,547 true and false claims and a version consisting of 3,786 pairs (claim, evidence). We describe the methodology for creating the corpus and the annotation process. Using the introduced corpus, we also develop two machine learning baselines for two proposed tasks: claim verification and stance prediction. Our best model utilizes pretraining (BERT) and achieves 76.7 F1 on the stance prediction task and 64.3 F1 on the claim verification task. Our preliminary experiments shed some light on the limits of automatic claim verification that relies on claims text only. Results hint that while the linguistic features and world knowledge learned during pretraining are useful for stance prediction, such learned representations from pretraining are insufficient for verifying claims without access to context or evidence.",
            }""",
            "link": "https://github.com/latynt/ans",
            "download_url": (
                "https://github.com/latynt/ans/archive/refs/heads/master.zip"
            ),
            "splits": {
                "test": "ans-master/data/stance/test.csv",
                "train": "ans-master/data/stance/train.csv",
            },
            "task_type": TaskType.Classification,
            "class_labels": ["agree", "disagree"],
        }

    @staticmethod
    def get_data_sample():
        return {
            "input": {
                "sentence_1": "الجملة الاولى",
                "sentence_2": "الجملة الثانية",
            },
            "label": "agree",
        }

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)
        data = []
        with open(data_path, "r", encoding="utf-8") as fp:
            next(fp)  # skip header
            for line_idx, line in enumerate(fp):
                s1, s2, label = line.strip().split(",")

                data.append(
                    {
                        "input": {"sentence_1": s1.strip(), "sentence_2": s2.strip()},
                        "label": label,
                        "line_number": line_idx,
                    }
                )

        return data
