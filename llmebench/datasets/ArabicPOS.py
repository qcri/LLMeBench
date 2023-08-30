from llmebench.datasets.dataset_base import DatasetBase


class ArabicPOSDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ArabicPOSDataset, self).__init__(**kwargs)

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
        }

    def get_data_sample(self):
        return {
            "input": "Original sentence",
            "label": "Sentence with POS tags",
        }

    def load_data(self, data_path, no_labels=False):
        data = []

        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                data.append(
                    {
                        "input": line.strip().split("\t")[0],
                        "label": line.strip().split("\t")[1],
                        "line_number": line_idx,
                    }
                )

        return data
