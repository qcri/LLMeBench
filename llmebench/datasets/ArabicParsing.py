from llmebench.datasets.dataset_base import DatasetBase


class ArabicParsingDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ArabicParsingDataset, self).__init__(**kwargs)

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
            "label": {
                "1": "2",
                "2": "0",
            },
        }

    def load_data(self, data_path, no_labels=False):
        data = []
        send_id = 0
        sent_lab = {}
        sent_src = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                if len(line.split("\t")) < 6:
                    data.append(
                        {
                            "input": "\n".join(sent_src),
                            "label": sent_lab,
                            "sent_number": send_id,
                        }
                    )
                    send_id += 1
                    sent_lab = {}
                    sent_src = []
                else:
                    sent_src.append("\t".join(line.split("\t")[:6]))
                    lid = line.split("\t")[0]
                    sent_lab[lid] = line.split("\t")[6]

        return data
