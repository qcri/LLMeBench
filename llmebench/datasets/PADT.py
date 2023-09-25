from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class PADTDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(PADTDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{buchholz-marsi-2006-conll,
                title = "{C}o{NLL}-{X} Shared Task on Multilingual Dependency Parsing",
                author = "Buchholz, Sabine  and
                  Marsi, Erwin",
                booktitle = "Proceedings of the Tenth Conference on Computational Natural Language Learning ({C}o{NLL}-X)",
                month = jun,
                year = "2006",
                address = "New York City",
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/W06-2920",
                pages = "149--164",
            }
            @inproceedings{smrz2002prague,
                title={Prague dependency treebank for Arabic: Multi-level annotation of Arabic corpus},
                author={Smrz, Otakar and {\\v{S}}naidauf, Jan and Zem{\\'a}nek, Petr},
                booktitle={Proc. of the Intern. Symposium on Processing of Arabic},
                pages={147--155},
                year={2002}
            }
            @misc{hajic2004prague,
                title={Prague Arabic Dependency Treebank 1.0. LDC2004T23},
                author={Hajic, Jan and Smrz, Otakar and Zem{\'a}nek, Petr and Pajas, Petr and {\v{S}}naidauf, Jan and Be{\v{s}}ka, Emanuel and Kr{\'a}cmar, Jakub and Hassanov{\'a}, Kamila},
                year={2004},
                publisher={2004a}
            }
            """,
            "link": "https://ufal.mff.cuni.cz/padt/PADT_1.0/docs/index.html",
            "splits": {
                "test": "arabic_PADT_test_gs.conll",
                "train": "arabic_PADT_train.conll",
            },
            "task_type": TaskType.Other,
        }

    @staticmethod
    def get_data_sample():
        return {
            "input": "Original sentence",
            "label": {
                "1": "2",
                "2": "0",
            },
        }

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)

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
