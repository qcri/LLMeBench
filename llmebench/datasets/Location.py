from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class LocationDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(LocationDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{mubarak2021ul2c,
                title={{UL2C}: Mapping user locations to countries on Arabic Twitter},
                author={Mubarak, Hamdy and Hassan, Sabit},
                booktitle={Proceedings of the Sixth Arabic Natural Language Processing Workshop},
                pages={145--153},
                year={2021}
            }""",
            "link": "https://alt.qcri.org/resources/UL2C-UserLocationsToCountries.tsv",
            "splits": {
                "test": "arab+others.txt",
                "train": "arab+others_dev.txt",
            },
            "task_type": TaskType.Classification,
            "class_labels": [
                "ae",
                "OTHERS",
                "bh",
                "dz",
                "eg",
                "iq",
                "jo",
                "kw",
                "lb",
                "ly",
                "ma",
                "om",
                "ps",
                "qa",
                "sa",
                "sd",
                "so",
                "sy",
                "tn",
                "UNK",
                "ye",
                "mr",
            ],
        }

    @staticmethod
    def get_data_sample():
        return {"input": "Doha, Qatar", "label": "QA"}

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)

        # Format: location \t country_code
        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                location, country = line.split("\t")
                location = location.strip()
                country = country.strip()
                data.append(
                    {"input": location, "label": country, "line_number": line_idx}
                )

        return data
