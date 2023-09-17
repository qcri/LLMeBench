from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class NameInfoDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(NameInfoDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{Under review...}""",
            "splits": {
                "test": "wikidata_test.txt",
                "train": "wikidata_dev.txt",
            },
            "task_type": TaskType.Classification,
            "class_labels": [
                "gb",
                "us",
                "cl",
                "fr",
                "ru",
                "pl",
                "in",
                "it",
                "kr",
                "gh",
                "ca",
                "sa",
                "at",
                "de",
                "cn",
                "br",
                "dk",
                "se",
                "bd",
                "cu",
                "jp",
                "be",
                "es",
                "co",
                "id",
                "iq",
                "pk",
                "tr",
                "il",
                "ch",
                "ar",
                "ro",
                "nl",
                "ps",
                "ug",
                "ir",
                "cg",
                "do",
                "ee",
                "tn",
                "gr",
                "np",
                "ie",
                "sy",
                "hu",
                "eg",
                "ma",
                "ve",
                "ph",
                "no",
                "bg",
                "si",
                "ke",
                "au",
                "et",
                "py",
                "af",
                "pt",
                "th",
                "bo",
                "mx",
                "lb",
                "za",
                "fi",
                "hr",
                "vn",
                "ly",
                "nz",
                "qa",
                "kh",
                "ci",
                "ng",
                "sg",
                "cm",
                "dz",
                "tz",
                "ae",
                "pe",
                "az",
                "lu",
                "ec",
                "cz",
                "ua",
                "uy",
                "sd",
                "ao",
                "my",
                "lv",
                "kw",
                "tw",
                "bh",
                "lk",
                "ye",
                "cr",
                "jo",
                "pa",
                "om",
                "uz",
                "by",
                "kz",
            ],
        }

    @staticmethod
    def get_data_sample():
        return {"input": "جورج واشنطن", "label": "GB"}

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)

        # Format:
        # جورج واشنطن	United Kingdom	GB
        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                name, country, country_code = line.split("\t")
                name = name.strip()
                country_code = country_code.strip().lower()
                data.append(
                    {"input": name, "label": country_code, "line_number": line_idx}
                )

        return data
