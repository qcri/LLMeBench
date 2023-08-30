from llmebench.datasets.dataset_base import DatasetBase


class NameInfoDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(NameInfoDataset, self).__init__(**kwargs)

    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{Under review...}
            }""",
        }

    def get_data_sample(self):
        return {"input": "جورج واشنطن", "label": "GB"}

    def load_data(self, data_path, no_labels=False):
        # TODO: modify to iterator
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
