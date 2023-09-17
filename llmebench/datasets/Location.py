from llmebench.datasets.dataset_base import DatasetBase


class LocationDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(LocationDataset, self).__init__(**kwargs)

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
        }

    def get_data_sample(self):
        return {"input": "Doha, Qatar", "label": "QA"}

    def load_data(self, data_path, no_labels=False):
        # TODO: modify to iterator
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
