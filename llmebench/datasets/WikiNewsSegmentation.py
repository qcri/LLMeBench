from llmebench.datasets.dataset_base import DatasetBase


class WikiNewsSegmentationDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(WikiNewsSegmentationDataset, self).__init__(**kwargs)

    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{darwish2016farasa,
                title={Farasa: A new fast and accurate {A}rabic word segmenter},
                author={Darwish, Kareem and Mubarak, Hamdy},
                booktitle={Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC'16)},
                pages={1070--1074},
                year={2016}
            }""",
        }

    def get_data_sample(self):
        return {
            "input": "Original sentence",
            "label": "Sentence with segmented words",
        }

    def load_data(self, data_path, no_labels=False):
        # TODO: modify to iterator
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
