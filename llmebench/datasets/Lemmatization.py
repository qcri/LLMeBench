from llmebench.datasets.dataset_base import DatasetBase


class LemmatizationDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(LemmatizationDataset, self).__init__(**kwargs)

    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{mubarak2018build,
                title={Build Fast and Accurate Lemmatization for Arabic},
                author={Mubarak, Hamdy},
                booktitle={Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
                year={2018}
            }""",
        }

    def get_data_sample(self):
        return {
            "input": "جوائز",
            "label": ("جوائز", "جائزة"),
        }

    def load_data(self, data_path, no_labels=False):
        # TODO: modify to iterator
        # Format: words \t lemmas
        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                words, lemmas = line.split("\t")
                words = words.split()
                lemmas = lemmas.split()
                for i, w in enumerate(words):
                    text = w
                    label = lemmas[i]

                    # Supply origin unlemmatized word in the label as well
                    # to handle failed predictions in evaluate
                    data.append(
                        {"input": text, "label": (text, label), "line_number": line_idx}
                    )

        return data
