from llmebench.datasets.dataset_base import DatasetBase


class QADIDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(QADIDataset, self).__init__(**kwargs)

    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{abdelali2021qadi,
                title={{QADI}: Arabic dialect identification in the wild},
                author={Abdelali, Ahmed and Mubarak, Hamdy and Samih, Younes and Hassan, Sabit and Darwish, Kareem},
                booktitle={Proceedings of the Sixth Arabic Natural Language Processing Workshop},
                pages={1--10},
                year={2021}
            }""",
        }

    def get_data_sample(self):
        return {"input": "طب ماتمشي هو حد ماسك فيك", "label": "EG"}

    def load_data(self, data_path, no_labels=False):
        # TODO: modify to iterator
        # Format: dialect_id_label \t text
        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                fields = line.split()
                label = fields[0]
                label = label.replace("__label__", "")

                text = ""
                for j in range(1, len(fields)):
                    text += f"{fields[j]} "

                data.append({"input": text, "label": label, "line_number": line_idx})

        return data
