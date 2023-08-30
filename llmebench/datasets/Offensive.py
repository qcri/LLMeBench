from llmebench.datasets.dataset_base import DatasetBase


class OffensiveDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(OffensiveDataset, self).__init__(**kwargs)

    def metadata():
        return {
            "language": "ar",
            "citation": """@article{zampieri2020semeval,
                title={SemEval-2020 task 12: Multilingual offensive language identification in social media (OffensEval 2020)},
                author={Zampieri, Marcos and Nakov, Preslav and Rosenthal, Sara and Atanasova, Pepa and Karadzhov, Georgi and Mubarak, Hamdy and Derczynski, Leon and Pitenis, ...},
                journal={arXiv preprint arXiv:2006.07235},
                year={2020}
            }""",
        }

    def get_data_sample(self):
        return {"input": "@USER يلا يا خوخة يا مهزئة ع دراستك", "label": "OFF"}

    def load_data(self, data_path, no_labels=False):
        # TODO: modify to iterator
        # Format: text \t offensive_label
        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                splits = line.strip().split("\t")
                if len(splits) < 2:
                    continue
                label = splits[1].strip()
                data.append(
                    {"input": splits[0], "label": label, "line_number": line_idx}
                )

        return data
