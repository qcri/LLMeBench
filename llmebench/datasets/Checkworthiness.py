import pandas as pd

from llmebench.datasets.dataset_base import DatasetBase


class CheckworthinessDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(CheckworthinessDataset, self).__init__(**kwargs)

    def get_data_sample(self):
        return {
            "input": "some tweet",
            "label": "checkworthy",
            "input_id": 0,
            "line_number": 0,
        }

    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{nakov2022overview,
                  title={Overview of the clef--2022 checkthat! lab on fighting the covid-19 infodemic and fake news detection},
                  author={Nakov, Preslav and Barr{\\'o}n-Cede{\\~n}o, Alberto and da San Martino, Giovanni and Alam, Firoj and Stru{\\ss}, Julia Maria and Mandl, Thomas and M{\\'\\i}guez, Rub{\\'e}n and Caselli, Tommaso and Kutlu, Mucahid and Zaghouani, Wajdi and others},
                  booktitle={Experimental IR Meets Multilinguality, Multimodality, and Interaction: 13th International Conference of the CLEF Association, CLEF 2022, Bologna, Italy, September 5--8, 2022, Proceedings},
                  pages={495--520},
                  year={2022},
                  organization={Springer}
            }""",
        }

    def load_data(self, data_path):
        data = []
        raw_data = pd.read_csv(data_path, sep="\t", dtype={"tweet_id": object})
        for index, row in raw_data.iterrows():
            text = row["tweet_text"]
            label = str(row["class_label"])
            tweet_id = str(row["tweet_id"])
            data.append(
                {
                    "input": text,
                    "label": label,
                    "input_id": tweet_id,
                    "line_number": index,
                }
            )
        return data
