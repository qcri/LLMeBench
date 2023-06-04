import pandas as pd

from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class CheckworthinessDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(CheckWorthinessDataset, self).__init__(**kwargs)

    def citation(self):
        return """
                @inproceedings{nakov2022overview,
                  title={Overview of the clef--2022 checkthat! lab on fighting the covid-19 infodemic and fake news detection},
                  author={Nakov, Preslav and Barr{\\'o}n-Cede{\\~n}o, Alberto and da San Martino, Giovanni and Alam, Firoj and Stru{\\ss}, Julia Maria and Mandl, Thomas and M{\\'\\i}guez, Rub{\\'e}n and Caselli, Tommaso and Kutlu, Mucahid and Zaghouani, Wajdi and others},
                  booktitle={Experimental IR Meets Multilinguality, Multimodality, and Interaction: 13th International Conference of the CLEF Association, CLEF 2022, Bologna, Italy, September 5--8, 2022, Proceedings},
                  pages={495--520},
                  year={2022},
                  organization={Springer}
                }

        """

    def load_data(self, data_path):
        data = []
        raw_data = pd.read_csv(data_path, sep="\t")
        for index, row in raw_data.iterrows():
            text = row["tweet_text"]
            label = str(row["class_label"])
            data.append({"input": text, "label": label, "line_number": index})
        return data
