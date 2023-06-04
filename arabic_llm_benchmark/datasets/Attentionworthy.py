import pandas as pd

from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class AttentionworthyDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(AttentionworthyDataset, self).__init__(**kwargs)

    def get_data_sample(self):
        return {"input": "some tweet", "label": "no_not_interesting"}

    def citation(self):
        return """
                @InProceedings{clef-checkthat:2022:task1,
                author = {Nakov, Preslav and Barr\\'{o}n-Cede\\~{n}o, Alberto and Da San Martino, Giovanni and Alam, Firoj and M\\'{\\i}guez, Rub\'{e}n and Caselli, Tommaso and Kutlu, Mucahid and Zaghouani, Wajdi and Li, Chengkai and Shaar, Shaden and Mubarak, Hamdy and Nikolov, Alex and Kartal, Yavuz Selim and Beltr\\'{a}n, Javier},
                title = "Overview of the {CLEF}-2022 {CheckThat}! Lab Task 1 on Identifying Relevant Claims in Tweets",
                year = {2022},
                booktitle = "Working Notes of {CLEF} 2022---Conference and Labs of the Evaluation Forum",
                series = {CLEF~'2022},
                address = {Bologna, Italy},
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
