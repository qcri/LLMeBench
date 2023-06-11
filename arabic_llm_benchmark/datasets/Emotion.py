from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class EmotionDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(EmotionDataset, self).__init__(**kwargs)

    def citation(self):
        return """FIXME"""

    def get_data_sample(self):
        return {"input": "Tweet", "label": [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]}

    def load_data(self, data_path, no_labels=False):
        # TODO: modify to iterator
        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                if line_idx == 0:
                    continue
                fields = line.split("\t")
                text = fields[1]
                ref_labels = []
                for j in range(2, 13):
                    ref_labels.append(int(fields[j]))
                data.append(
                    {"input": text, "label": ref_labels, "line_number": line_idx}
                )

        return data
        # raw_data = pd.read_csv(data_path, sep="\t")
        # for index, row in raw_data.iterrows():
        #     text = row["Tweet"]
        #     fields = line.split("\t")
        #     s = fields[1]
        #     ref_labels = []
        #     for j in range(2,13):
        #         ref_labels.append(int(fields[j]))
        #     label = str(row["class_label"])
        #     data.append({"input": text, "label": label, "line_number": index})
        # return data
