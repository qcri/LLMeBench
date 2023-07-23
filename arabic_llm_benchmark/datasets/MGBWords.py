from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class MGBWordsDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(MGBWordsDataset, self).__init__(**kwargs)

    def citation(self):
        return """ Hamdy Mubarak, Amir Hussein, Shammur Absar Chowdhury, and Ahmed Ali. 2021. 
                QASR: QCRI Aljazeera Speech Resource A Large Scale Annotated Arabic Speech Corpus. 
                In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics 
                and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 
                pages 2274–2285, Online. Association for Computational Linguistics. """

    def get_data_sample(self):
        return {"input": "sentence", "label": "named entity labels are here"}

    def load_data(self, data_path, no_labels=False):
        data = []
        with open(data_path, "rt", encoding="UTF8") as fp:
            for idx, line in enumerate(fp):
                if idx == 7906:
                    break
                fields = line.split("\t")
                words = fields[4].split()
                s = ""
                ref_label = []
                for w in words:
                    index1 = w.find("/")
                    if index1 < 0:
                        index1 = len(w)
                        w += "/O"
                    word = w[0:index1]
                    label = w[index1 + 1 :]
                    if label == "B-OTH" or label == "I-OTH":
                        label = "O"
                    ref_label.append(label)

                    s += f"{word} "
                s = s.strip()
                ref_label = " ".join(ref_label)

                data.append({"input": s, "label": ref_label})

        return data
