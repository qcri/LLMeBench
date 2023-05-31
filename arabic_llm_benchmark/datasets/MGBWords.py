from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class MGBWordsDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(MGBWordsDataset, self).__init__(**kwargs)

    def citation(self):
        return """..."""

    def load_data(self, data_path):
        data = []
        with open(data_path, "rt", encoding='UTF8') as fp:
            for i, line in enumerate(fp):
                # if i % 10 == 0:
                #     print(i)
                fields = line.split("\t")
                words = fields[4].split()
                s = ''
                ref_label = []
                for w in words:
                    index1 = w.find("/")
                    if index1 < 0:
                        index1 = len(w)
                        w += "/O"
                    word = w[0:index1]
                    label = w[index1+1:]
                    if label == 'B-OTH' or label == 'I-OTH':
                        label = 'O'
                    ref_label.append(label)

                    s += f"{word} "
                s = s.strip()
                s = s.split()

                data.append({
                    "input": s,
                    "label": ref_label,
                    "line_idx": i
                })

        return data
