from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class LemmatizationDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(LemmatizationDataset, self).__init__(**kwargs)

    def citation(self):
        return """@inproceedings{mubarak2018build,
          title={Build Fast and Accurate Lemmatization for Arabic},
          author={Mubarak, Hamdy},
          booktitle={Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
          year={2018}
        }
        """

    def get_data_sample(self):
        return {
            "input": "فيلم جاذبية يتصدر ترشيحات جوائز",
            "label": "فيلم جاذبية تصدر ترشيح جائزة",
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
                    data.append(
                        {"input": text, "label": label, "line_number": line_idx}
                    )

        return data
