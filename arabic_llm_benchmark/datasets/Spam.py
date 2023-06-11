from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class SpamDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(SpamDataset, self).__init__(**kwargs)

    def citation(self):
        return """@inproceedings{mubarak2020spam,
          title={Spam detection on arabic twitter},
          author={Mubarak, Hamdy and Abdelali, Ahmed and Hassan, Sabit and Darwish, Kareem},
          booktitle={Social Informatics: 12th International Conference, SocInfo 2020, Pisa, Italy, October 6--9, 2020, Proceedings 12},
          pages={237--251},
          year={2020},
          organization={Springer}
        }
        """

    def get_data_sample(self):
        return {"input": "أختر قلباً وليسّ شكلاً..", "label": "__label__NOTADS"}

    def load_data(self, data_path, no_labels=False):
        # TODO: modify to iterator
        #Format: spam_label \t text
        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                label, text = line.split("\t")
                data.append(
                    {"input": text, "label": label, "line_number": line_idx}
                )

        return data
