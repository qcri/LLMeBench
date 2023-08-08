from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class BnSentimentDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(BnSentimentDataset, self).__init__(**kwargs)

    def citation(self):
        return """
            @article{alam2021review,
              title={A Review of Bangla Natural Language Processing Tasks and the Utility of Transformer Models},
              author={Alam, Firoj and Hasan, Md Arid and Alam, Tanvir and Khan, Akib and Tajrin, Janntatul and Khan, Naira and Chowdhury, Shammur Absar},
              journal={arXiv preprint arXiv:2107.03844},
              year={2021}
            }        
            @inproceedings{iccit2020Arid,
                Author = {Md. Arid Hasan and Jannatul Tajrin and Shammur Absar Chowdhury and Firoj Alam},
                Booktitle = {23rd International Conference on Computer and Information Technology (ICCIT)},
                Month = {December},
                Title = {Sentiment Classification in Bangla Textual Content: A Comparative Study},
                Year = {2020},
                url={https://github.com/banglanlp/bangla-sentiment-classification},
            }
        """

    def get_data_sample(self):
        return {"input": "Tweet", "label": "Positive"}

    def load_data(self, data_path):
        data = []
        with open(data_path, "r") as fp:
            next(fp)
            for line_idx, line in enumerate(fp):
                id, text, label = line.strip().split("\t")
                label = label.capitalize()
                data.append({"input": text, "label": label, "line_number": id})

        return data
