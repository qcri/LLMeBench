import datasets

from llmebench.datasets.dataset_base import DatasetBase


class HuggingFaceDataset(DatasetBase):
    """
    Generic HuggingFace dataset loader

    This data loader provides a way to load datasets on HuggingFace Hub and transform
    them into the format required by the framework. Assets using this loader *must*
    provide a `custom_test_split`, which should correspond to a split in the dataset
    as defined on the Hub. Similarly, `custom_train_split` must also be provided for
    few shot assets.

    Attributes
    ----------
    data_dir : str
        Base path of data containing all datasets. Defaults to "data" in the current
        working directory.
    huggingface_dataset_name : str
        Name of the dataset on HuggingFace Hub (e.g. 'sst2')
    column_mapping : dict
        Mapping defining which of the columns in the loaded data are "input" and "label".
        The supplied dict must contain mappings for "input" and "label", and may contain
        other mappings (such as "input_id").
    """

    def __init__(self, huggingface_dataset_name, column_mapping, **kwargs):
        self.huggingface_dataset_name = huggingface_dataset_name

        # Check for column_mapping
        assert "input" in column_mapping
        assert "label" in column_mapping
        self.column_mapping = column_mapping

        super(HuggingFaceDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "citation": """@inproceedings{Lhoest_Datasets_A_Community_2021,
                author = {Lhoest, Quentin and Villanova del Moral, Albert and von Platen, Patrick and Wolf, Thomas and Šaško, Mario and Jernite, Yacine and Thakur, Abhishek and Tunstall, Lewis and Patil, Suraj and Drame, Mariama and Chaumond, Julien and Plu, Julien and Davison, Joe and Brandeis, Simon and Sanh, Victor and Le Scao, Teven and Canwen Xu, Kevin and Patry, Nicolas and Liu, Steven and McMillan-Major, Angelina and Schmid, Philipp and Gugger, Sylvain and Raw, Nathan and Lesage, Sylvain and Lozhkov, Anton and Carrigan, Matthew and Matussière, Théo and von Werra, Leandro and Debut, Lysandre and Bekman, Stas and Delangue, Clément},
                booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
                month = nov,
                pages = {175--184},
                publisher = {Association for Computational Linguistics},
                title = {{Datasets: A Community Library for Natural Language Processing}},
                url = {https://aclanthology.org/2021.emnlp-demo.21},
                year = {2021}
            }""",
            "link": "https://huggingface.co/datasets/",
            "generic": True,
        }

    @staticmethod
    def get_data_sample():
        return {"input": "Test Input", "label": "0"}

    def load_data(self, data_split, no_labels=False):
        dataset = datasets.load_dataset(
            self.huggingface_dataset_name, split=data_split, cache_dir=self.data_dir
        )

        data = []
        for sample in dataset:
            processed_sample = {}
            for sample_key, column_name in self.column_mapping.items():
                processed_sample[sample_key] = sample[column_name]
            data.append(processed_sample)

        return data

    @classmethod
    def download_dataset(cls, data_dir, download_url=None, default_url=None):
        # HuggingFace datasets library has its own data downloader
        pass
