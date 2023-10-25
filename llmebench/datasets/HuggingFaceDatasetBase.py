import datasets

from .dataset_base import DatasetBase


class HuggingFaceDatasetBase(DatasetBase):
    def __init__(self, huggingface_model_name, **kwargs):
        self.huggingface_model_name = huggingface_model_name

        # Loading the dataset will automatically download it
        self.dataset = datasets.load_dataset(
            huggingface_model_name, cache_dir=kwargs["data_dir"]
        )

        super(HuggingFaceDatasetBase, self).__init__(**kwargs)

    @classmethod
    def download_dataset(cls, data_dir, download_url=None, default_url=None):
        # HuggingFace datasets library has its own data downloader
        pass
