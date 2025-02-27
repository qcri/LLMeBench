import base64
import json
import os

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class ArMemesDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ArMemesDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """to add""",
            "link": "",
            "license": "Research Purpose Only",
            "splits": {
                "dev": "arabic_memes_categorization_dev.jsonl",
                "test": "arabic_memes_categorization_test.jsonl",
                "train": "arabic_memes_categorization_train.jsonl",
            },
            "task_type": TaskType.Classification,
            "class_labels": ["propaganda", "not_propaganda", "not-meme", "other"],
        }

    @staticmethod
    def get_data_sample():
        return {"input": {"text": "text", "image": "base64"}, "label": "propaganda"}

    # Function to encode the image
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def load_data(self, data_path):
        data_path = self.resolve_path(data_path)

        base_dir = os.path.dirname(data_path)
        print(base_dir)
        data = []
        with open(data_path, "r") as fp:
            for line in fp:
                json_object = json.loads(line)
                id = json_object["id"]
                text = json_object["text"]
                img_path = base_dir + "/" + json_object["img_path"]
                class_label = json_object["class_label"]
                base64_image = self.encode_image(img_path)
                data.append(
                    {
                        "input": {
                            "text": text,
                            "image": base64_image,
                        },
                        "label": class_label,
                        "line_number": id,
                    }
                )

        return data
