import base64
import json
import os

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class HatefulMemesDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(HatefulMemesDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "en",
            "citation": """@article{kiela2020hateful,
            title={The hateful memes challenge: Detecting hate speech in multimodal memes},
            author={Kiela, Douwe and Firooz, Hamed and Mohan, Aravind and Goswami, Vedanuj and Singh, Amanpreet and Ringshia, Pratik and Testuggine, Davide},
            journal={Advances in neural information processing systems},
            volume={33},
            pages={2611--2624},
            year={2020}
            }""",
            "link": "https://ai.meta.com/tools/hatefulmemes/",
            "license": "Research Purpose Only",
            "splits": {
                "train": "train.jsonl",
                "dev": "dev_unseen.jsonl",
                "test": "test_unseen.jsonl",
            },
            "task_type": TaskType.Classification,
            "class_labels": ["hateful", "not-hateful"],
        }

    @staticmethod
    def get_data_sample():
        return {"input": {"text": "text", "image": "base64"}, "label": "hateful"}

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
                img_path = base_dir + "/" + json_object["img"]
                label = json_object["label"]
                class_label = "hateful" if label == 1 else "not-hateful"
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
