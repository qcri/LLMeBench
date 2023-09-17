import random

from llmebench.models.model_base import ModelBase

from llmebench.tasks import TaskType


class RandomModel(ModelBase):
    def __init__(self, task_type, **kwargs):
        self.task_type = task_type
        if self.task_type == TaskType.Classification:
            self.class_labels = kwargs["class_labels"]
        elif self.task_type == TaskType.SequenceLabeling:
            self.class_labels = kwargs["class_labels"]
        elif self.task_type == TaskType.MultiLabelClassification:
            self.class_labels = kwargs["class_labels"]
        elif self.task_type == TaskType.Regression:
            self.score_range = kwargs["score_range"]
        else:
            self.task_type = TaskType.Other

        random.seed(2023)

        super(RandomModel, self).__init__(**kwargs)

    def summarize_response(self, response):
        if "random_response" in response:
            return response["random_response"]

        return None

    def prompt(self, processed_input):
        if self.task_type == TaskType.Classification:
            random_response = random.choice(self.class_labels)
        elif self.task_type == TaskType.SequenceLabeling:
            assert isinstance(
                processed_input, str
            ), "RandomModel only works with string `input` for labeling tasks"
            random_response = " ".join(
                [random.choice(self.class_labels) for _ in processed_input.split()]
            )
        elif self.task_type == TaskType.MultiLabelClassification:
            random_response = [
                label for label in self.class_labels if random.random() > 0.5
            ]
        elif self.task_type == TaskType.Regression:
            min_val, max_val = self.score_range
            random_response = min_val + random.random() * (max_val - min_val)
        else:
            random_response = processed_input

        # elif self.task_type == "labeling-index":
        #     assert isinstance(
        #         processed_input, str
        #     ), "RandomModel only works with string `input` for labeling tasks"
        #     tokens = processed_input.split()
        #     random_labels = [str(idx) for idx in range(len(tokens) + 1)]
        #     random_response = {
        #         str(idx + 1): random.choice(random_labels) for idx in range(len(tokens))
        #     }
        # elif self.task_type == "regression":
        #     min_val, max_val = self.score_range
        #     random_response = min_val + random.random() * (max_val - min_val)
        # elif self.task_type == "multilabel-list":

        # elif self.task_type == "multilabel-length":
        #     random_response = [
        #         random.choice(self.class_labels) for _ in range(self.num_labels)
        #     ]
        # elif self.task_type == "identity":
        #     random_response = processed_input
        # else:
        #     raise ValueError(f"Unsupported task_type {self.task_type} in RandomModel")

        return {"random_response": random_response}
