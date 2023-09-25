import regex as re

from llmebench.datasets import SemEval23T3PropagandaDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import MultilabelPropagandaTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
    }


def config():
    return {
        "dataset": SemEval23T3PropagandaDataset,
        "dataset_args": {"techniques_path": "techniques_subtask3.txt"},
        "task": MultilabelPropagandaTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": [
                "Appeal_to_Authority",
                "Appeal_to_Fear-Prejudice",
                "Appeal_to_Hypocrisy",
                "Appeal_to_Popularity",
                "Appeal_to_Time",
                "Appeal_to_Values",
                "Causal_Oversimplification",
                "Consequential_Oversimplification",
                "Conversation_Killer",
                "Doubt",
                "Exaggeration-Minimisation",
                "False_Dilemma-No_Choice",
                "Flag_Waving",
                "Guilt_by_Association",
                "Loaded_Language",
                "Name_Calling-Labeling",
                "Obfuscation-Vagueness-Confusion",
                "Questioning_the_Reputation",
                "Red_Herring",
                "Repetition",
                "Slogans",
                "Straw_Man",
                "Whataboutism",
                "no_technique",
            ],
            "max_tries": 30,
        },
        "general_args": {"test_split": "pl/dev"},
    }


def prompt(input_sample):
    prompt_string = f"""We're seeking your analytical insights to dissect the following text excerpt from a news article. Please categorize it, determining which propaganda techniques are at play. You should provide your response as a list of strings representing the techniques you identify.

        If you conclude that no particular technique is in evidence, include 'no_technique' in your list. Below, you'll find a list of possible techniques to guide your analysis:

        'no_technique', 'Appeal_to_Authority', 'Appeal_to_Fear-Prejudice', 'Appeal_to_Hypocrisy', 'Appeal_to_Popularity', 'Appeal_to_Time', 'Appeal_to_Values', 'Causal_Oversimplification', 'Consequential_Oversimplification', 'Conversation_Killer', 'Doubt', 'Exaggeration-Minimisation', 'False_Dilemma-No_Choice', 'Flag_Waving', 'Guilt_by_Association', 'Loaded_Language', 'Name_Calling-Labeling', 'Obfuscation-Vagueness-Confusion', 'Questioning_the_Reputation', 'Red_Herring', 'Repetition', 'Slogans', 'Straw_Man', 'Whataboutism'.

        Text for analysis:
        {input_sample}

        Please provide the applicable labels below:
        """

    return [
        {
            "role": "system",
            "content": "You are an expert social media content analyst.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def fix_label(pred_label):
    # Load class labels from config
    class_labels = config()["model_args"]["class_labels"]
    class_labels = [c.lower() for c in class_labels]

    pred_labels_bool = [bool(re.search(c.lower(), pred_label)) for c in class_labels]
    pred_labels = [class_labels[i].lower() for i, c in enumerate(pred_labels_bool) if c]

    # Define a function to process each label
    def process_label(label):
        label = label.replace(".", "").strip().lower()
        label = re.sub("-", " ", label)
        return label

    label_mappings = {
        "slogan": "Slogans",
        "false_dilemma_no_choice": "False_Dilemma-No_Choice",
        "false_dilemma no_choice": "False_Dilemma-No_Choice",
        "conversation_killer": "Conversation_Killer",
        "questioning_the_reputation": "Questioning_the_Reputation",
        "conversation killer": "Conversation_Killer",
        "appeal_to_popularity": "Appeal_to_Popularity",
        "appeal_to_hypocrisy": "Appeal_to_Hypocrisy",
        "appeal_to_values": "Appeal_to_Values",
        "guilt_by_association": "Guilt_by_Association",
        "appeal_to_time": "Appeal_to_Time",
        "loaded": "Loaded_Language",
        "prejudice": "Appeal_to_Fear-Prejudice",
        "fear": "Appeal_to_Fear-Prejudice",
        "mongering": "Appeal_to_Fear-Prejudice",
        "terminating": "Thought-terminating cliché",
        "thought": "Thought-terminating cliché",
        "calling": "Name_Calling-Labeling",
        "name c": "Name_Calling-Labeling",
        "minimisation": "Exaggeration-Minimisation",
        "exaggeration minim": "Exaggeration-Minimisation",
        "glittering": "Appeal_to_Values",
        "flag": "Flag_Waving",
        "obfuscation": "Obfuscation-Vagueness-Confusion",
        "oversimplification": "Causal_Oversimplification",
        "causal": "Causal_Oversimplification",
        "authority": "Appeal_to_Authority",
        "dictatorship": "False_Dilemma-No_Choice",
        "black": "False_Dilemma-No_Choice",
        "white": "False_Dilemma-No_Choice",
        "herring": "Red_Herring",
        "irrelevant": "Red_Herring",
        "straw": "Straw_Man",
        "misrepresentation": "Straw_Man",
        "whataboutism": "Whataboutism",
    }

    # Define a set for labels that should be set to 'no_technique'
    no_technique_keywords = {
        "no propaganda",
        "technique",
        "",
        "no",
        "appeal to history",
        "no_technique",
        "appeal to emotion",
        "appeal to",
        "appeal",
        "appeal to author",
        "emotional appeal",
        "no techn",
        "hashtag",
        "theory",
        "specific mention",
        "religious",
        "gratitude",
    }

    labels_fixed = []
    for label in pred_labels:
        label_processed = process_label(label)

        # Handle special cases using the dictionary
        matched = False
        for key, value in label_mappings.items():
            if key in label_processed:
                labels_fixed.append(value)
                matched = True
                break

        # If no special case matched, use default behavior
        if not matched:
            if label_processed in no_technique_keywords:
                labels_fixed.append("no_technique")
            else:
                labels_fixed.append(label_processed.capitalize())

    # Remove 'no_technique' label when we have other techniques for the same text
    if len(labels_fixed) > 1 and "no_technique" in labels_fixed:
        labels_fixed.remove("no_technique")

    return labels_fixed


def post_process(response):
    label = response["choices"][0]["message"]["content"].strip()  # .lower()
    label = label.replace("'label:", "")

    try:
        if label == "'no_technique'":
            pred_label = ["no_technique"]
        else:
            label = label.replace("'", "").replace(" ", "")
            pred_label = label.split(",")
    except Exception as ex:
        pred_label = fix_label(label.strip())

    return pred_label
