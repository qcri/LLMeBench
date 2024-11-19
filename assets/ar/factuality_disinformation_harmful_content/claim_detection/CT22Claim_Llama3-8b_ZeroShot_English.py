from llmebench.datasets import CT22ClaimDataset
from llmebench.models import AzureModel
from llmebench.tasks import ClaimDetectionTask




def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "Llama-3.1-8B-Instruct",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054)."
    }




def config():
    return {
        "dataset": CT22ClaimDataset,
        "task": ClaimDetectionTask,
        "model": AzureModel,
        "model_args": {"max_tries": 30},
        "general_args": {"test_split": "ar"},
    }

def prompt(input_sample):
    return [
        {
            "role": "user",
            "content": (
                "Does this sentence contain a factual claim? Please answer with 'yes' or 'no' only.\n\n"
                + f"Sentence: {input_sample}\n"
                + "Label: "
            ),
        }
    ]
import random
def post_process(response):
    try:
        label = ""

        # Assuming 'response' contains an 'output' directly. Adjust if structure differs.
        if "output" in response:
            label = response["output"].strip().lower()

        # Debug print to check the extracted label
        print(f"Extracted Label: {label}")
        if "لا أستطيع" in label or "I cannot" in label:
            return random.choice(["0","1"])

        # Determining the prediction label based on the response content
        if "yes" in label or "contains a factual claim" in label or "label: 1" in label:
            pred_label = "1"
        elif "no" in label or "label: 0" in label or "does not contain a factual claim" in label or "label: no" in label:
            pred_label = "0"
        else:
            # If none of the expected labels are found, default to a negative claim (most conservative approach)
            pred_label = "0"

        # Debug print to check the final predicted label
        print(f"Predicted Label: {pred_label}")

        return pred_label
    except Exception as e:
        print(f"Error in post-processing: {str(e)}")
        # Return a default negative label in case of error to prevent unknown targets
        return "0"
