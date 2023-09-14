import os

from llmebench.models.OpenAI import OpenAIModel


class FastChatModel(OpenAIModel):
    def __init__(self, api_base, api_key, model_name, **kwargs):
        api_base = api_base or os.getenv("FASTCHAT_API_BASE")
        api_key = api_key or os.getenv("FASTCHAT_API_KEY")
        model_name = model_name or os.getenv("FASTCHAT_MODEL_NAME")
        # checks for valid config settings)
        super(FastChatModel, self).__init__(
            api_base=api_base, api_key=api_key, model_name=model_name, **kwargs
        )
