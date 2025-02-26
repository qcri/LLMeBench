<<<<<<< HEAD
## Benchmarking huggingface models via FastChat
=======
## Benchmarking huggingface models via FastChat 
>>>>>>> e43d86b764f63d931638192f1b692ee4cd086147

FastChat supports a range of huggingface models and provide serving APIs. LLMeBench has an interface to FastChat APIs allowing to query and get responses from models.
Here is a quick start to load a huggingface model with FastChat and run an asset from LLMeBench on the model. 
### Install FastChat

```bash
pip3 install "fschat[model_worker,webui]"
```

## Serving models with Web GUI

#### Launch the controller
```bash
python3 -m fastchat.serve.controller
```

#### Launch the model worker(s)
```bash
python3 -m fastchat.serve.model_worker --model-path gpt2 --host localhost --port 5004
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...". The model worker will register itself to the controller .


#### Run the asset from LLMeBench

``` bash
ENGINE_NAME="gpt2" AZURE_API_KEY="EMPTY" AZURE_API_URL="http://localhost:5004/v1" python3 -m llmebench --filter "AraBench_Ara2Eng_FastChat_ZeroShot*" --ignore_cache assets/benchmark_v1/ results/
```
