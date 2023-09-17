# Updating Existing Assets

There are over 200 benchmarking assets within the framework. It is possible to start from any of them to perform further evaluation experiments. Below is one common use case that might be of interest, taking **sentiment classification** over the **ArSAS dataset** using **GPT4** as an example.

## Prompt Engineering
It is possible to study the performance of a model (e.g., GPT4) with different prompts as follows.
- Start from the asset [ArSAS_GPT4_ZeroShot.py](https://github.com/qcri/LLMeBench/blob/main/assets/ar/sentiment_emotion_others/sentiment/ArSAS_GPT4_ZeroShot.py)
- Create as many copies of it as the number of prompts to test, giving each a different name (e.g., ArSAS_GPT4_ZeroShot_v2.py, ArSAS_GPT4_ZeroShot_v3.py, etc)
- Change the prompt function in each according to the target prompt. For example, these are two versions of the prompt function:

<table>
<tr>
<th>Prompt V1</th>
<th>Prompt V2</th>
</tr>
<tr>
<td>

```python
def prompt(input_sample):
    return [
          {
             "role": "system",
             "content": "You are an AI assistant that helps \
                        people find information.",
          },
          {
              "role": "user",
              # Original prompt
              "content": f"Choose only one sentiment between: \
                          Positive, Negative, Neutral, \
                          or Mixed for this Arabic sentence: \
                          \n {input_sample}",
          }
      ]
```
</td>
<td>
  
``` python
def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "You are an AI assistant that helps \
                        people find information.",
        },
        {
            "role": "user",
            # Changed prompt for the task
            "content": f"Classify the given sentence by the \
                        sentiment it shows using one of these labels: \
                        Positive, Negative, Neutral, or Mixed.: \
                        \n {input_sample}",
        }
    ] 
  ```
</td>
</tr>
</table>

Then, run the following command (after specifying required environment variables for GPT4) to evaluate the different versions, where `'*ArSAS_GPT4_ZeroShot*'` will match all assets starting with that prefix.
```bash
python -m llmebench --filter '*ArSAS_GPT4_ZeroShot*' assets/ar/sentiment_emotion_others/sentiment/ results/
```

It is also possible to run such experiment by giving the updated asset files any name, then placing them in one folder (e.g., "arsas_prompt_testing") and running a command as follows: 
```bash
python -m llmebench arsas_prompt_testing/ results/
```
- `arsas_prompt_testing/`: This folder should be kept in the working directory of the framework or provide its full path as part of the command. 
