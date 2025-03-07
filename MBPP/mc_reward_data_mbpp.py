import json
import re

import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
from vllm import LLM, SamplingParams
import util
import time
import os
import torch
from datasets import load_dataset
from transformers import HfArgumentParser
from pathlib import Path
from utils import check_gpu_memory

from huggingface_hub import login
login(token='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')

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
model_name = 'deepseek-ai/deepseek-coder-7b-instruct-v1.5'
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
# usage_guide: https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5
mbpp_path =  Path(__file__).parent/ "MBPP_eval"/ "data"/ 'mbpp.jsonl'
mbpp_raw = [json.loads(x) for x in open(mbpp_path)]

reason_data_path  = Path(__file__).parent/ "gen_reasoning_from_train_output"/"gen_reasoning_from_answer.json"

@dataclass
class ScriptArguments:
    completion_model_name_or_path: str = field(default=model_name, metadata={"help": "the completion model name or path locally or from huggingface."})
    dataset_path: str = field(default= reason_data_path, metadata={"help": "dataset path for generator data."})
    output_dir: str = field(default="gen_reward_data",metadata={"help":"location to store the PRM data."})
    tensor_parallel_size: int = field(default=1,metadata={"help":""}) # indicates the number of parallel processes, for distributed programming
    num_gpus: int = field(default=1)
    local_rank:int = field(default=0)
    sampling_num:int = field(default=16)
    split:int = field(default=0)

def get_question_test_list(mbpp_raw, task_id):
    mbpp_df = pd.DataFrame(mbpp_raw)
    mbpp_df['task_id'] = mbpp_df['task_id'].astype(str)
    task = mbpp_df.loc[mbpp_df['task_id'] == task_id]
    return task['text'].iloc[0], task['test_list'].iloc[0]

def process_reason_data(raw_dataset):
    new_dataset = []
    for sample in tqdm(raw_dataset):
        new_dataset.append({"prompt":sample['text'],"code":sample['code'], 'task_id': sample['task_id']})
    return new_dataset # change it from datasets objects to list of dict

def format_test_example(q, tests, code = None):
    prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(q.strip(), "\n".join(tests))
    if code:
        code = code.replace("\r", "").replace("\t", "    ")
        prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
    return prompt

def gen_prompt(question, steps):
    return f'''
    Your task is to infer and generate a complete 5-step reasoning process of a programming solution based on the given problem and the partial steps already provided (these are not the complete answer).
    Your output should consist of a complete 5-step reasoning process that logically leads to the solution.
    Let's think step by step. You will generate all remaining steps, ensuring clarity and systematic progression. 
    
    **Input format:**
    - {question}
    1. Step 1: {steps[0]}
    
    2. Step 2: {steps[1]}
    
    3. Step 3: {steps[2]}
    
    4. Step 4: {steps[3]}
    
    5. Step 5: {steps[4]}
    
    **Output format:**
    1. Step 1: {steps[0]}
    
    2. Step 2: {steps[1]}
    
    3. Step 3: {steps[2]}
    
    4. Step 4: {steps[3]}
    
    5. Step 5: {steps[4]}
    
    where ... indicates the omitted output information that you should fill in. Please provide the reasoning clearly and systematically.
    
    Please output according to the specified output format, do not output unnecessary information.
    If there are existing steps, generate the next steps to complete the reasoning up to a total of five steps.
    The complete solution should consist of a total of five steps, so ensure all essential steps are included.
    Please stop after you have finished providing the reasoning process for step 5.
    Each step should be concise and focus on a specific part of the reasoning process. Avoid making the steps too complex or combining multiple ideas into one.
    Each step should have one line and one title only.
    Ensure clarity and systematic progression in your explanations.
    The reasoning will help to decompose complex problems into a series of simpler subproblems and sequentially generate the corresponding steps to solve each subproblem.
    The steps you generate will be passed to a code generation model, so they should be structured in a way that is easy for the model to understand.
    Note: You are only allowed to describe the reasoning steps in natural language. Do not output any code.
    **If your answer includes code, it will cause unforeseen losses!**
    '''

def get_step_by_format(reasoning, step):
    steps = reasoning.split("\n\n")
    step_prefix = f"{step}. Step {step}: "
    # Iterate through the steps to find the one matching the prefix
    for step_text in steps:
        if step_text.startswith(step_prefix):
            return step_text[len(step_prefix):].strip() # Return the matching step, stripped of leading/trailing whitespace
    return "Step not found."

def gen_step_prompt(question, reasoning, step):
    steps  = ["..."] * 5
    for i in range(1, step+1):
        steps[i -1] = get_step_by_format(reasoning, i)

    return gen_prompt(question, steps)


if __name__ == '__main__':
    parser = HfArgumentParser((ScriptArguments))
    args = parser.parse_args_into_dataclasses()[0]
    with open(reason_data_path, 'r', encoding='utf-8') as file:
        reason_data = json.load(file)
    print("------------")
    print("begin to preprocess the sampling data")
    print("------------")
    processed_dataset = []
    dataset = reason_data[:] # delete after debug
    dataset = dataset[:20]

    #len(dataset) = number of task_id
    for sample in tqdm(dataset):
        # try:
        task_id = str(sample['task_id'])
        reasoning = sample['reasoning']
        question, test_list = get_question_test_list(mbpp_raw, task_id)
        question = format_test_example(question, test_list, code=None)

        for i in range(5): # this forloop is to make [A\n\n, A\n\nB\n\n, A\n\nB\n\nC\n'] out of the given answer in training
            prompt = gen_step_prompt(question, reasoning, step = i+1)
            processed_dataset.append({'prompt': prompt, 'task_id': task_id}) # this prompt and task id is correct

    stop_tokens = ["<|EOT|>"]
    sampling_params = SamplingParams(n=args.sampling_num, temperature=1, top_p=1, max_tokens=1024, stop=stop_tokens)
    print('sampling =====', sampling_params)
    llm = LLM(model=args.completion_model_name_or_path, tensor_parallel_size=args.tensor_parallel_size,
              enforce_eager=True, dtype="float16", gpu_memory_utilization=0.9, swap_space=64)
    print("------------")
    print("begin to label with markov process.")
    print("------------")

    prompt = [{"role": "user", "content": i['prompt']} for i in processed_dataset]
    tokenizer = llm.get_tokenizer()
    format_prompt = []
    for i in range(len(prompt)):
        conversations = tokenizer.apply_chat_template(
            [prompt[i]],
            tokenize=False,
            add_generation_prompt=True,
        )
        format_prompt.append({
            'conversations':conversations,
            'task_id': processed_dataset[i]['task_id'],
        })  # just format the text

    count = 0
    completions_list = []
    for batch_num in range(0, len(format_prompt), 200):  # doing batch is just for computational faster
        batch = format_prompt[batch_num:batch_num + 200]

        batch_converstaions = [b['conversations'] for b in batch]
        completions = llm.generate(batch_converstaions, sampling_params)  # sampling completion
        for j, output in enumerate(completions):
            prompt_temp = output.prompt
            generated_text = [output.outputs[i].text for i in
                              range(len(output.outputs))]  # get rid of wrappers and becomes list of string (of each completionm ie there are multiple (16) completions)

            entry = {
                "task_id":processed_dataset[count]['task_id'],
                "prompt": processed_dataset[count]['prompt'],
                "completions": generated_text
            }
            print("Processing task_id: ", entry["task_id"], "prompt: ", entry["prompt"], "completions ", entry["completions"])
            completions_list.append(entry)
            count += 1

    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/data_mbpp_split{args.split}_{args.local_rank}.json", 'w') as f:
        json.dump(completions_list, f, indent=4, ensure_ascii=False)
        # except Exception as e:
        #     print(f"An error occurred: {e} in sample: {sample}")
        # pd.DataFrame(completions_list).to_csv('reward_data.csv')