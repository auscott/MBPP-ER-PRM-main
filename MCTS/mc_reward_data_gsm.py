import json
import re
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
from vllm import LLM, SamplingParams
import spacy
# from eval_gsm8k import is_number, extract_answer_number
# from eval_math import remove_boxed, process_results
import util
import time
import os
from util import is_number, extract_answer_number, remove_boxed, process_results
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, pipeline

from huggingface_hub import login
login(token='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')

# Set the environment variable
import os
import gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ['VLLM_TARGET_DEVICE'] = 'cpu'
# model_name = 'deepseek-ai/deepseek-math-7b-rl' # occupied 42782.00 MB this also works
# model_name = 'codellama/CodeLlama-7b-Python-hf'  # occupied 42002.00 MB this also works
torch.cuda.empty_cache()
# gc.collect()
# del llm
# model_name = 'meta-llama/Llama-3.2-1B-Instruct' # this work for 48GB GPU
model_name = 'deepseek-ai/deepseek-coder-7b-instruct-v1.5'

def check_gpu_memory():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # Convert to MB
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
        cached_memory = torch.cuda.memory_reserved() / (1024 ** 2)  # Convert to MB

        print(f"Total GPU Memory: {total_memory:.2f} MB")
        print(f"Allocated Memory: {allocated_memory:.2f} MB")
        print(f"Cached Memory: {cached_memory:.2f} MB")
    else:
        print("No GPU available.")

check_gpu_memory()

@dataclass
class ScriptArguments:
    completion_model_name_or_path: str = field(default=model_name, metadata={"help": "the completion model name or path locally or from huggingface."})
    dataset_path: str = field(default="HanningZhang/deepseek-gsm-new", metadata={"help": "dataset path for generator data."})
    output_dir: str = field(default="mc_data",metadata={"help":"location to store the PRM data."})
    tensor_parallel_size: int = field(default=1,metadata={"help":""}) # indicates the number of parallel processes, for distributed programming
    num_gpus: int = field(default=1)
    local_rank:int = field(default=0)
    sampling_num:int = field(default=16)
    split:int = field(default=0)

def process_dataset(raw_dataset):
    new_dataset = []
    for sample in tqdm(raw_dataset):
        new_dataset.append({"prompt":sample['prompt'],"answer":sample['answer']})
    return new_dataset # change it from datasets objects to list of dict

def check_math_answer(content,ground_truth):
    
    split_ans = content.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('ки')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        
        gt_ans = remove_boxed(util.last_boxed_only_string(ground_truth))
        #print(f"gt ans is {gt_ans}")
        # gt_ans = ground_truth.split('The answer is: ')
        # gt = gt_ans[-1]
        # extract_gt_temp = gt.split('ки')[0]
        # extract_gt_temp = extract_gt_temp.strip()
        # if len(extract_gt_temp)>0 and extract_gt_temp[-1] == '.':
        #     extract_gt = extract_gt_temp[0:-1]
        # else:
        #     extract_gt = extract_gt_temp
        # extract_gt = extract_gt.strip()
        if util.is_equiv(extract_ans, gt_ans):
            return True
        else:
            return False
    else:
        return False
    
def check_answer(sample,content,ground_truth):

    if "GSM" in sample['task']:
        temp_ans = sample['ground_truth'].split('#### ')[1]
        temp_ans = int(temp_ans.replace(',', ''))
        #print(float(extract_answer_number(sample['ground_truth'])))
        if not extract_answer_number(content):
            return 1
        if float(extract_answer_number(content)) == float(temp_ans):
            label = 0
        else:
            label = 1
    else:
        if check_math_answer(content,sample['ground_truth']):
            label = 0
        else:
            label = 1
            
    return label
    

def generate_completion(llm,sampling_params,sample,prompt,ground_truth):
    
    #label = None    ## 0 denotes good completion. 1 denotes bad completion.
    label_list = []
    if isinstance(prompt, list):
        pass
    else:
        prompt = [prompt]

    completions = llm.generate(prompt, sampling_params)
    completions_list = []
    for output in completions:
        prompt_temp = output.prompt
        generated_text = [output.outputs[i].text for i in range(len(output.outputs))]
        completions_list.append([prompt_temp,generated_text])

    for completion in completions_list:
        results = [check_answer(sample,content,ground_truth) for content in completion[1]]
    
        if 0 in results:
            label = 0
        else:
            label = 1
        
        label_list.append({"prompt":completion[0],"label":label})
        
    return label_list

if __name__ == "__main__":
    
    parser = HfArgumentParser((ScriptArguments))
    args = parser.parse_args_into_dataclasses()[0]

    nlp = spacy.load("en_core_web_sm")
    raw_dataset = load_dataset(args.dataset_path,split='train')
    dataset = process_dataset(raw_dataset)
    print("------------")
    print("begin to preprocess the sampling data")
    print("------------")
    
    #step_tag = 'ки' 
    processed_dataset = []
    dataset = dataset[:] # this do nothing?
    # if args.split == 0:
    #     dataset = dataset[:int(0.5*len(dataset))]
    # else:
    #     dataset = dataset[int(0.5*len(dataset)):]
    dataset = dataset[int((args.local_rank)/args.num_gpus * len(dataset)):int((args.local_rank+1)/args.num_gpus * len(dataset))]
    dataset = dataset[:5]



    for sample in tqdm(dataset):
        prompt = sample['prompt']
        answer = sample['answer']
        doc = nlp(answer)
        sentences = [sent.text for sent in doc.sents] # this change the whole text to list of string, break down sentence
        # step_list = sample.split('\n')
        # step_list = [i.strip() for i in step_list]
        # temp_str = ""
        for i in range(len(sentences)-1): # this forloop is to make [A\n\n, A\n\nB\n\n, A\n\nB\n\nC\n'] out of the given answer in training
            prompt += sentences[i] + "\n"
            processed_dataset.append(prompt)
        prompt += sentences[-1]
        processed_dataset.append(prompt) # this has structure of [A\n\n, A\n\nB\n\n, A\n\nB\n\nC\n'], the last on does have double \n, no \n and next question start with itself, not including previous answer
    
    stop_tokens = []
    sampling_params = SamplingParams(n=args.sampling_num, temperature=1, top_p=1, max_tokens=1024, stop=stop_tokens)
    print('sampling =====', sampling_params)
    # llm = LLM(model=args.completion_model_name_or_path, enforce_eager=True, dtype = "float32",swap_space=64)
    llm = LLM(model=args.completion_model_name_or_path,tensor_parallel_size=args.tensor_parallel_size, enforce_eager=True, dtype = "float16",gpu_memory_utilization=0.9,swap_space=64)
    print("------------")
    print("begin to label with markov process.")
    print("------------")
    
    prompt = [{"role":"user","content":i+"\nPlease reason step by step, and put your final answer within \\boxed{}."} for i in processed_dataset]

    tokenizer = llm.get_tokenizer()
    format_prompt = []
    for i in prompt[:]:
        conversations = tokenizer.apply_chat_template(
            [i],
            tokenize=False,
            add_generation_prompt=True, 
        )
        format_prompt.append(conversations) # just format the text
    count = 0
    completions_list = []
    for batch_num in range(0,len(format_prompt),3000): # doing batch is just for computational faster
        batch = format_prompt[batch_num:batch_num+3000]
        completions = llm.generate(batch, sampling_params) # sampling completion
        for j,output in enumerate(completions):
            prompt_temp = output.prompt
            generated_text = [output.outputs[i].text for i in range(len(output.outputs))] # get rid of wrappers and becomes list of string (of each completionm ie there are multiple (16) completions)
            if "The answer is" in processed_dataset[count]:
                completions_list.append({"prompt":processed_dataset[count],"completions":["" for i in range(len(generated_text))]}) # if "The answer is in prompt, then dun record the generated text
            else:
                completions_list.append({"prompt":processed_dataset[count],"completions":generated_text})
            count += 1
        
        os.makedirs(args.output_dir,exist_ok=True)
        with open(f"{args.output_dir}/data_gsm_split{args.split}_{args.local_rank}.json",'w') as f:
            json.dump(completions_list,f,indent=4,ensure_ascii=False)