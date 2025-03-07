import argparse
import json
import os
import torch
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


data_abs_dir = Path(__file__).parent/ "MBPP_eval"/ "data"
model_name = 'deepseek-ai/deepseek-coder-7b-instruct-v1.5'

def extract_steps(text):
    # Regular expression to match steps
    steps = re.findall(r'(\d+\.\sStep\s\d+:.*?)(?=\n\d+\.\sStep\s\d+:|\Z)', text, re.DOTALL)

    # Clean up the steps and return them
    return [step.strip() for step in steps]


def read_test_examples(data_path: str):
    def format_test_example(q, tests, code: str = None):
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(q.strip(), "\n".join(tests))
        if code:
            code = code.replace("\r", "").replace("\t", "    ")
            prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
        return prompt

    examples = [json.loads(x) for x in open(data_path)]
    print("Read all {} examples from {} over!".format(len(examples), data_path))

    for i in tqdm(range(510), desc="Processing"):  # 510
        ex = examples[i]
        q, test, code = ex['text'], ex['test_list'], ex['code']

        prompt = format_test_example(q, test, code=code)  # after providing the example, this is none

        prompt_with_shots = f'''
        You are tasked with inferring the reasoning behind a coding solution based on the given problem.
        Your output should consist of a complete 5-step reasoning process that logically leads to the solution.

        **Input format:**
        - Problem: {prompt}

        **Output format:**
        1. Step 1: 
        2. Step 2: 
        3. Step 3: 
        4. Step 4: 
        5. Step 5: 

        Each step should be concise and focus on a specific part of the reasoning process. 
        Ensure clarity and systematic progression in your explanations.
        The reasoning will help to decompose complex problems into a series of simpler subproblems and sequentially generate the corresponding steps to solve each subproblem.
        The steps you generate will be passed to a code generation model, so they should be structured in a way that is easy for the model to understand.
        Note: You are only allowed to describe the reasoning steps in natural language. Do not output any code
        '''.strip()
        yield {
            'task_id': ex['task_id'],
            'prompt': prompt_with_shots
        }
def generate_one(example, tokenizer, model):
    prompt = example['prompt']
    inputs = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt }],
        return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    stop_id = tokenizer.convert_tokens_to_ids("<|EOT|>")
    assert isinstance(stop_id, int), "Invalid tokenizer, EOT id not found"
    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        do_sample=False,
        # top_p=0.95,
        # temperature=temperature,
        pad_token_id=stop_id,
        eos_token_id=stop_id
    )

    output = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    # print(output)
    example['gpt_reasoning'] = output
    return example


def generate_main(args):
    model_name_or_path = args.model
    output_folder = args.output_folder
    problem_file = os.path.join(data_abs_dir, f"mbpp.jsonl")

    print("model", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    print("load tokenizer {} from {} over.".format(tokenizer.__class__, model_name_or_path))
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    examples = list(read_test_examples(problem_file))
    print("Read {} examples for evaluation over.".format(len(examples)))

    examples = examples[20:50]

    generated_examples = []
    for ex in tqdm(examples, desc='Generating'):
        gen_example = generate_one(ex, tokenizer, model)
        gen_example_to_json = {}
        gen_example_to_json['task_id'] = gen_example['task_id']
        reasoning_ls = extract_steps(gen_example['gpt_reasoning'])
        gen_example_to_json['reasoning'] = "\n\n".join(reasoning_ls)

        generated_examples.append(gen_example_to_json)
        print("Generate {}/{} over...".format(len(generated_examples), len(examples)))

    print("Generate all over!!!")

    file_path = os.path.join(output_folder, args.file_name)
    os.makedirs(output_folder, exist_ok=True)
    with open(file_path, 'w') as fw:
        json.dump(generated_examples, fw, indent=4, ensure_ascii=False)
        print("Save {} processed examples into {} over!".format(len(generated_examples), file_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=model_name, type=str, help="model name or path")
    parser.add_argument('--output_folder', default="gen_reasoning_from_train_output", type=str, help="output path of your generation")
    parser.add_argument('--file_name', default="gen_reasoning_from_answer.json", type=str, help="file name of your generation")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    generate_main(args)
    pass
