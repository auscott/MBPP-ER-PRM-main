import json
import re

import pandas as pd
from tqdm import tqdm
from pathlib import Path
import os
import math
from MBPP_eval.human_eval.evaluation import evaluate_functional_correctness
from huggingface_hub import login
login(token="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")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = 'deepseek-ai/deepseek-coder-7b-instruct-v1.5'
saved_path = 'generated_format_data_score.json'
temp_dir = 'generated_format_data_test_result'
data_abs_dir = Path(__file__).parent / "MBPP_eval"/ "data"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

score_data_filename = 'mbpp_score_data.json'
mcts_filename = 'mbpp_mcts_data.json'

# this script is to output prompt label pair, giving score for each prompt
# format : [{'text': 'some random prompt', 'label': 0.723}, {'text': 'another random prompt', 'label': 0.323}, ....]
# each sentence is seperated by '\n\n'


# then it will also create the mcit data :
# [
#     {
#         "text": "Janet pays $40/hour for 3 hours per week of clarinet lessons.."
#         "value": [0.8, 0.6, 0.3 ...]
#     },
#     {
#         "text": "Val cuts a single watermelon into 40 slices.."
#         "value": [0.5, 0.7, 0.2 ...]
#     }
# ]

# example of the text: Janet pays $40/hour for 3 hours per week of clarinet lessons and $28/hour for 5 hours a week of piano lessons. How much more does she spend on piano lessons than clarinet lessons in a year? Step 1: Janet spends 3 hours + 5 hours = <<3+5=8>>8 hours per week on music lessons. ки\nStep 2: She spends 40 * 3 = <<40*3=120>>120 on clarinet lessons per week. ки\nStep 3: She spends 28 * 5 = <<28*5=140>>140 on piano lessons per week. ки\nStep 4: Janet spends 120 + 140 = <<120+140=260>>260 on music lessons per week. ки\nStep 5: She spends 260 * 52 = <<260*52=13520>>13520 on music lessons in a year. The answer is: 13520 ки

def extract_steps(prompt):
    steps = []
    current_step = ""

    # Split the prompt by new lines to process each line
    for line in prompt.split("\n"):
        line = line.strip()  # Clean up the line

        # Check if the line contains a step number
        if re.match(r'^\d+\.\s*Step \d+:', line):
            if current_step:  # If we have a current step, save it before starting a new one
                steps.append(current_step.strip())
            current_step = line  # Start a new step
        elif current_step:  # If we are in a step
            if "..." in line:
                current_step += " " + line  # Append the line containing "..."
                steps.append(current_step.strip())  # Save the step
                current_step = ""  # Reset for the next step
            else:
                current_step += " " + line  # Append normal content

    # Append the last step if it exists
    if current_step:
        steps.append(current_step.strip())

    # Clean up steps to keep only the necessary parts
    for i in range(len(steps)):
        if "..." in steps[i]:
            steps[i] = steps[i].split("...")[0] + "..."  # Keep only up to "..."

    return steps

def classify_steps_and_return_numbers(input_steps):
    filled_step_numbers = []
    placeholder_step_numbers = []

    # Iterate through each step in the input list
    for step in input_steps:
        # Extract the step number using regex
        step_number = step.split('.')[0]  # Get the part before the first dot
        if "..." in step:
            placeholder_step_numbers.append(step_number.strip())
        else:
            filled_step_numbers.append(step_number.strip())
    return filled_step_numbers, placeholder_step_numbers

def get_step_ls(prompt):
    prompt_ls = extract_steps(prompt)
    filled_step_numbers, unfilled_step_numbers = classify_steps_and_return_numbers(prompt_ls)
    all_steps = set(filled_step_numbers + unfilled_step_numbers)
    required_steps = {'1', '2', '3', '4', '5'}

    if required_steps.issubset(all_steps):
        return filled_step_numbers, unfilled_step_numbers, prompt_ls
    else:
        return None, None, None
def get_prompt_step(prompt):
    start_index = prompt.find("**Output format:**")
    if start_index == -1:
        return ""
    # Locate the end of the relevant section
    end_index = prompt.find("where ... indicates the omitted output information that you should fill in.", start_index)
    if end_index == -1:
        return ""  # Return empty if the end phrase is not found

    relevant_part = prompt[start_index:end_index + len(
        "where ... indicates the omitted output information that you should fill in.")]
    return relevant_part.strip()

def get_quality_data(data):
    # check completion quality (must have step 1, step 2, step 3, step 4, step 5)
    # get prompt and see the step from prompt
    # combine with the rest of the gen completion
    task_id = data.get("task_id")
    prompt = data.get("prompt")
    completions = data.get("completions", [])

    core_prompt = get_prompt_step(prompt)
    filled_step_numbers, unfilled_step_numbers, prompt_ls = get_step_ls(core_prompt)
    if prompt_ls is None:# bad completions
        return None

    filled_reasonsing_completions = []
    for completion in completions[:]:
        # Check if the completion contains all required steps
        _, _, completions_ls = get_step_ls(completion)

        if completions_ls is None: # bad completions
            continue

        # filled the prompt ls:
        filled_reasonsing = prompt_ls[:]
        for i in (unfilled_step_numbers):
            filled_reasonsing[int(i) - 1] = completions_ls[int(i) - 1]
        filled_reasonsing_completions.append(
            {"filled_reasonsing" : filled_reasonsing,
             "task_id" :task_id,
             "prompt_ls": prompt_ls
             }
        )
    return filled_reasonsing_completions

def format_test_for_quality_data(quality_data):

    def give_example_str():
        examples_str = []
        for i in range(1, 4):
            ex = mbpp_raw[i]
            q, test, code = ex['text'], ex['test_list'], ex['code']
            ex_prompt = format_test_example(q, test, code)
            example_prompt = '- Example {}:\n{}'.format(i, ex_prompt)
            examples_str += [example_prompt]
        return '\n\n'.join(examples_str)
    def format_test_example(q, tests, code: str = None):
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(q.strip(), "\n".join(tests))
        if code:
            code = code.replace("\r", "").replace("\t", "    ")
            prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
        return prompt

    def give_full_prompt(examples_str, sample):
        prompt = format_test_example(q = sample['text'], tests = sample['test_list'], code = None)
        reasoning = "\n\n".join(sample['filled_reasonsing'])

        prompt_with_shots = f'''
        Please refer the given examples, follow strictly and implement the step by step reasoning to generate a python function for my problem.
        Examples are listed as follows:
        {examples_str}

        Here is my problem:
        {prompt}
        
        Step by step reasoning:
        {reasoning}
        '''.strip()
        return prompt_with_shots

    def format_single_sample(instance):
        task_id = instance[0]['task_id']
        task = mbpp_df.loc[mbpp_df['task_id'] ==task_id].iloc[0].to_dict()

        examples_str = give_example_str() # give examples

        format_samples = []
        for sample in instance:
            format_sample = sample
            format_sample['task_id'] = int(format_sample['task_id']) # need to change to integer
            format_sample['text'] = task['text']
            format_sample['test_list'] = task['test_list']
            format_sample['prompt'] = give_full_prompt(examples_str, format_sample)
            format_samples.append(format_sample)
        return format_samples

    mbpp_df = pd.DataFrame(mbpp_raw)
    mbpp_df['task_id'] = mbpp_df['task_id'].astype(str)

    format_data = []
    for instance in quality_data:
        if len(instance) > 0:
            format_data.append(format_single_sample(instance))
        else:
            continue

    return format_data

def convert_for_evaluation(example):
    gpt_completion = example['gpt_completion']
    generation = gpt_completion
    try:
        code_block: str = re.findall(f'```python\n(.*?)```', gpt_completion, re.DOTALL | re.IGNORECASE)[0]
        generation = code_block
    except Exception as ex:
        print("Failed to extract codeblock:\n{}".format(gpt_completion))

    example['generation'] = generation
    return example

def generate_one(example, tokenizer, model):
    prompt = example['prompt']
    inputs = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
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
    example['gpt_completion'] = output
    return convert_for_evaluation(example)

def unfold_format_data(format_data):
    rows = []
    task_count = {}

    for completions in format_data:
        for completion in completions:
            task_id = completion['task_id']
            if task_id not in task_count:
                task_count[task_id] = 0
            else:
                task_count[task_id] += 1

            rows.append({
                'task_id': task_id,
                'prompt_ls': "\n".join(completion['prompt_ls']),
                'text': completion['text'],
                'completion_id': task_count[task_id]
            })
    return pd.DataFrame(rows)


def normalize_score(label_list, coefficient=1.0):
    sum = 0
    coefficient = coefficient
    num_0 = label_list.count(0)  # count how many 0
    num_1 = label_list.count(1)
    sum += num_0 / len(label_list) * math.exp((1) * coefficient * 0)  # math.exp((1)*coefficient*0) this is just 1
    sum += num_1 / len(label_list) * math.exp((1) * coefficient * 1)
    log_sum = math.log(sum)

    return round(math.fabs((1) / coefficient * log_sum), 3) # this is per prompt (each prompt, not task_id)


def score_data_to_mcts(score_data, step_tag=' ки'):
    mcts = {}
    prev_id = -1
    prompts = []
    values = []
    start_step = 1

    for i in score_data:
        task_id = i['task_id']
        text = i['prompt_ls']
        value = i['label']

        if task_id != prev_id:  # new task
            if len(prompts) > 0:
                mcts[prev_id] = {
                    'text': (step_tag + '/n').join(prompts),
                    'value': values,
                }

            prompts = []
            start_step = 1
            filled_steps, start_step = extract_filled_steps(text, start_step)
            prompts.append(filled_steps)

            values = []
            values.append(value)
        else:
            filled_steps = extract_filled_steps(text)
            add_steps, start_step = extract_filled_steps(text, start_step)
            prompts.append(add_steps)
            values.append(value)

        prev_id = task_id

    # record the last one
    if len(prompts) > 0:
        print(prompts)
        mcts[prev_id] = {
            'text': (step_tag + '/n').join(prompts),
            'value': values,
        }
    return mcts


def extract_filled_steps(prompt, start_step=0):
    steps = prompt.split('\n')
    filled_steps = [step for step in steps if ': ...' not in step]
    filled_steps = [filled_step.replace('where ...', '').strip() for filled_step in filled_steps]
    end_step = len(filled_steps) + 1
    sliced_steps = filled_steps[start_step - 1:]

    return "".join(sliced_steps), end_step


if __name__ == "__main__":
    with open("combined_data.json", 'r') as f:
        data = json.load(f) # (5 * num of task_id) of instance with each instance has 16 completion

    # data = data[25:35] # for debug

    mbpp_path = Path(__file__).parent / "MBPP_eval" / "data" / 'mbpp.jsonl'
    mbpp_raw = [json.loads(x) for x in open(mbpp_path)]

    # process data
    quality_data = []
    for sample in tqdm(data, desc="Processing Samples"):
        quality_sample = get_quality_data(sample)
        if quality_sample is None:
            continue
        else:
            quality_data.append(quality_sample)

    # now i get bunch of completion, i need to evaluate the reasoning score
    # by create code base on the question, reasoning and test, then gen code and see the it pass the test
    format_data = format_test_for_quality_data(quality_data)

    ###################### this whole block cloud be comment out after the example are gen and saved ####################
    # set up model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    print("load tokenizer {} from {} over.".format(tokenizer.__class__, model_name_or_path))
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # ********************************************** generate **********************************************#
    # len(format_date) is how many instance = (5 (steps) * different task id)
    # len(instance) is valid completions ~ 10 - 16, one instance is of the same prompt (step filled) and task id

    generated_format_data = []
    for instance in tqdm(format_data, desc='Generating'):
        count = 0
        for sample in instance:
            gen_example = generate_one(sample, tokenizer, model)  # this is the one you need to gen
            generated_format_data.append(gen_example)
            count += 1
            print("Generate {}/{} over...".format(count, len(instance)))

    print("Generate all over!!!")
    with open(saved_path, 'w', encoding='utf-8') as fw:
        for ex in generated_format_data:
            fw.write(json.dumps(ex) + '\n')
        print("Save {} processed generated_format_data into {} over!".format(len(generated_format_data), saved_path))
    ##################### this whole block cloud be comment out after the example are gen and saved ####################

    result = evaluate_functional_correctness(
        input_file=saved_path,
        tmp_dir=temp_dir,
        problem_file=os.path.join(data_abs_dir, f"mbpp_test.jsonl"),
        language='python',
        is_mbpp=True,
        timeout=50.0
    )
    format_data_df = unfold_format_data(format_data)
    result_df = pd.merge(result, format_data_df, on = ['task_id', 'completion_id'], how = 'left')
    print(model_name_or_path)

    # after getting the result, need to cal the prob of each prompt
    print("-----------------")
    print("begin to label each process")
    print("-----------------")
    result_df['label'] = (result_df['passed'] * 1)
    score_df = result_df.groupby(['task_id', 'prompt_ls'])['label'].apply(lambda x: normalize_score(x.tolist())).reset_index()
    score_data = score_df.to_dict(orient='records')

    mcts = score_data_to_mcts(score_data, step_tag=' ки')

    mcts_autoregressive = [
        {
            "text": i['text'],
            "value": i['value']
        }
        for i in mcts.values()
    ]

    with open("autoregressive_deepseek_mbpp_data.json", 'w') as f:
        json.dump(mcts_autoregressive, f, indent=4, ensure_ascii=False)

    print("finish saving and the process")

