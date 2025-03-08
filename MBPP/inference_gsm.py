import os
import time
import pandas as pd
import torch as th
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from datasets import load_dataset
import logging
from huggingface_hub import login
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams  # Assuming vllm is properly installed
from tqdm import tqdm
import re
import gc
from utils import check_gpu_memory

# Set environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ['VLLM_TARGET_DEVICE'] = 'cpu'

# Login to Hugging Face
login(token="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")

# Clear CUDA cache
th.cuda.empty_cache()

# Model name
# model_name = "gpt2"
model_name = "meta-llama/Llama-3.2-1B-Instruct"
save_model_name = 'Llama' if model_name == "meta-llama/Llama-3.2-1B-Instruct" else model_name
only_last = True
sft = False # train_loss: 2.09387 to 0.63933 after 5 epoch (gsm8k)，using the HZ data can make it to 0.3 for 1 epoch
HF = False
len_of_example = 100 # max 1319
model_save_path = f"./models/mar2_{save_model_name}/"
data_set = "HanningZhang/deepseek-gsm-new"

# Print GPU memory at the start
check_gpu_memory()

def prompt_eng(prompt):
    return f"Question: {prompt}\n\nAnswer: "

# match = re.search(r'#### (\d+)', output)


def extract_answer(text,with_example = False):
    # Split the text to isolate the section after "Actual Problem:"
    if with_example:
        text= text.split("Actual Problem:")[-1]

    # Define a comprehensive regex pattern to match various answer format
    actual_problem_part = text.split("The answer is")[-1].split('.')[0]

    pattern = r'''
        (?:                           # Non-capturing group for the entire answer format
            \s*                      # Matches optional whitespace
            (\d+)                   # Captures the digits
            \s*                     # Matches optional whitespace
        )
    '''
    # Search for the pattern in the actual problem part
    match = re.search(pattern, actual_problem_part, re.VERBOSE)

    if match:
        # Extract the captured number
        extracted_number = match.group(1)
        return extracted_number
    else:
        print("Number not found in the provided text.")
        return None


# def extract_ans(text, with_example = False):
#     # Split the text to isolate the section after "Actual Problem:"
#     if with_example:
#         actual_problem_part = text.split("Actual Problem:")[-1]
#
#     # Look for the answer in various formats
#     # This captures "The answer is: $number$", "The answer is $number$", "The answer is number", "$number$", "number"
#     pattern = r'(?:The answer is:? ?|\$?\\boxed{?(\d+)}?|\$?(\d+))'
#     match = re.search(pattern, actual_problem_part)
#
#     if match:
#         extracted_number = match.group(1)  # Get the number part
#         return extracted_number
#     else:
#         print("Number not found in the provided text.")
#         return None

def few_shot_prompt(train_examples):
    ex_prompts = ["Please refer to the format of the Examples, then let's think step by step to provide answer for the Actual Problem.\n\n Examples: "]
    for ex in train_examples.select(range(2,4)):
        ex_prompts.append(f"Question: {ex['question']}\n\nAnswer: {ex['answer']}")
    return '\n\n'.join(ex_prompts)

def few_shot_prompt_eng(train_examples, sample_prompt):
    ex_prompts = few_shot_prompt(train_examples)
    return ex_prompts + '\n\nActual Problem:\n\n' + prompt_eng(sample_prompt)

def main():
    # Configure logging
    # logging.disable(logging.CRITICAL)
    logging.basicConfig(filename='training_mar2.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.info(f"#******* Process Start *******#")
    logger.info("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # dataset = load_dataset("gsm8k", 'main')
    dataset = load_dataset(data_set)
    train_test_split = dataset['train'].train_test_split(test_size=0.2)

    train_examples =  train_test_split['train']
    test_examples = train_test_split['test']

    train_examples = train_examples.rename_column('prompt', 'question')
    test_examples = test_examples.rename_column('prompt', 'question')

    device = "cuda" if th.cuda.is_available() else "cpu"
    if sft:
        config = AutoConfig.from_pretrained(model_name)
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Freeze all layers
        if only_last:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.lm_head.parameters():
                param.requires_grad = True

        model.to(device)
        model.train()

        train_loader = DataLoader(train_examples, batch_size=8, shuffle=True)
        initial_lr = 5e-4
        optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_lr)

        num_epochs = 1
        num_training_steps = num_epochs * len(train_loader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optim,
            num_warmup_steps=int(num_training_steps * 0.1),
            num_training_steps=num_training_steps,
        )

        # DataFrame to record training metrics
        training_metrics = pd.DataFrame(columns=["epoch", "step", "loss", "train_time"])


        logger.info("Starting training...")
        pbar = tqdm(range(num_training_steps))
        start_time = time.time()

        for epoch in range(num_epochs):
            for batch in train_loader:
                optim.zero_grad()
                combined_input = f"Question: {batch['question']}\n\nAnswer: {batch['answer']}"
                input = tokenizer(combined_input, max_length=512, padding=True, truncation=True, return_tensors="pt").to(
                    device)

                outputs = model(**input, labels=input['input_ids'])
                loss = outputs.loss
                loss.backward()
                optim.step()
                lr_scheduler.step()
                pbar.update(1)
                pbar.set_description(f"train_loss: {loss.item():.5f}")

                # Log the training loss and time every few steps
                if pbar.n % 10 == 0:
                    elapsed_time = time.time() - start_time
                    training_metrics = training_metrics._append({
                        "epoch": epoch + 1,
                        "step": pbar.n,
                        "loss": loss.item(),
                        "train_time": elapsed_time
                    }, ignore_index=True)
                    logger.info(f"Epoch: {epoch + 1}, Step: {pbar.n}, Loss: {loss.item():.5f}")

            logger.info(f"Completed Epoch: {epoch + 1} with Loss: {loss.item():.5f}")
            th.cuda.empty_cache()

        total_training_time = time.time() - start_time
        logger.info(f"Total {save_model_name}_Training Time: {total_training_time:.2f} seconds")

        # Save the model
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        logger.info(f"Model saved to {model_save_path}")

        # Save training metrics to CSV
        training_metrics.to_csv(f"training_metrics_{save_model_name}.csv", index=False)

    if HF:
        # Load the trained model for inference with Hugging Face
        model = AutoModelForCausalLM.from_pretrained(model_save_path).to(device)
        model.eval()
        inference_metrics_hf = pd.DataFrame(columns=["input", "generated_output", "actual_output"])


        start_time = time.time()
        for example in tqdm(test_examples.select(range(len_of_example))):
            input_text = example['question']
            # input_text = prompt_eng(input_text)  # f"Question: {prompt}\n\nAnswer: " regulate the behavior
            input_text = few_shot_prompt_eng(train_examples, input_text) # few shot
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

            with th.no_grad():
                generated_ids = model.generate(input_ids, max_length=1024)

            generated_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            actual_output = example['answer']

            #extract answer
            generated_numeric_ans = extract_answer(generated_output, with_example = True)
            actual_numeric_ans = extract_answer(actual_output)

            inference_metrics_hf = inference_metrics_hf._append({
                "input": input_text,
                "generated_output": generated_output,
                "actual_output": actual_output,
                "generated_numeric_ans": generated_numeric_ans,
                "actual_numeric_ans": actual_numeric_ans,
            }, ignore_index=True)

        inference_metrics_hf.to_csv(f"inference_metrics_hf_{save_model_name}.csv", index=False)
        total_hf_inference_time = time.time() - start_time
        logger.info(f"Total {save_model_name}_HF inference Time: {total_hf_inference_time:.2f} seconds")

    # Clear GPU memory before calling LLM
    # th.cuda.empty_cache()
    # gc.collect()
    # del model
    # th.cuda.empty_cache()

    # Load the trained model for inference with VLLM
    else:
        logger.info("Starting inference with VLLM...")
        stop_tokens = []
        sampling_params = SamplingParams(n=1, temperature=0.7, top_p=0.9, max_tokens=1024, stop=stop_tokens)
        llm_model = model_save_path if not sft else model_name
        llm = LLM(model=llm_model, tensor_parallel_size=1,
                  enforce_eager=True, dtype="float16", gpu_memory_utilization=0.9, swap_space=64)
        inference_metrics_vllm = pd.DataFrame(columns=["input", "generated_output", "actual_output"])

        # Loop through test examples to collect outputs
        start_time = time.time()
        for example in tqdm(test_examples.select(range(len_of_example))):
            input_text = example['question']  # f"Question: {prompt}\n\nAnswer: " regulate the behavior
            input_text = few_shot_prompt_eng(train_examples, input_text) # few show
            generated_output = llm.generate(input_text, sampling_params)
            actual_output = example['answer']

            #extract answer
            generated_numeric_ans = extract_answer(generated_output[0].outputs[0].text)
            actual_numeric_ans = extract_answer(actual_output)

            inference_metrics_vllm = inference_metrics_vllm._append({
                "input": input_text,
                "generated_output": generated_output[0].outputs[0].text,
                "actual_output": actual_output,
                "generated_numeric_ans": generated_numeric_ans,
                "actual_numeric_ans": actual_numeric_ans,
            }, ignore_index=True)

        inference_metrics_vllm.to_csv(f"inference_metrics_vllm_{save_model_name}.csv", index=False)
        total_vllm_inference_time = time.time() - start_time
        logger.info(f"Total {save_model_name}_vLLM inference Time: {total_vllm_inference_time:.2f} seconds")


    logger.info(f"#******* Process Finished *******#")


if __name__ == "__main__":
    main()