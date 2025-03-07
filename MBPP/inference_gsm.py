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

# Set environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ['VLLM_TARGET_DEVICE'] = 'cpu'

# Login to Hugging Face
login(token="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")

# Clear CUDA cache
th.cuda.empty_cache()

# Model name
# model_name = "gpt2"  # Change to "meta-llama/Llama-3.2-1B-Instruct" for Llama
model_name = "meta-llama/Llama-3.2-1B-Instruct"
save_model_name = 'Llama' if model_name == "meta-llama/Llama-3.2-1B-Instruct" else model_name
only_last = True
sft = False

# Function to print GPU memory
def print_gpu_memory():
    if th.cuda.is_available():
        device = th.cuda.current_device()
        total_memory = th.cuda.get_device_properties(device).total_memory
        allocated_memory = th.cuda.memory_allocated()
        reserved_memory = th.cuda.memory_reserved()
        free_memory = total_memory - allocated_memory

        print(f"Total Memory: {total_memory / (1024 ** 2):.2f} MB")
        print(f"Allocated Memory: {allocated_memory / (1024 ** 2):.2f} MB")
        print(f"Reserved Memory: {reserved_memory / (1024 ** 2):.2f} MB")
        print(f"Free Memory: {free_memory / (1024 ** 2):.2f} MB")
    else:
        print("CUDA is not available.")


# Print GPU memory at the start
print_gpu_memory()

def prompt_eng(prompt):
    return f"Question: {prompt}\n\nAnswer: "

def extract_ans(output, with_example = False):
    if with_example:
        output = output.split('Actual Problems:\n\n')[1]
    match = re.search(r'#### (\d+)', output)
    if match:
        extracted_number = match.group(1)  # Get the number part
        # print(extracted_number)
        return extracted_number
    else:
        print("Number not found.", output)
        return None

def few_shot_prompt(train_examples):
    ex_prompts = ['Examples: ']
    for ex in train_examples.select(range(3)):
        ex_prompts.append(f"Question: {ex['question']}\n\nAnswer: {ex['answer']}")
    return '\n\n'.join(ex_prompts)

def few_shot_prompt_eng(train_examples, sample_prompt):
    ex_prompts = few_shot_prompt(train_examples)
    return ex_prompts + '\n\nActual Problems:\n\n' + prompt_eng(sample_prompt)

def main():
    # Configure logging
    logging.disable(logging.CRITICAL)
    logging.basicConfig(filename='training_mar2.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.info(f"#******* Process Start *******#")
    logger.info("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("gsm8k", 'main')

    train_examples = dataset['train']
    test_examples = dataset['test']

    device = "cuda" if th.cuda.is_available() else "cpu"
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

    num_epochs = 10
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optim,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps,
    )

    # DataFrame to record training metrics
    training_metrics = pd.DataFrame(columns=["epoch", "step", "loss", "train_time"])

    if sft:
        logger.info("Starting training...")
        pbar = tqdm(range(num_training_steps))
        start_time = time.time()

        for epoch in range(num_epochs):
            for batch in train_loader:
                optim.zero_grad()
                combined_input = f"Question: {batch['question']}\n\nAnswer: {batch['answer']}<|endoftext|>"
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
        model_save_path = f"./models/mar2_{save_model_name}/"
        model.save_pretrained(model_save_path)
        logger.info(f"Model saved to {model_save_path}")

    # Save training metrics to CSV
    training_metrics.to_csv(f"training_metrics_{save_model_name}.csv", index=False)

    # Load the trained model for inference with Hugging Face
    model = AutoModelForCausalLM.from_pretrained(model_save_path).to(device)
    model.eval()
    inference_metrics_hf = pd.DataFrame(columns=["input", "generated_output", "actual_output"])


    start_time = time.time()
    for example in tqdm(test_examples.select(range(30))):
        input_text = example['question']
        # input_text = prompt_eng(input_text)  # f"Question: {prompt}\n\nAnswer: " regulate the behavior
        input_text = few_shot_prompt_eng(train_examples, input_text) # few show
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        with th.no_grad():
            generated_ids = model.generate(input_ids, max_length=1024)

        generated_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        actual_output = example['answer']

        #extract answer
        generated_numeric_ans = extract_ans(generated_output, with_example = True)
        actual_numeric_ans = extract_ans(actual_output)

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
    th.cuda.empty_cache()
    gc.collect()
    del model
    th.cuda.empty_cache()

    # Load the trained model for inference with VLLM
    logger.info("Starting inference with VLLM...")
    stop_tokens = ["<|EOT|>"]
    sampling_params = SamplingParams(n=1, temperature=1, top_p=1, max_tokens=1024, stop=stop_tokens)
    llm = LLM(model=model_save_path, tensor_parallel_size=1,
              enforce_eager=True, dtype="float16", gpu_memory_utilization=0.9, swap_space=64)
    inference_metrics_vllm = pd.DataFrame(columns=["input", "generated_output", "actual_output"])

    # Loop through test examples to collect outputs
    start_time = time.time()
    for example in tqdm(test_examples.select(range(30))):
        input_text = example['question']
        input_text = prompt_eng(input_text)  # f"Question: {prompt}\n\nAnswer: " regulate the behavior
        generated_output = llm.generate(input_text, sampling_params)
        actual_output = example['answer']

        #extract answer
        generated_numeric_ans = extract_ans(generated_output)
        actual_numeric_ans = extract_ans(actual_output)

        inference_metrics_vllm = inference_metrics_vllm._append({
            "input": input_text,
            "generated_output": generated_output,
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