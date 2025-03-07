import torch


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
