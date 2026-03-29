# how fast is the model, and how much memory does it use?

import torch
import time


def measure_memory_mb():
    """Returns current GPU memory used in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0  # on Mac/CPU, we skip GPU memory tracking


def benchmark_inference(model, tokenizer, prompt: str, num_runs: int = 5):
    """
    Runs inference num_runs times and returns:
    - avg_tokens_per_sec
    - avg_latency_ms
    - memory_mb (peak GPU memory)
    """

    # tokenize the input once
    inputs = tokenizer(prompt, return_tensors="pt")

    # move inputs to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    # warmup run — first run is always slower due to CUDA kernel compilation
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False  # greedy decoding — deterministic, fair comparison
        )

    # reset memory stats before actual measurement
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    latencies = []

    for _ in range(num_runs):
        start = time.perf_counter()

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )

        end = time.perf_counter()
        latencies.append(end - start)

    # how many NEW tokens were generated
    output_len = output.shape[1] - input_len

    avg_latency = sum(latencies) / len(latencies)
    avg_tokens_per_sec = output_len / avg_latency
    peak_memory = measure_memory_mb()

    return {
        "avg_latency_ms": round(avg_latency * 1000, 2),
        "tokens_per_sec": round(avg_tokens_per_sec, 2),
        "peak_memory_mb": round(peak_memory, 2),
        "num_runs": num_runs
    }