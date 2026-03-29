# Project Architecture (v1.1.1)

This project is a benchmarking suite for comparing quantization methods in Large Language Models (LLMs).

## Core Components
- **`main.py`**: The entry point for the benchmarking workflow. It handles the iteration through different model precisions, triggers the performance measurements, and compiles the final summary.
- **`models/loader.py`**: A specialized model loader with support for `fp16`, `int8`, and `int4` (NormalFloat4/Nested Quantization) via Hugging Face `transformers` and `bitsandbytes`.
- **`benchmark/runner.py`**: The performance profiler. It measures:
    - **Throughput**: Calculated as tokens per second over multiple inference runs.
    - **Latency**: Measures average response time in milliseconds.
    - **Peak VRAM**: Tracked via `torch.cuda.memory_allocated()`, specifically for NVIDIA GPUs.
- **`eval/accuracy.py`**: Uses a fixed set of evaluation texts to calculate **Perplexity**. This measure quantifies the model's confidence in its language predictions, allowing users to estimate the "accuracy penalty" of quantization.
- **`results/`**: Used for storing persistent results if enabled.

## Technology Stack
- **`transformers`**: Core model and tokenizer handling.
- **`bitsandbytes`**: Backend for `int8` and `int4` quantization.
- **`torch`**: Deep learning framework, memory measurement, and device management.
- **`time` / `gc`**: Performance profiling and memory management.

## Performance Profile
The benchmarking workflow includes a **warmup run** for each precision to ensure CUDA kernel compilation and graph optimization don't skew the results. Memory is explicitly cleared between benchmarks (`del model`, `torch.cuda.empty_cache()`, `gc.collect()`) to allow for fair comparisons on limited hardware.
