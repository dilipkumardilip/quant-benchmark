# Project Architecture (v1.0.0)

This project is structured as a modular benchmarking suite.

## Components
- **`main.py`**: CLI entry point.
- **`models/loader.py`**: Model loading functionality with support for multiple quantization types.
- **`benchmark/runner.py`**: Inference performance measurements (Latency, Throughput, VRAM).
- **`eval/accuracy.py`**: Model accuracy calculation via perplexity.
- **`results/`**: Directory for benchmark output results.

## Performance Metrics Calculation
- **Latency**: Measured in milliseconds per generated token. Includes warmup.
- **Throughput**: Calculated as tokens generated per second.
- **VRAM**: Tracked using `torch.cuda.max_memory_allocated()`.
- **Perplexity**: Inverse probability of the evaluation text sequence under the model's distribution.
