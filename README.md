# Model Quantization & Optimization Benchmark (v1.1.0)

A comprehensive tool for benchmarking Large Language Models (LLMs) across different quantization methods, focusing on performance, memory usage, and accuracy.

## Features
- **Precision Comparison**: Benchmark models in `fp16`, `int8`, and `int4` (NormalFloat4).
- **Inference Performance**: Captures Average Latency (ms) and Throughput (tokens/sec).
- **VRAM Tracking**: Measures peak GPU memory usage in MB.
- **Accuracy Evaluation**: Calculates Perplexity on a selected set of texts to measure the "accuracy tax" of quantization.
- **Automated Workflow**: Iterates through multiple precisions and summarizes results in a single run.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Simply run the main script to benchmark the default model (`meta-llama/Meta-Llama-3-8B`):
```bash
python main.py
```

The script will automatically:
1. Load the model in `fp16`, `int8`, and `int4`.
2. Measure inference speed and memory.
3. Calculate perplexity for each version.
4. Output a summary comparison.

## Results
Benchmark results are printed to the console as a JSON summary at the end of the run.
