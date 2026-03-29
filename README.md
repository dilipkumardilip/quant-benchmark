# Model Quantization & Optimization Benchmark (v1.0.0)

A comprehensive tool for benchmarking Large Language Models (LLMs) across different quantization methods, focusing on performance, memory usage, and accuracy.

## Features
- **Multi-Quantization Support**: BitsAndBytes (Int8/Int4), GPTQ, and AWQ.
- **Performance Metrics**: Latency (ms/token), Throughput (tok/sec), and Peak VRAM.
- **Accuracy Evaluation**: Perplexity calculation on sample datasets.
- **Flexible CLI**: easy-to-use command line interface for running benchmarks.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
### Run basic benchmark
```bash
python main.py --model llama-3-8b --quant int4 --device cuda
```

### Run with perplexity evaluation
```bash
python main.py --model Qwen/Qwen2-7B-Instruct --quant gptq --perplexity
```

## Results
Benchmark results are saved to `results/benchmark_results.json` by default.
