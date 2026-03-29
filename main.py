import argparse
import json
import os
from models.loader import load_model
from benchmark.runner import BenchmarkRunner
from eval.accuracy import calculate_perplexity

def main():
    parser = argparse.ArgumentParser(description="Model Quantization & Optimization Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model ID")
    parser.add_argument("--quant", type=str, choices=[None, "int8", "int4", "gptq", "awq"], default=None, help="Quantization method")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda, mps, cpu)")
    parser.add_argument("--perplexity", action="store_true", help="Calculate perplexity")
    parser.add_argument("--max_tokens", type=int, default=50, help="Max tokens to generate in benchmark")
    parser.add_argument("--output", type=str, default="results/benchmark_results.json", help="Path to save results")

    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model(args.model, args.quant, args.device)

    # Benchmark Performance
    runner = BenchmarkRunner(model, tokenizer, args.device)
    prompts = [
        "Explain the importance of model quantization in LLMs.",
        "What are the benefits of 4-bit quantization over FP16?",
        "Write a short story about an AI that wants to be more efficient."
    ]
    
    print("\nRunning Performance Benchmark...")
    bench_results = runner.run_benchmark(prompts, args.max_tokens)
    
    # Optional Accuracy Benchmark
    ppl = None
    if args.perplexity:
        print("\nCalculating Perplexity...")
        # Use a small sample text for quick eval
        sample_text = "Quantization and pruning are core techniques in model optimization. " * 10
        ppl = calculate_perplexity(model, tokenizer, sample_text, args.device)
        bench_results["perplexity"] = ppl

    # Display results
    print("\n" + "="*50)
    print(f"RESULTS for {args.model} ({args.quant})")
    print(f"Avg Tokens/Sec: {bench_results['avg_tokens_per_sec']:.2f}")
    print(f"Avg Latency/Token: {bench_results['avg_latency_per_token_ms']:.2f} ms")
    print(f"Max VRAM Usage: {bench_results['max_vram_mb']:.2f} MB")
    if ppl:
        print(f"Perplexity: {ppl:.2f}")
    print("="*50 + "\n")

    # Save to file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(bench_results, f, indent=4)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
