import time
import torch
import numpy as np

class BenchmarkRunner:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def get_vram_usage(self):
        """Returns current VRAM usage in MB."""
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / 1024**2
        return 0

    def get_max_vram_usage(self):
        """Returns peak VRAM usage in MB."""
        if self.device == "cuda":
            return torch.cuda.max_memory_allocated() / 1024**2
        return 0

    def run_inference(self, prompt, max_new_tokens=50):
        """Runs inference and measures latency and throughput."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]
        
        # Warmup
        _ = self.model.generate(**inputs, max_new_tokens=5)
        
        # Benchmark start
        torch.cuda.synchronize() if self.device == "cuda" else None
        start_time = time.time()
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True
            )
        
        torch.cuda.synchronize() if self.device == "cuda" else None
        end_time = time.time()
        
        total_time = end_time - start_time
        num_generated_tokens = output.shape[1] - input_len
        tokens_per_sec = num_generated_tokens / total_time
        latency_per_token = (total_time / num_generated_tokens) * 1000 # ms
        
        return {
            "total_time": total_time,
            "tokens_per_sec": tokens_per_sec,
            "latency_per_token_ms": latency_per_token,
            "num_generated_tokens": num_generated_tokens,
            "peak_vram_mb": self.get_max_vram_usage()
        }

    def run_benchmark(self, prompts, max_new_tokens=50):
        results = []
        for prompt in prompts:
            res = self.run_inference(prompt, max_new_tokens)
            results.append(res)
        
        # Aggregated stats
        avg_tokens_per_sec = np.mean([r["tokens_per_sec"] for r in results])
        avg_latency = np.mean([r["latency_per_token_ms"] for r in results])
        max_vram = np.max([r["peak_vram_mb"] for r in results])
        
        return {
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "avg_latency_per_token_ms": avg_latency,
            "max_vram_mb": max_vram,
            "raw_results": results
        }
