from models.loader import load_model
from benchmark.runner import benchmark_inference
from eval.accuracy import compute_perplexity
import json

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
PRECISIONS = ["fp16", "int8", "int4"]
PROMPT = "Explain the theory of relativity in simple terms:"

results = {}

for precision in PRECISIONS:
    print(f"\n{'='*40}")
    print(f"Running: {precision}")
    print(f"{'='*40}")

    model, tokenizer = load_model(MODEL_NAME, precision)

    bench = benchmark_inference(model, tokenizer, PROMPT)
    ppl   = compute_perplexity(model, tokenizer)

    results[precision] = {**bench, "perplexity": ppl}

    print(f"Results: {json.dumps(results[precision], indent=2)}")

    # free memory before loading next precision
    del model
    import torch, gc
    torch.cuda.empty_cache()
    gc.collect()

print("\n\nFINAL RESULTS:")
print(json.dumps(results, indent=2))
