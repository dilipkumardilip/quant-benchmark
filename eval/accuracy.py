# This measures perplexity — the standard way to check if quantization hurt the model's language understanding.

import torch
from torch.nn import CrossEntropyLoss

# A small fixed set of sentences to evaluate on
# In a real benchmark you'd use WikiText-2 dataset
# For our project this is enough to show relative accuracy
EVAL_TEXTS = [
    "The capital of France is Paris and it is known for the Eiffel Tower.",
    "Machine learning models are trained using gradient descent optimization.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Quantum computing uses qubits which can exist in superposition states.",
    "The human brain contains approximately 86 billion neurons connected by synapses.",
]


def compute_perplexity(model, tokenizer, texts: list = None):
    """
    Perplexity = how surprised the model is by the text.
    Lower = better. fp16 baseline will have lowest perplexity.
    int4 will be slightly higher — that's the accuracy cost.
    """
    if texts is None:
        texts = EVAL_TEXTS

    device = next(model.parameters()).device
    loss_fn = CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            input_ids = inputs["input_ids"]

            outputs = model(**inputs, labels=input_ids)

            # loss is already averaged over tokens by HuggingFace
            # we weight it by number of tokens for a proper average
            n_tokens = input_ids.shape[1]
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return round(perplexity, 4)