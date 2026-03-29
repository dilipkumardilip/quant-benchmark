# This is the heart of the project. It loads the same model in three different precisions: fp16, int8, int4.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # Apple Silicon — CPU fallback for quant
    else:
        return "cpu"


def load_model(model_name: str, precision: str):
    """
    Load a HuggingFace model in fp16, int8, or int4.

    Args:
        model_name: e.g. "facebook/opt-125m" or "meta-llama/Meta-Llama-3-8B"
        precision:  "fp16", "int8", or "int4"

    Returns:
        model, tokenizer
    """
    print(f"Loading {model_name} in {precision}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if precision == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    elif precision == "int8":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )

    elif precision == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # compute in fp16 for speed
            bnb_4bit_use_double_quant=True,  # nested quantization — saves more memory
            bnb_4bit_quant_type="nf4"  # NormalFloat4 — better than plain int4
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )

    else:
        raise ValueError(f"Unknown precision: {precision}. Choose fp16, int8, or int4.")

    model.eval()  # disable dropout etc — we're only doing inference
    return model, tokenizer