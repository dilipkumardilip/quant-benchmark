import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPTQConfig, AWQConfig

def load_model(model_id, quantization=None, device="cuda"):
    """
    Load a model with specified quantization.
    
    Args:
        model_id (str): Hugging Face model ID.
        quantization (str): One of [None, 'int8', 'int4', 'gptq', 'awq'].
        device (str): Device to load the model on ('cuda', 'mps', 'cpu').
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    quant_config = None
    if quantization == "int8":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "int4":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "gptq":
        # Assumes model is already GPTQ quantized or we use GPTQConfig for loading
        quant_config = GPTQConfig(bits=4, disable_exllama=False)
    elif quantization == "awq":
        # Assumes model is already AWQ quantized or we use AWQConfig
        quant_config = AWQConfig(bits=4)

    print(f"Loading model {model_id} with quantization={quantization} on {device}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto" if device == "cuda" else { "": device },
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        trust_remote_code=True
    )
    
    return model, tokenizer
