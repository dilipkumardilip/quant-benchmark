import torch
from tqdm import tqdm

def calculate_perplexity(model, tokenizer, dataset_text, device="cuda"):
    """
    Calculate perplexity on a given text string or list of strings.
    
    Args:
        model: Loaded LLM.
        tokenizer: Loaded tokenizer.
        dataset_text: String or list of strings to evaluate.
        device: Device to run evaluation on.
    """
    if isinstance(dataset_text, list):
        dataset_text = " ".join(dataset_text)
    
    encodings = tokenizer(dataset_text, return_tensors="pt")
    
    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # how many tokens we want to predict
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # NLL = loss * trg_len
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()
