from tokenizers import Tokenizer
from transformer import Transformer
import torch
import yaml
from easydict import EasyDict

def set_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def sample(logits:torch.Tensor, p = 0.9, temperature = 1.0, top_k: int | None = 200):
    """
    logits: (B, V)  —— 只传最后一步 y_hat[:, -1, :]
    return: (B,) next token ids
    """
    logits = logits / temperature
    logits = logits - logits.max(dim=-1, keepdim=True).values

    logits_sum = logits.exp().sum(dim=-1, keepdim=True)
    thereshold = p * logits_sum
    
    if top_k is None:
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    else:
        sorted_logits, sorted_indices = torch.topk(logits, k=top_k)
    
    sorted_logits.exp_()
    cdf = torch.cumsum(sorted_logits, dim=-1)

    keep = cdf <= thereshold
    keep[..., 0] = True

    w = sorted_logits * keep
    probs = w / w.sum(dim=-1, keepdim=True)
    pos = torch.multinomial(probs, 1)
    next_ids = sorted_indices.gather(dim=-1, index=pos)

    return next_ids.squeeze(-1)

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        cfg = EasyDict(cfg)

    device = set_device()
    print(f"Using device: {device}")

    model = Transformer(cfg.vocab_size, 
                    cfg.context_length,
                    cfg.d_model,
                    cfg.num_layers,
                    cfg.num_heads,
                    cfg.d_ff,
                    cfg.rope_theta,
                )
    model.to(device)

    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    text = input("Enter a prompt: ")
    tokenizer = Tokenizer.from_files(cfg.vocab_file, cfg.merges_file, special_tokens=cfg.special_tokens)
    tokens = torch.tensor(tokenizer.encode(text))
    tokens = tokens.unsqueeze(0).to(device)

    eos_tk = "<|endoftext|>"
    eot_id = tokenizer.vocab_inv[eos_tk.encode("utf-8")]

    with torch.no_grad():
        for _ in range(cfg.inference_length):
            inputs = tokens[:, -cfg.context_length:]

            y_hat = model(inputs)
            next_token = sample(y_hat[:, -1, :], 
                                p=cfg.p, 
                                temperature=cfg.temperature, 
                                top_k=cfg.top_k)

            if eot_id == next_token.item():
                break

            tokens = torch.cat([tokens, next_token.unsqueeze(-1)], dim=1) 

    generated_text = tokenizer.decode(tokens.squeeze().tolist())
    print("Generated text:")
    print(generated_text)   

if __name__ == "__main__":
    main()