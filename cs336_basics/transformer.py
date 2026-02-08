import torch.nn as nn  
import torch 
import math
from typing import Optional
from jaxtyping import Bool, Float, Int
from collections.abc import Callable, Iterable
from torch import Tensor

class Linear(nn.Module):
    def __init__ (self, 
                  in_features: int, 
                  out_features: int, 
                  device: torch.device | None=None, 
                  dtype:torch.dtype | None=None):
        
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        sigma = math.sqrt(2.0/(in_features + out_features))
        nn.init.trunc_normal_(self.weight, 0, sigma, a= -3 * sigma, b= 3 * sigma)
    
    def forward(self, x: torch.Tensor):
        return x  @ self.weight.T
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, 0, 1, a=-3, b=3)
    
    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids, :]
    
class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # GPT version
        # rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        # rms_x = x * self.gain / (rms + self.eps)
        x_sum = (x ** 2).sum(dim=-1, keepdim=True)
        rms = torch.sqrt(x_sum / self.d_model + self.eps)
        rms_x = x * self.weight / rms 
        
        return rms_x.to(in_dtype)


class SwiGLU_FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device:torch.device | None = None, dtype: torch.dtype | None = None ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff,   device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model,  device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x):

        x_w1 = self.w1(x)
        SiLU_X = x_w1 * torch.sigmoid(x_w1)
        X = self.w3(x)
        return self.w2(SiLU_X * X)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, 
                 device:torch.device | None = None,
                 dtype:torch.dtype | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.seq_len = max_seq_len

        assert self.d_k % 2 == 0
        self.half_k = self.d_k // 2
        k = torch.arange(start=1, end=self.half_k + 1, device=device, dtype=dtype)
        inv_freq = 1.0 / (self.theta ** ((2 * k - 2) / self.d_k))

        pos = torch.arange(self.seq_len, device=device)

        angles = pos[:, None] * inv_freq[None, :]

        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor])-> torch.Tensor:
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        token_positions = token_positions.to(x.device, dtype=torch.long)
        
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rot_even = x_even * cos - sin * x_odd
        x_rot_odd  = x_even * sin + cos * x_odd

        x_rot = torch.empty_like(x)
        x_rot[..., 0::2] = x_rot_even
        x_rot[..., 1::2] = x_rot_odd

        return x_rot

def SoftMax(x:torch.Tensor, dim: int = -1):
    x_max, _ = x.max(dim=dim, keepdim=True)
    x_scale = x - x_max
    x_exp = x_scale.exp()

    log_sum = torch.log(x_exp.sum(dim=dim, keepdim=True))
    log_p = x_scale - log_sum
    
    x_probi = x_exp / x_exp.sum(dim=dim, keepdim=True)
    return x_probi

def scaled_dot_product_attention(Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask: Optional[torch.Tensor] = None, is_causal=True):
    score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
    S = Q.shape[-2]
    eff_mask = mask
    if is_causal:
        if eff_mask is not None:
            raise ValueError("attn mask and is_causal cannot both be set")
        eff_mask = torch.tril(torch.ones((S, S), dtype=torch.bool, device=score.device))

    if eff_mask is not None:
        score.masked_fill_(~eff_mask, float("-inf"))
    
    probi = SoftMax(score, dim=-1)
    out = probi @ V
    return out

def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
):

    Q = in_features @ q_proj_weight.T
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T

    head_dim = d_model // num_heads

    Q = Q.unflatten(-1, (num_heads, head_dim)).transpose(-3, -2)
    K = K.unflatten(-1, (num_heads, head_dim)).transpose(-3, -2)
    V = V.unflatten(-1, (num_heads, head_dim)).transpose(-3, -2)

    out = scaled_dot_product_attention(Q, K, V)
    out = out.transpose(-3, -2).flatten(-2, -1)

    return out @ o_proj_weight.T

def multihead_self_attention_rope(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    max_seq_len: int,
    theta: float,
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
):

    Q = in_features @ q_proj_weight.T
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T

    head_dim = d_model // num_heads

    Q = Q.unflatten(-1, (num_heads, head_dim)).transpose(-3, -2)
    K = K.unflatten(-1, (num_heads, head_dim)).transpose(-3, -2)
    V = V.unflatten(-1, (num_heads, head_dim)).transpose(-3, -2)


    seq_len = Q.shape[-2]
    rope = RotaryPositionalEmbedding(theta, head_dim, max_seq_len, Q.device)
    Q_r = rope(Q, token_positions)
    K_r = rope(K, token_positions)
    out = scaled_dot_product_attention(Q_r, K_r, V)

    out = out.transpose(-3, -2).flatten(-2, -1)

    return out @ o_proj_weight.T

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, max_seq_length: int,theta: float = 10000,
                 attn_mask:torch.Tensor | None = None, is_causal: bool =True,
                 device:torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.is_causal = is_causal
        self.attn_mask = attn_mask
        self.device = device
        self.max_seq_len = max_seq_length

        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)
        self.RoPE = RotaryPositionalEmbedding(theta, self.d_k, max_seq_length, device, dtype)

    def forward(self, x: torch.Tensor, token_positions: Int[Tensor, " ... sequence_length"] | None = None):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.unflatten(-1, (self.num_heads, self.d_k)).transpose(-2, -3)
        K = K.unflatten(-1, (self.num_heads, self.d_k)).transpose(-2, -3)
        V = V.unflatten(-1, (self.num_heads, self.d_k)).transpose(-2, -3)

        Q_r = self.RoPE(Q, token_positions)
        K_r = self.RoPE(K, token_positions)

        score = torch.matmul(Q_r, K_r.transpose(-2, -1)) / math.sqrt(Q_r.shape[-1])
        seq_len = Q_r.shape[-2]
        eff_mask = self.attn_mask
        if self.is_causal:
            if eff_mask is not None:
                raise ValueError("attn mask and is_causal cannot both be set")
            eff_mask = torch.tril(torch.ones(seq_len, seq_len, dtype = torch.bool, device=self.device))
        
        if eff_mask is not None:
            score.masked_fill_(~eff_mask, float("-inf"))
        probi = SoftMax(score, dim=-1)
        out = probi @ V

        out = out.transpose(-3, -2).flatten(-2, -1)
        context_x = self.output_proj(out)
            
        return context_x
        
class Transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype

        self.ln1 = RMSNorm(self.d_model, device=self.device, dtype=self.dtype)
        self.ln2 = RMSNorm(self.d_model, device=self.device, dtype=self.dtype)
        self.attn = MultiHeadAttention(d_model=self.d_model, 
                                                     d_ff=self.d_ff, 
                                                     num_heads=self.num_heads,
                                                     max_seq_length=max_seq_len,
                                                     theta=theta,
                                                     device=self.device, dtype=self.dtype)
        self.ffn = SwiGLU_FFN(self.d_model, self.d_ff, device=self.device, dtype=self.dtype)

    def forward(self, x: Float[Tensor, "batch seq_len d_model"]):
        x_norm = self.ln1(x)
        h = x + self.attn(x_norm)
        h_norm = self.ln2(h)
        out = h + self.ffn(h_norm)

        return out
    
class Transformer(nn.Module):
    def __init__(self, vocab_size: int,
                context_length: int,
                d_model: int,
                num_layers: int,
                num_heads: int,
                d_ff: int,
                rope_theta: float,
                device: torch.device | None = None, dtype: torch.dtype | None = None
                 ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                Transformer_block(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x:Float[Tensor, "batch seq_len"]):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x_norm = self.ln_final(x)
        logit = self.lm_head(x_norm)

        return logit

def LogSoftMax(x:torch.Tensor, dim: int = -1)-> Float[Tensor, "batch seq_length vocab_size"]:
    x_max, _ = x.max(dim=dim, keepdim=True)
    x_scale = x - x_max
    x_exp = x_scale.exp()

    log_sum = torch.log(x_exp.sum(dim=dim, keepdim=True))
    log_p = x_scale - log_sum

    return log_p

def CrossEntropyLoss(logits: Float[Tensor, "batch seq_length vocab_size"], labels: Int[Tensor, "batch seq_length"],
                     ignore_index:int = -100):
    
    log_probi = LogSoftMax(logits)
    log_probi = log_probi.view(-1, log_probi.shape[-1])
    labels = labels.view(-1)

    valid_mask = labels != ignore_index
    valid_probi = log_probi[valid_mask]
    valid_labels = labels[valid_mask]
    loss = valid_probi[torch.arange(valid_probi.shape[0]), valid_labels]
    
    return -loss.mean()

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr= 1e-3, betas:tuple[float, float]=(0.9, 0.99), weight_decay = 0.1, eps = 1e-6):
        if lr < 0 :
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr":lr, "betas": betas, "weight_decay" : weight_decay, "eps":eps}
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():  
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                g = p.grad
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["t"] += 1
                t = state["t"]
                m = state["m"]
                v = state["v"]

                m.mul_(beta1).add_(g, alpha = 1- beta1)
                v.mul_(beta2).addcmul_(g, g, value= 1-beta2)

                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                if weight_decay !=0 :
                    p.mul_(1-lr * weight_decay)
                p.addcdiv_(m ,v.sqrt().add_(eps), value=-lr_t)

        return loss

def cosine_schedule_with_warmup(t, alpha_min, alpha_max,T_w, T_c):
    if t < T_w:
        alpha_t = alpha_max * t / T_w 
    elif t >= T_w and t <=T_c:
        alpha_t = alpha_min + 1/2 *(1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))) * (alpha_max - alpha_min)
    else:
        alpha_t = alpha_min
    
    return alpha_t

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    total = 0
    params = list(parameters)
    for p in params:
        if p.grad is None:
            continue
        g = p.grad
        total += g.pow(2).sum()

    total_g = total.sqrt_().item()

    if total_g > max_l2_norm:
        scale = max_l2_norm / (total_g + eps)
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)
                
    return total_g







