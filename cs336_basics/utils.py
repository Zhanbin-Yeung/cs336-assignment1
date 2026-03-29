import typing
from typing import Callable, Optional, Iterable
import os
import math
import torch
from torch.optim.lr_scheduler import LRScheduler

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
                    model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None):
    ckpt = torch.load(src)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    iteration = ckpt["global_step"]
    epoch = ckpt["epoch"]

    return iteration, epoch

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
                    iteration: int, epoch: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    ckpt = {
        "model" : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "scheduler" : scheduler.state_dict() if scheduler is not None else None,
        "global_step" : iteration,
        "epoch" : epoch,
    }
    torch.save(ckpt, out)

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    total = 0
    params = list(parameters)
    for p in params:
        if p.grad is None:
            continue
        g = p.grad
        total += g.float().pow(2).sum()

    total_g = total.sqrt().item()

    if total_g > max_l2_norm:
        scale = max_l2_norm / (total_g + eps)
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)

    return total_g

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
    

class WarmupCosineLR(LRScheduler):
    def __init__(self, 
                 optimizer, 
                 warmup_steps: int, 
                 max_steps: int, 
                 min_lr: float ,
                 last_epoch = -1,):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch

        lrs = []

        for base_lr in self.base_lrs:
            if t < self.warmup_steps:
                lr = base_lr * t / self.warmup_steps

            elif t <= self.max_steps:
                progress = (t - self.warmup_steps) / (
                    self.max_steps - self.warmup_steps
                )
                cosine = 0.5 * (1 + math.cos(math.pi * progress))
                lr = self.min_lr + cosine * (base_lr - self.min_lr)

            else:
                lr = self.min_lr

            lrs.append(lr)

        return lrs