
import typing
import os
import torch

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    ckpt = torch.load(src)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    iteration = ckpt["global_step"]
    epoch = ckpt["epoch"]

    return iteration, epoch

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, epoch: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    ckpt = {
        "model" : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "global_step" : iteration,
        "epoch" : epoch,
    }
    torch.save(ckpt, out)