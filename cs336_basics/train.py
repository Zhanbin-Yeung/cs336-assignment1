from torch.utils.data import DataLoader
import torch
import yaml
from easydict import EasyDict
import tqdm
from collections import deque
import os

from dataset import Datasets, IterableDatasets
from transformer import Transformer, AdamW, CrossEntropyLoss
from utils import save_checkpoint, load_checkpoint

def set_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

 
def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        cfg = EasyDict(cfg)
    os.makedirs(getattr(cfg, "checkpoint_path", "checkpoints"), exist_ok=True)

    dataset = IterableDatasets(cfg.input_file, cfg.context_length, cfg.batch_size)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=4, pin_memory=True)

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
    optimizer = AdamW(model.parameters(), cfg.lr, (cfg.beta1, cfg.beta2), cfg.weight_decay)

    start_epoch = 0
    global_step = 0

    if getattr(cfg, "resume", None):
        print(f"Resuming from {cfg.resume}")
        global_step, start_epoch = load_checkpoint(cfg.resume, model, optimizer)
    else:
        print("Training from scratch")

    log_interval = cfg.log_interval
    loss_buf = deque(maxlen=log_interval)

    for epoch in range(start_epoch, cfg.epoch):
        model.train()
        dataset.set_epoch(epoch)
        pbar = tqdm(dataloader, desc=f"Epoch: {epoch + 1} / {cfg.epoch}")

        for i, x, y in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            y_hat = model(x)
            loss = CrossEntropyLoss(y_hat, y)
            loss_value = loss.item()
            loss_buf.append(loss_value)

            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % log_interval == 0:
                pbar.set_postfix({"loss": f"{sum(loss_buf) / len(loss_buf):.4f}",
                                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                                })
                
            if global_step % cfg.save_interval == 0:
                path = os.path.join(cfg.ckpt_dir, f"ckpt_epoch{epoch}_step{global_step}.pt")
                save_checkpoint(model, optimizer, global_step, epoch, path)

        path = os.path.join(cfg.ckpt_dir, f"ckpt_epoch{epoch}.pt")
        save_checkpoint(model, optimizer, global_step, epoch, path)

if __name__ == "__main__":
    main()
        



