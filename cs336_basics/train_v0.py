from torch.utils.data import DataLoader
import torch
import yaml
from easydict import EasyDict
import tqdm
from collections import deque
import os
import time
from torch.amp import autocast, GradScaler
import wandb

from dataset import Datasets, IterableDatasets
from transformer import Transformer, CrossEntropyLoss
from utils import save_checkpoint, load_checkpoint, gradient_clipping, AdamW, WarmupCosineLR

def set_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def evaluate(model, dataloader, device, max_steps=100):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        vbar = tqdm.tqdm(total=max_steps, desc="val", leave=False)
        for step, (x, y) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(device_type=device, dtype=torch.float16):
                y_hat = model(x)
                loss = CrossEntropyLoss(y_hat, y)
            
            B, T = x.shape
            total_loss += loss.item() * B * T
            total_tokens += B * T

            vbar.update(1)
            if step + 1 >= max_steps:
                break

        vbar.close()

    model.train()

    return total_loss / total_tokens


def main():
    wandb.init(project="cs336-basics", name="tinystories_train")

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        cfg = EasyDict(cfg)
    os.makedirs(getattr(cfg, "checkpoint_path", "checkpoints"), exist_ok=True)

    dataset = IterableDatasets(cfg.input_file, cfg.context_length, cfg.batch_size)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=8, pin_memory=True)

    val_dataset = IterableDatasets(cfg.val_input_file, cfg.context_length, cfg.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=4, pin_memory=True)

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
    optimizer = AdamW(model.parameters(), float(cfg.lr), (cfg.beta1, cfg.beta2), cfg.weight_decay)
    scheduler = WarmupCosineLR(optimizer, warmup_steps=cfg.warmup_steps, max_steps=cfg.max_steps, min_lr=cfg.min_lr)

    start_epoch = 0
    global_step = 0

    if getattr(cfg, "resume", None):
        print(f"Resuming from {cfg.resume}")
        global_step, start_epoch = load_checkpoint(cfg.resume, model, optimizer,scheduler)
    else:
        print("Training from scratch")

    log_interval = cfg.log_interval
    loss_buf = deque(maxlen=log_interval)
    scaler = GradScaler()
    
    for epoch in range(start_epoch, cfg.epoch):
        model.train()
        dataset.set_epoch(epoch)
        pbar = tqdm.tqdm(dataloader, desc=f"Epoch: {epoch + 1} / {cfg.epoch}")


        t_data = 0.0
        t_compute = 0.0
        for step, (x, y) in enumerate(pbar):
            t0 = time.time()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            torch.cuda.synchronize()
            t1 = time.time()

            optimizer.zero_grad(set_to_none=True)
            # with autocast():
            with autocast(device_type=device, dtype=torch.float16):
                y_hat = model(x)
                loss = CrossEntropyLoss(y_hat, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = gradient_clipping(model.parameters(), max_l2_norm=cfg.grad_clip)

            scaler.step(optimizer)

            torch.cuda.synchronize()
            t2 = time.time()
            t_data += (t1 - t0)
            t_compute += (t2 - t1)
            if step % 50 == 0:
                print(f"data={t_data/50:.4f}s/it  compute={t_compute/50:.4f}s/it")
                t_data = t_compute = 0.0

            scheduler.step()
            scaler.update()
            # y_hat = model(x)
            # loss = CrossEntropyLoss(y_hat, y)
            loss_value = loss.item()
            loss_buf.append(loss_value)

            # loss.backward()
            # optimizer.step()

            global_step += 1

            if global_step % cfg.val_interval == 0:
                val_loss = evaluate(model, val_dataloader, device)
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "epoch": epoch,
                    },
                    step=global_step
                )
                
                # tqdm.tqdm.write(
                #     f"Epoch {epoch+1}/{cfg.epoch} | "
                #     f"step {global_step} | "
                #     f"val_loss {val_loss:.4f}"
                # )

            if global_step % log_interval == 0:

                avg_loss = sum(loss_buf) / len(loss_buf)
                lr = optimizer.param_groups[0]["lr"]
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/grad_norm": float(grad_norm),
                        "epoch": epoch,
                    },
                    step=global_step
                )

                tqdm.tqdm.write(
                    f"Epoch {epoch+1}/{cfg.epoch} | "
                    f"step {global_step} | "
                    f"train_loss {avg_loss:.4f} | "
                    f"val_loss {val_loss:.4f} | "
                    f"lr {lr:.4e}"
                )
            
            
            if global_step % cfg.save_interval == 0:
                path = os.path.join(cfg.checkpoint_path, f"ckpt_epoch{epoch}_step{global_step}.pt")
                save_checkpoint(model, optimizer, scheduler, global_step, epoch, path)

        path = os.path.join(cfg.checkpoint_path, f"ckpt_epoch{epoch}.pt")
        save_checkpoint(model, optimizer, scheduler, global_step, epoch, path)
    
    wandb.finish()

if __name__ == "__main__":
    main()
        
