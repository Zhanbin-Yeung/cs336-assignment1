from torch.utils.data import DataLoader
import torch
import yaml
from easydict import EasyDict
import tqdm
from collections import deque
import os
from torch.amp import autocast, GradScaler
import wandb

from dataset import IterableDatasets
from transformer import Transformer, CrossEntropyLoss
from utils import save_checkpoint, load_checkpoint, gradient_clipping, AdamW, WarmupCosineLR


def set_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

@torch.no_grad()
def evaluate(model, dataloader, device, max_steps=100):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        vbar = tqdm.tqdm(total=max_steps, desc="val", leave=False)
        for step, (x, y) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(device_type=device, dtype=torch.bfloat16):
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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = EasyDict(yaml.safe_load(f))

    # --- W&B: 同组对比不同学习率曲线 ---
    wandb.init(
        project=cfg.wandb_project,
        group=cfg.wandb_group,
        name=f"owt-train",
        config=dict(cfg),
        id="2rhut8jx",
        resume="allow"
    )

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    dataset = IterableDatasets(cfg.input_file, cfg.context_length, cfg.micro_batch_size)
    dataset.set_epoch(0)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=True)

    val_dataset = IterableDatasets(cfg.val_input_file, cfg.context_length, cfg.micro_batch_size)
    val_dataset.set_epoch(0)
    val_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=0, pin_memory=True)

    device = set_device()
    print(f"Using device: {device}")

    model = Transformer(
        cfg.vocab_size,
        cfg.context_length,
        cfg.d_model,
        cfg.num_layers,
        cfg.num_heads,
        cfg.d_ff,
        cfg.rope_theta,
    ).to(device)

    optimizer = AdamW(model.parameters(), float(cfg.lr), (cfg.beta1, cfg.beta2), cfg.weight_decay)
    scheduler = WarmupCosineLR(optimizer, warmup_steps=cfg.warmup_steps, max_steps=cfg.max_steps, min_lr=cfg.min_lr)
    # scheduler = None
    global_step = 0
    accum_steps = cfg.batch_size // cfg.micro_batch_size
    assert cfg.batch_size % cfg.micro_batch_size == 0

    if cfg.resume is not None:
        print(f"Resuming from {cfg.resume}")
        global_step, _ = load_checkpoint(cfg.resume, model, optimizer, scheduler)
        scheduler.max_steps = cfg.max_steps
    else:
        print("Training from scratch")

    log_interval = cfg.log_interval
    loss_buf = deque(maxlen=log_interval)

    # --- no epoch: 只训练到 max_steps ---
    last_val_loss = None

    pbar = tqdm.tqdm(total=cfg.max_steps - global_step, desc="train", initial=0)


    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    while global_step < cfg.max_steps:
        
        # 每次进入此 for 循环，DataLoader 会自动初始化 Worker 进程
        # 当数据集遍历完一轮抛出 StopIteration 时，for 循环结束，Worker 资源被释放回收
        for x, y in dataloader:

            if global_step >= cfg.max_steps:
                break  
            
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(device_type=device, dtype=torch.bfloat16):
                y_hat = model(x)
                loss = CrossEntropyLoss(y_hat, y)

            loss = loss / accum_steps
            loss.backward()
            
            micro_step += 1

            if micro_step == accum_steps:
                
                grad_norm = gradient_clipping(model.parameters(), max_l2_norm=cfg.grad_clip)

                optimizer.step()
                scheduler.step()

                optimizer.zero_grad(set_to_none=True)
                micro_step = 0

                loss_value = loss.item() * accum_steps
                loss_buf.append(loss_value)

                global_step += 1
                pbar.update(1)

                # --- val + 达标自动停 ---
                if global_step % cfg.val_interval == 0:
                    val_loss = evaluate(model, val_dataloader, device, max_steps=cfg.val_steps)
                    last_val_loss = float(val_loss)

                    wandb.log({"val/loss": last_val_loss}, step=global_step)

                    tqdm.tqdm.write(f"step {global_step} | val_loss {last_val_loss:.4f}")
                    if last_val_loss <= cfg.target_val_loss:
                        print(f"[STOP] val_loss {last_val_loss:.4f} <= {cfg.target_val_loss:.4f} at step {global_step}")
                        # 注意：这里需要跳出两层循环。为了简单，可以直接返回或设置标志位
                        return 

                # --- train log ---
                if global_step % log_interval == 0:
                    avg_loss = sum(loss_buf) / len(loss_buf)
                    lr = optimizer.param_groups[0]["lr"]

                    wandb.log(
                        {
                            "train/loss": float(avg_loss),
                            "train/lr": float(lr),
                            "train/grad_norm": float(grad_norm),
                        },
                        step=global_step
                    )

                    tqdm.tqdm.write(
                        f"step {global_step} | "
                        f"train_loss {avg_loss:.4f} | "
                        # f"val_loss {(last_val_loss if last_val_loss is not None else float('nan')):.4f} | "
                        f"lr {lr:.4e}"
                    )

                # --- save ---
                if global_step % cfg.save_interval == 0:
                    path = os.path.join(cfg.checkpoint_dir, f"ckpt_step{global_step}.pt")
                    save_checkpoint(model, optimizer, scheduler, global_step, 0, path)

    # 最后存一次
    path = os.path.join(cfg.checkpoint_dir, f"ckpt_final_step{global_step}.pt")
    save_checkpoint(model, optimizer, scheduler, global_step, 0, path)

    pbar.close()
    wandb.finish()


if __name__ == "__main__":
    main()