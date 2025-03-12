# %%
import copy
import json
import os
import warnings
import torch
from tensorboardX import SummaryWriter
from utils import (
    UNet,
    GaussianDiffusionTrainer,
    GaussianDiffusionSampler,
    ema,
    infiniteloop,
    warmup_lr,
    make_grid,
    save_image,
    evaluate,
    load_cifar10,
    trange,
)

# Define configuration parameters
# General settings
train_mode = True
eval_mode = False

# UNet parameters
ch = 128
ch_mult = [1, 2, 2, 2]
attn = [1]
num_res_blocks = 2
dropout = 0.1

# Gaussian Diffusion parameters
beta_1 = 1e-4
beta_T = 0.02
T = 1000
mean_type = "epsilon"
var_type = "fixedlarge"

# Training parameters
lr = 2e-4
grad_clip = 1.0
total_steps = 800000
img_size = 32
warmup = 5000
batch_size = 128
num_workers = 4
ema_decay = 0.9999
parallel = False

# Logging & Sampling
logdir = "./logs/DDPM_CIFAR10_EPS"
sample_size = 64
sample_step = 1000

# Evaluation
save_step = 5000
eval_step = 0
num_images = 50000
fid_use_torch = False
fid_cache = "./stats/cifar10.train.npz"

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Ensure all tensors are float32 for MPS compatibility
torch.set_default_dtype(torch.float32)


def train():
    # dataset
    dataset = load_cifar10(root="./data", train=True, download=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    datalooper = infiniteloop(dataloader)

    # model setup
    net_model = UNet(
        T=T,
        ch=ch,
        ch_mult=ch_mult,
        attn=attn,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
    )
    ema_model = copy.deepcopy(net_model)
    net_model.to(device)
    ema_model.to(device)

    optim = torch.optim.Adam(net_model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: warmup_lr(step, warmup)
    )
    trainer = GaussianDiffusionTrainer(net_model, beta_1, beta_T, T).to(device)
    net_sampler = GaussianDiffusionSampler(
        net_model,
        beta_1,
        beta_T,
        T,
        img_size,
        mean_type,
        var_type,
    ).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model,
        beta_1,
        beta_T,
        T,
        img_size,
        mean_type,
        var_type,
    ).to(device)
    if parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    # log setup
    os.makedirs(os.path.join(logdir, "sample"), exist_ok=True)
    x_T = torch.randn(sample_size, 3, img_size, img_size, dtype=torch.float32)
    x_T = x_T.to(device)

    # Get a batch of real samples for reference
    data_iter = iter(dataloader)
    real_batch = next(data_iter)
    real_samples = real_batch[0][:sample_size]
    grid = (make_grid(real_samples) + 1) / 2

    writer = SummaryWriter(logdir)
    writer.add_image("real_sample", grid)
    writer.flush()

    # backup all configuration parameters
    with open(os.path.join(logdir, "config.json"), "w") as f:
        config = {
            "ch": ch,
            "ch_mult": ch_mult,
            "attn": attn,
            "num_res_blocks": num_res_blocks,
            "dropout": dropout,
            "beta_1": beta_1,
            "beta_T": beta_T,
            "T": T,
            "mean_type": mean_type,
            "var_type": var_type,
            "lr": lr,
            "grad_clip": grad_clip,
            "total_steps": total_steps,
            "img_size": img_size,
            "warmup": warmup,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "ema_decay": ema_decay,
            "parallel": parallel,
            "sample_size": sample_size,
            "sample_step": sample_step,
            "save_step": save_step,
            "eval_step": eval_step,
            "num_images": num_images,
        }
        json.dump(config, f, indent=2)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    # start training
    with trange(total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0 = next(datalooper).to(device)
            loss = trainer(x_0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, ema_decay)

            # log
            writer.add_scalar("loss", loss, step)
            pbar.set_postfix(loss="%.3f" % loss)

            # sample
            if sample_step > 0 and step % sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = ema_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(logdir, "sample", "%d.png" % step)
                    save_image(grid, path)
                    writer.add_image("sample", grid, step)
                net_model.train()

            # save
            if save_step > 0 and step % save_step == 0:
                ckpt = {
                    "net_model": net_model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "sched": sched.state_dict(),
                    "optim": optim.state_dict(),
                    "step": step,
                    "x_T": x_T,
                }
                torch.save(ckpt, os.path.join(logdir, "ckpt.pt"))

            # evaluate
            if eval_step > 0 and step % eval_step == 0:
                net_IS, net_FID, _ = evaluate(
                    net_sampler,
                    net_model,
                    num_images,
                    batch_size,
                    img_size,
                    device,
                    fid_cache,
                    fid_use_torch,
                )
                ema_IS, ema_FID, _ = evaluate(
                    ema_sampler,
                    ema_model,
                    num_images,
                    batch_size,
                    img_size,
                    device,
                    fid_cache,
                    fid_use_torch,
                )
                metrics = {
                    "IS": net_IS[0],
                    "IS_std": net_IS[1],
                    "FID": net_FID,
                    "IS_EMA": ema_IS[0],
                    "IS_std_EMA": ema_IS[1],
                    "FID_EMA": ema_FID,
                }
                pbar.write(
                    "%d/%d " % (step, total_steps)
                    + ", ".join("%s:%.3f" % (k, v) for k, v in metrics.items())
                )
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(logdir, "eval.txt"), "a") as f:
                    metrics["step"] = step
                    f.write(json.dumps(metrics) + "\n")
    writer.close()


def eval():
    # model setup
    model = UNet(
        T=T,
        ch=ch,
        ch_mult=ch_mult,
        attn=attn,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
    )
    model.to(device)

    sampler = GaussianDiffusionSampler(
        model,
        beta_1,
        beta_T,
        T,
        img_size=img_size,
        mean_type=mean_type,
        var_type=var_type,
    ).to(device)
    if parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    ckpt = torch.load(os.path.join(logdir, "ckpt.pt"))

    model.load_state_dict(ckpt["ema_model"])
    (IS, IS_std), FID, samples = evaluate(
        sampler,
        model,
        num_images,
        batch_size,
        img_size,
        device,
        fid_cache,
        fid_use_torch,
    )
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))

    # Create directory if it doesn't exist
    os.makedirs(logdir, exist_ok=True)

    save_image(
        torch.tensor(samples[:256], dtype=torch.float32),
        os.path.join(logdir, "samples_ema.png"),
        nrow=16,
    )


# %%
# suppress annoying inception_v3 initialization warning
warnings.simplefilter(action="ignore", category=FutureWarning)

# Add multiprocessing safeguard
if __name__ == "__main__":
    # This ensures multiprocessing works correctly
    import multiprocessing

    multiprocessing.freeze_support()

    if train_mode:
        train()
    if eval_mode:
        eval()
    if not train_mode and not eval_mode:
        print(
            "Set train_mode=True and/or eval_mode=True to execute corresponding tasks"
        )

# %%
