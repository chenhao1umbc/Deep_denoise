import copy
import json
import os
import warnings

import torch
from tensorboardX import SummaryWriter
from tqdm import trange
import numpy as np
from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNet
from score.both import get_inception_and_fid_score

# Define configuration parameters
# General settings
train_mode = False
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, warmup) / warmup


def make_grid(images, nrow=8):
    """Simple function to create a grid of images"""
    b, c, h, w = images.shape
    rows = (b + nrow - 1) // nrow
    grid_img = torch.zeros((c, rows * h, nrow * w))

    for idx, img in enumerate(images):
        row_idx = idx // nrow
        col_idx = idx % nrow
        grid_img[
            :, row_idx * h : (row_idx + 1) * h, col_idx * w : (col_idx + 1) * w
        ] = img

    return grid_img


def save_image(tensor, path, nrow=8):
    """Save a tensor as an image"""
    grid = make_grid(tensor, nrow=nrow)
    # Convert to numpy and transpose to HWC format
    grid = grid.cpu().numpy().transpose(1, 2, 0)
    # Clip values to [0, 1]
    grid = np.clip(grid, 0, 1)
    # Save using PIL
    from PIL import Image

    img = Image.fromarray((grid * 255).astype(np.uint8))
    img.save(path)


def evaluate(sampler, model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, num_images, batch_size, desc=desc):
            current_batch_size = min(batch_size, num_images - i)
            x_T = torch.randn((current_batch_size, 3, img_size, img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images,
        fid_cache,
        num_images=num_images,
        use_torch=fid_use_torch,
        verbose=True,
    )
    return (IS, IS_std), FID, images


def load_cifar10(root="./data", train=True, download=True):
    """Load CIFAR10 dataset without torchvision"""
    import numpy as np
    import pickle
    from PIL import Image

    class CIFAR10:
        def __init__(self, root, train=True, transform=None, download=True):
            self.root = root
            self.train = train
            self.transform = transform

            if download:
                self.download()

            if train:
                files = [
                    "data_batch_1",
                    "data_batch_2",
                    "data_batch_3",
                    "data_batch_4",
                    "data_batch_5",
                ]
            else:
                files = ["test_batch"]

            self.data = []
            self.targets = []

            for file in files:
                file_path = os.path.join(root, "cifar-10-batches-py", file)
                with open(file_path, "rb") as f:
                    entry = pickle.load(f, encoding="latin1")
                    self.data.append(entry["data"])
                    self.targets.extend(entry["labels"])

            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            return img, target

        def __len__(self):
            return len(self.data)

        def download(self):
            import tarfile
            import urllib.request

            if os.path.exists(os.path.join(self.root, "cifar-10-batches-py")):
                return

            os.makedirs(self.root, exist_ok=True)

            url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            filename = os.path.join(self.root, "cifar-10-python.tar.gz")

            if not os.path.exists(filename):
                print(f"Downloading {url} to {filename}")
                urllib.request.urlretrieve(url, filename)

            with tarfile.open(filename, "r:gz") as tar:
                tar.extractall(path=self.root)

    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, img):
            for t in self.transforms:
                img = t(img)
            return img

    class ToTensor:
        def __call__(self, pic):
            import numpy as np

            img = np.array(pic, dtype=np.float32) / 255.0
            img = img.transpose((2, 0, 1))  # Convert HWC to CHW
            return torch.from_numpy(img)

    class Normalize:
        def __init__(self, mean, std):
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def __call__(self, tensor):
            return (tensor - self.mean) / self.std

    class RandomHorizontalFlip:
        def __call__(self, img):
            import random

            if random.random() < 0.5:
                return img.transpose(Image.FLIP_LEFT_RIGHT)
            return img

    transform = Compose(
        [
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    return CIFAR10(root=root, train=train, download=download, transform=transform)


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
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
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
    x_T = torch.randn(sample_size, 3, img_size, img_size)
    x_T = x_T.to(device)

    # Get a batch of real samples for reference
    real_samples = next(iter(dataloader))[0][:sample_size]
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
                net_IS, net_FID, _ = evaluate(net_sampler, net_model)
                ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model)
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
    import numpy as np

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
    (IS, IS_std), FID, samples = evaluate(sampler, model)
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))

    # Create directory if it doesn't exist
    os.makedirs(logdir, exist_ok=True)

    save_image(
        torch.tensor(samples[:256]),
        os.path.join(logdir, "samples_ema.png"),
        nrow=16,
    )


def main():
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action="ignore", category=FutureWarning)
    if train_mode:
        train()
    if eval_mode:
        eval()
    if not train_mode and not eval_mode:
        print(
            "Set train_mode=True and/or eval_mode=True to execute corresponding tasks"
        )
