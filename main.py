import copy
import json
import os
import warnings

import torch
from absl import app, flags
from tensorboardX import SummaryWriter
from tqdm import trange

from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNet
from score.both import get_inception_and_fid_score

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean("train", False, "train from scratch")
flags.DEFINE_boolean("eval", False, "load ckpt.pt and evaluate FID and IS")

# UNet parameters
flags.DEFINE_integer("ch", 128, "base channel of UNet")
flags.DEFINE_list("ch_mult", [1, 2, 2, 2], "channel multiplier")
flags.DEFINE_list("attn", [1], "add attention to these levels")
flags.DEFINE_integer("num_res_blocks", 2, "number of resblock in each level")
flags.DEFINE_float("dropout", 0.1, "dropout rate of resblock")

# Gaussian Diffusion parameters
flags.DEFINE_float("beta_1", 1e-4, "start beta value")
flags.DEFINE_float("beta_T", 0.02, "end beta value")
flags.DEFINE_integer("T", 1000, "total diffusion steps")
flags.DEFINE_string(
    "mean_type", "epsilon", "predict variable: 'xprev', 'xstart', or 'epsilon'"
)
flags.DEFINE_string(
    "var_type", "fixedlarge", "variance type: 'fixedlarge' or 'fixedsmall'"
)

# Training parameters
flags.DEFINE_float("lr", 2e-4, "target learning rate")
flags.DEFINE_float("grad_clip", 1.0, "gradient norm clipping")
flags.DEFINE_integer("total_steps", 800000, "total training steps")
flags.DEFINE_integer("img_size", 32, "image size")
flags.DEFINE_integer("warmup", 5000, "learning rate warmup")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("num_workers", 4, "workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, "ema decay rate")
flags.DEFINE_boolean("parallel", False, "multi gpu training")

# Logging & Sampling
flags.DEFINE_string("logdir", "./logs/DDPM_CIFAR10_EPS", "log directory")
flags.DEFINE_integer("sample_size", 64, "sampling size of images")
flags.DEFINE_integer("sample_step", 1000, "frequency of sampling")

# Evaluation
flags.DEFINE_integer("save_step", 5000, "frequency of saving checkpoints")
flags.DEFINE_integer("eval_step", 0, "frequency of evaluating model")
flags.DEFINE_integer(
    "num_images", 50000, "the number of generated images for evaluation"
)
flags.DEFINE_boolean("fid_use_torch", False, "calculate IS and FID on gpu")
flags.DEFINE_string("fid_cache", "./stats/cifar10.train.npz", "FID cache")

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
    return min(step, FLAGS.warmup) / FLAGS.warmup


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
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images,
        FLAGS.fid_cache,
        num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch,
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
    import numpy as np

    # dataset
    dataset = load_cifar10(root="./data", train=True, download=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )
    datalooper = infiniteloop(dataloader)

    # model setup
    net_model = UNet(
        T=FLAGS.T,
        ch=FLAGS.ch,
        ch_mult=[int(m) for m in FLAGS.ch_mult],
        attn=[int(a) for a in FLAGS.attn],
        num_res_blocks=FLAGS.num_res_blocks,
        dropout=FLAGS.dropout,
    )
    ema_model = copy.deepcopy(net_model)
    net_model.to(device)
    ema_model.to(device)

    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T
    ).to(device)
    net_sampler = GaussianDiffusionSampler(
        net_model,
        FLAGS.beta_1,
        FLAGS.beta_T,
        FLAGS.T,
        FLAGS.img_size,
        FLAGS.mean_type,
        FLAGS.var_type,
    ).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model,
        FLAGS.beta_1,
        FLAGS.beta_T,
        FLAGS.T,
        FLAGS.img_size,
        FLAGS.mean_type,
        FLAGS.var_type,
    ).to(device)
    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    # log setup
    os.makedirs(os.path.join(FLAGS.logdir, "sample"), exist_ok=True)
    x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size)
    x_T = x_T.to(device)

    # Get a batch of real samples for reference
    real_samples = next(iter(dataloader))[0][: FLAGS.sample_size]
    grid = (make_grid(real_samples) + 1) / 2

    writer = SummaryWriter(FLAGS.logdir)
    writer.add_image("real_sample", grid)
    writer.flush()

    # backup all arguments
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), "w") as f:
        f.write(FLAGS.flags_into_string())

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    # start training
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0 = next(datalooper).to(device)
            loss = trainer(x_0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            # log
            writer.add_scalar("loss", loss, step)
            pbar.set_postfix(loss="%.3f" % loss)

            # sample
            if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = ema_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(FLAGS.logdir, "sample", "%d.png" % step)
                    save_image(grid, path)
                    writer.add_image("sample", grid, step)
                net_model.train()

            # save
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                ckpt = {
                    "net_model": net_model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "sched": sched.state_dict(),
                    "optim": optim.state_dict(),
                    "step": step,
                    "x_T": x_T,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, "ckpt.pt"))

            # evaluate
            if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
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
                    "%d/%d " % (step, FLAGS.total_steps)
                    + ", ".join("%s:%.3f" % (k, v) for k, v in metrics.items())
                )
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, "eval.txt"), "a") as f:
                    metrics["step"] = step
                    f.write(json.dumps(metrics) + "\n")
    writer.close()


def eval():
    import numpy as np

    # model setup
    model = UNet(
        T=FLAGS.T,
        ch=FLAGS.ch,
        ch_mult=[int(m) for m in FLAGS.ch_mult],
        attn=[int(a) for a in FLAGS.attn],
        num_res_blocks=FLAGS.num_res_blocks,
        dropout=FLAGS.dropout,
    )
    model.to(device)

    sampler = GaussianDiffusionSampler(
        model,
        FLAGS.beta_1,
        FLAGS.beta_T,
        FLAGS.T,
        img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type,
        var_type=FLAGS.var_type,
    ).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    ckpt = torch.load(os.path.join(FLAGS.logdir, "ckpt.pt"))

    model.load_state_dict(ckpt["ema_model"])
    (IS, IS_std), FID, samples = evaluate(sampler, model)
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))

    # Create directory if it doesn't exist
    os.makedirs(FLAGS.logdir, exist_ok=True)

    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, "samples_ema.png"),
        nrow=16,
    )


def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action="ignore", category=FutureWarning)
    if FLAGS.train:
        train()
    if FLAGS.eval:
        eval()
    if not FLAGS.train and not FLAGS.eval:
        print("Add --train and/or --eval to execute corresponding tasks")


if __name__ == "__main__":
    app.run(main)
