import os
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from tqdm import tqdm, trange
from PIL import Image
from tqdm import trange

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
torch.set_default_dtype(torch.float32)
# ===== MODEL COMPONENTS =====


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), "attn index out of bound"
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(
                    ResBlock(
                        in_ch=now_ch,
                        out_ch=out_ch,
                        tdim=tdim,
                        dropout=dropout,
                        attn=(i in attn),
                    )
                )
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList(
            [
                ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
                ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
            ]
        )

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(
                    ResBlock(
                        in_ch=chs.pop() + now_ch,
                        out_ch=out_ch,
                        tdim=tdim,
                        dropout=dropout,
                        attn=(i in attn),
                    )
                )
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1),
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h


# ===== DIFFUSION COMPONENTS =====


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer("betas", torch.linspace(beta_1, beta_T, T).float())
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0
            + extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )
        loss = F.mse_loss(self.model(x_t, t), noise, reduction="none")
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(
        self,
        model,
        beta_1,
        beta_T,
        T,
        img_size=32,
        mean_type="eps",
        var_type="fixedlarge",
    ):
        assert mean_type in ["xprev", "xstart", "epsilon"]
        assert var_type in ["fixedlarge", "fixedsmall"]
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer("betas", torch.linspace(beta_1, beta_T, T).float())
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_recip_alphas_bar", torch.sqrt(1.0 / alphas_bar))
        self.register_buffer(
            "sqrt_recipm1_alphas_bar", torch.sqrt(1.0 / alphas_bar - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            "posterior_var", self.betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        )
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_var_clipped",
            torch.log(torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            torch.sqrt(alphas_bar_prev) * self.betas / (1.0 - alphas_bar),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            torch.sqrt(alphas) * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar),
        )

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            "fixedlarge": torch.log(
                torch.cat([self.posterior_var[1:2], self.betas[1:]])
            ),
            "fixedsmall": self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == "xprev":  # the model predicts x_{t-1}
            x_prev = self.model(x_t, t)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == "xstart":  # the model predicts x_0
            x_0 = self.model(x_t, t)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == "epsilon":  # the model predicts epsilon
            eps = self.model(x_t, t)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        x_0 = torch.clip(x_0, -1.0, 1.0)

        return model_mean, model_log_var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = (
                x_t.new_ones(
                    [
                        x_T.shape[0],
                    ],
                    dtype=torch.long,
                )
                * time_step
            )
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
        x_0 = x_t
        return torch.clip(x_0, -1, 1)


# ===== UTILITY FUNCTIONS =====


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


def warmup_lr(step, warmup):
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
    img = Image.fromarray((grid * 255).astype(np.uint8))
    img.save(path)


# ===== DATASET FUNCTIONS =====


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor:
    def __call__(self, pic):
        img = np.array(pic, dtype=np.float32) / 255.0
        img = img.transpose((2, 0, 1))  # Convert HWC to CHW
        return torch.from_numpy(img)


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


class RandomHorizontalFlip:
    def __call__(self, img):
        import random

        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


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


def load_cifar10(root="./data", train=True, download=True):
    """Load CIFAR10 dataset without torchvision"""
    transform = Compose(
        [
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    return CIFAR10(root=root, train=train, download=download, transform=transform)


# ===== EVALUATION FUNCTIONS =====


def evaluate(
    sampler,
    model,
    num_images,
    batch_size,
    img_size,
    device,
    fid_cache,
    fid_use_torch=False,
):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, num_images, batch_size, desc=desc):
            current_batch_size = min(batch_size, num_images - i)
            # Explicitly set dtype to float32 for MPS compatibility
            x_T = torch.randn(
                (current_batch_size, 3, img_size, img_size), dtype=torch.float32
            )
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


# ===== INCEPTION SCORE FUNCTIONS =====


def get_inception_and_fid_score(
    images,
    fid_cache,
    num_images=None,
    splits=10,
    batch_size=50,
    use_torch=False,
    verbose=False,
    parallel=False,
):
    """when `images` is a python generator, `num_images` should be given"""
    import types
    import pickle

    # Fix imports to avoid relative imports
    try:
        from score.inception import InceptionV3
        from score.fid import calculate_frechet_distance, torch_cov
    except ImportError:
        # Fallback if the score directory is not in the Python path
        import sys
        import os

        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from score.inception import InceptionV3
        from score.fid import calculate_frechet_distance, torch_cov

    if num_images is None and isinstance(images, types.GeneratorType):
        raise ValueError(
            "when `images` is a python generator, " "`num_images` should be given"
        )

    if num_images is None:
        num_images = len(images)

    block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    block_idx2 = InceptionV3.BLOCK_INDEX_BY_DIM["prob"]
    model = InceptionV3([block_idx1, block_idx2]).to(device)
    model.eval()

    if parallel:
        model = torch.nn.DataParallel(model)

    if use_torch:
        fid_acts = torch.empty((num_images, 2048), dtype=torch.float32).to(device)
        is_probs = torch.empty((num_images, 1008), dtype=torch.float32).to(device)
    else:
        fid_acts = np.empty((num_images, 2048), dtype=np.float32)
        is_probs = np.empty((num_images, 1008), dtype=np.float32)

    iterator = iter(
        tqdm(
            images,
            total=num_images,
            dynamic_ncols=True,
            leave=False,
            disable=not verbose,
            desc="get_inception_and_fid_score",
        )
    )

    start = 0
    while True:
        batch_images = []
        # get a batch of images from iterator
        try:
            for _ in range(batch_size):
                batch_images.append(next(iterator))
        except StopIteration:
            if len(batch_images) == 0:
                break
            pass
        batch_images = np.stack(batch_images, axis=0)
        end = start + len(batch_images)

        # calculate inception feature
        batch_images = torch.from_numpy(batch_images).type(torch.FloatTensor)
        batch_images = batch_images.to(device)
        with torch.no_grad():
            pred = model(batch_images)
            if use_torch:
                fid_acts[start:end] = pred[0].view(-1, 2048)
                is_probs[start:end] = pred[1]
            else:
                fid_acts[start:end] = pred[0].view(-1, 2048).cpu().numpy()
                is_probs[start:end] = pred[1].cpu().numpy()
        start = end

    # Inception Score
    scores = []
    for i in range(splits):
        part = is_probs[
            (i * is_probs.shape[0] // splits) : ((i + 1) * is_probs.shape[0] // splits),
            :,
        ]
        if use_torch:
            kl = part * (
                torch.log(part) - torch.log(torch.unsqueeze(torch.mean(part, 0), 0))
            )
            kl = torch.mean(torch.sum(kl, 1))
            scores.append(torch.exp(kl))
        else:
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
    if use_torch:
        scores = torch.stack(scores)
        is_score = (torch.mean(scores).cpu().item(), torch.std(scores).cpu().item())
    else:
        is_score = (np.mean(scores), np.std(scores))

    # FID Score
    f = np.load(fid_cache)
    m2, s2 = f["mu"][:], f["sigma"][:]
    f.close()
    if use_torch:
        m1 = torch.mean(fid_acts, axis=0)
        s1 = torch_cov(fid_acts, rowvar=False)
        m2 = torch.tensor(m2, dtype=torch.float32).to(device)
        s2 = torch.tensor(s2, dtype=torch.float32).to(device)
    else:
        m1 = np.mean(fid_acts, axis=0)
        s1 = np.cov(fid_acts, rowvar=False)
    fid_score = calculate_frechet_distance(m1, s1, m2, s2, use_torch=use_torch)

    del fid_acts, is_probs, scores, model
    return is_score, fid_score
