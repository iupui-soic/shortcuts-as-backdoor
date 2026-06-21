"""Train a CycleGAN demographic counterfactual generator for the Phase 7 §8.3
CF demographic audit.

Learns an unpaired image translation between the two race subgroups of the
*matched* MIMIC-CXR cohort (WHITE <-> BLACK_OR_AA), so that at audit time we can
generate a demographic counterfactual CF(x) and score CF-inconsistency
|f(x) - f(CF(x))| on the target label. This is the real generator that replaces
`IdentityGenerator` in `src/defenses/cf_demographic_audit.py`.

Design notes:
- Standard CycleGAN: ResNet-9-block generators, 70x70 PatchGAN discriminators,
  LSGAN adversarial loss + L1 cycle (lambda 10) + identity (lambda 5), Adam
  2e-4 (betas 0.5/0.999), linear LR decay over the back half, 50-image replay
  buffer per discriminator.
- Trained at 224px in a canonical [-1, 1] RGB space (tanh output). The audit-time
  wrapper denormalizes the classifier's ImageNet-normalized input to [-1, 1],
  runs G, and renormalizes -- so the generator never sees a resolution/normalization
  mismatch. Keep --img-size in sync with the classifier eval size (224).
- Domain A = WHITE, Domain B = BLACK_OR_AA. We save BOTH directions
  (G_A2B = white->black, G_B2A = black->white) so the audit can flip either way.

Smoke: `python scripts/train_cf_cyclegan.py --smoke` runs 2 optimizer steps on a
tiny subset and exits non-error.
"""
from __future__ import annotations

import argparse
import functools
import itertools
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_ROOT = Path("/data0/MIMIC-CXR/files")
MANIFEST = Path("data/manifests/mimic_cxr_matched.parquet")
DOMAIN_A = "WHITE"
DOMAIN_B = "BLACK_OR_AA"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class UnalignedRaceDataset(Dataset):
    """Returns one domain-A image and one random domain-B image per index."""

    def __init__(self, paths_a, paths_b, transform, seed: int = 0):
        self.paths_a = list(paths_a)
        self.paths_b = list(paths_b)
        self.transform = transform
        # deterministic shuffle of B (no Math.random equivalent needed; torch gen)
        g = torch.Generator().manual_seed(seed)
        self._perm = torch.randperm(len(self.paths_b), generator=g).tolist()

    def __len__(self) -> int:
        return max(len(self.paths_a), len(self.paths_b))

    def _load(self, p):
        try:
            img = Image.open(p).convert("RGB")
        except OSError:
            time.sleep(0.05)
            img = Image.open(p).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx: int):
        a = self.paths_a[idx % len(self.paths_a)]
        # decorrelate B from A by indexing through a fixed permutation
        b = self.paths_b[self._perm[idx % len(self.paths_b)]]
        return {"A": self._load(a), "B": self._load(b)}


def build_paths(n_per_domain: int | None, split: str = "train"):
    df = pd.read_parquet(MANIFEST)
    df = df[df["official_split"] == split]
    a = [str(IMAGE_ROOT / r) for r in df[df["race_group"] == DOMAIN_A]["relpath"]]
    b = [str(IMAGE_ROOT / r) for r in df[df["race_group"] == DOMAIN_B]["relpath"]]
    if n_per_domain is not None:
        a, b = a[:n_per_domain], b[:n_per_domain]
    return a, b


# --------------------------------------------------------------------------- #
# Models (standard CycleGAN building blocks)
# --------------------------------------------------------------------------- #
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), norm(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), norm(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=9, norm=nn.InstanceNorm2d):
        super().__init__()
        nrm = functools.partial(norm, affine=False, track_running_stats=False)
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_ch, ngf, 7), nrm(ngf), nn.ReLU(True)]
        # downsample x2
        mult = 1
        for _ in range(2):
            layers += [nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                       nrm(ngf * mult * 2), nn.ReLU(True)]
            mult *= 2
        for _ in range(n_blocks):
            layers += [ResnetBlock(ngf * mult, nrm)]
        # upsample x2
        for _ in range(2):
            layers += [nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3, stride=2,
                                          padding=1, output_padding=1),
                       nrm(ngf * mult // 2), nn.ReLU(True)]
            mult //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_ch, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, ndf=64, n_layers=3, norm=nn.InstanceNorm2d):
        super().__init__()
        nrm = functools.partial(norm, affine=False, track_running_stats=False)
        layers = [nn.Conv2d(in_ch, ndf, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layers):
            prev, mult = mult, min(2 ** n, 8)
            layers += [nn.Conv2d(ndf * prev, ndf * mult, 4, stride=2, padding=1),
                       nrm(ndf * mult), nn.LeakyReLU(0.2, True)]
        prev, mult = mult, min(2 ** n_layers, 8)
        layers += [nn.Conv2d(ndf * prev, ndf * mult, 4, stride=1, padding=1),
                   nrm(ndf * mult), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(ndf * mult, 1, 4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ImagePool:
    """50-image replay buffer (Shrivastava et al. / CycleGAN trick)."""

    def __init__(self, size=50):
        self.size = size
        self.imgs = []

    def query(self, images):
        if self.size == 0:
            return images
        out = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.imgs) < self.size:
                self.imgs.append(img)
                out.append(img)
            else:
                # deterministic-ish replace: cycle through buffer slots by hash of norm
                idx = int(img.abs().sum().item()) % self.size
                if idx % 2 == 0:
                    out.append(self.imgs[idx].clone())
                    self.imgs[idx] = img
                else:
                    out.append(img)
        return torch.cat(out, 0)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=40, help="constant LR for first half, linear decay second half")
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda-cycle", type=float, default=10.0)
    ap.add_argument("--lambda-identity", type=float, default=5.0)
    ap.add_argument("--ngf", type=int, default=64)
    ap.add_argument("--n-blocks", type=int, default=9)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default="results/phase7/cf_cyclegan")
    ap.add_argument("--sample-every", type=int, default=500, help="iters between sample grids")
    ap.add_argument("--save-every", type=int, default=1, help="epochs between checkpoints")
    ap.add_argument("--n-per-domain", type=int, default=None, help="cap images/domain (debug)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.batch_size, args.n_per_domain, args.workers = 1, 2, 8, 0
        args.img_size, args.sample_every = 128, 100000

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)

    tf = transforms.Compose([
        transforms.Resize(int(args.img_size * 1.12)),
        transforms.RandomCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
    ])
    paths_a, paths_b = build_paths(args.n_per_domain)
    ds = UnalignedRaceDataset(paths_a, paths_b, tf)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True)
    print(f"[data] A({DOMAIN_A})={len(paths_a)}  B({DOMAIN_B})={len(paths_b)}  "
          f"iters/epoch={len(dl)}  img={args.img_size}  bs={args.batch_size}", flush=True)

    G_A2B = ResnetGenerator(ngf=args.ngf, n_blocks=args.n_blocks).to(device).apply(init_weights)
    G_B2A = ResnetGenerator(ngf=args.ngf, n_blocks=args.n_blocks).to(device).apply(init_weights)
    D_A = PatchDiscriminator().to(device).apply(init_weights)
    D_B = PatchDiscriminator().to(device).apply(init_weights)

    crit_gan, crit_cyc, crit_id = nn.MSELoss(), nn.L1Loss(), nn.L1Loss()
    opt_g = torch.optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
                             lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()),
                             lr=args.lr, betas=(0.5, 0.999))

    half = max(1, args.epochs // 2)
    lam = lambda e: 1.0 - max(0, e - half) / float(args.epochs - half + 1)
    sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda=lam)
    sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda=lam)

    pool_a, pool_b = ImagePool(), ImagePool()

    def set_grad(nets, flag):
        for net in nets:
            for p in net.parameters():
                p.requires_grad = flag

    json.dump(vars(args), open(out / "train_args.json", "w"), indent=1)
    gstep = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        for i, batch in enumerate(dl):
            real_a = batch["A"].to(device, non_blocking=True)
            real_b = batch["B"].to(device, non_blocking=True)
            valid = torch.ones(1, device=device)  # broadcast vs patch map

            # ---- Generators ----
            set_grad([D_A, D_B], False)
            opt_g.zero_grad()
            fake_b = G_A2B(real_a)
            fake_a = G_B2A(real_b)
            # identity
            id_loss = (crit_id(G_A2B(real_b), real_b) + crit_id(G_B2A(real_a), real_a)) \
                * args.lambda_identity
            # adversarial
            gan_loss = crit_gan(D_B(fake_b), valid.expand_as(D_B(fake_b))) \
                + crit_gan(D_A(fake_a), valid.expand_as(D_A(fake_a)))
            # cycle
            cyc_loss = (crit_cyc(G_B2A(fake_b), real_a) + crit_cyc(G_A2B(fake_a), real_b)) \
                * args.lambda_cycle
            g_loss = id_loss + gan_loss + cyc_loss
            g_loss.backward()
            opt_g.step()

            # ---- Discriminators ----
            set_grad([D_A, D_B], True)
            opt_d.zero_grad()
            for D, real, fake_pool, fake in [(D_A, real_a, pool_a, fake_a),
                                             (D_B, real_b, pool_b, fake_b)]:
                pred_real = D(real)
                d_real = crit_gan(pred_real, torch.ones_like(pred_real))
                fk = fake_pool.query(fake.detach())
                pred_fake = D(fk)
                d_fake = crit_gan(pred_fake, torch.zeros_like(pred_fake))
                (0.5 * (d_real + d_fake)).backward()
            opt_d.step()

            gstep += 1
            if i % 50 == 0:
                print(f"[e{epoch} i{i}/{len(dl)}] G={g_loss.item():.3f} "
                      f"(gan={gan_loss.item():.3f} cyc={cyc_loss.item():.3f} "
                      f"id={id_loss.item():.3f}) lr={sch_g.get_last_lr()[0]:.2e}", flush=True)
            if gstep % args.sample_every == 0:
                with torch.no_grad():
                    grid = torch.cat([real_a[:4], fake_b[:4], real_b[:4], fake_a[:4]], 0)
                save_image(grid * 0.5 + 0.5, out / "samples" / f"e{epoch}_s{gstep}.png", nrow=4)
            if args.smoke and gstep >= 2:
                print("[smoke] 2 steps ok", flush=True)
                save_image((real_a[:2] * 0.5 + 0.5), out / "samples" / "smoke.png")
                return

        sch_g.step(); sch_d.step()
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            torch.save({"G_A2B": G_A2B.state_dict(), "G_B2A": G_B2A.state_dict(),
                        "epoch": epoch, "args": vars(args),
                        "domain_a": DOMAIN_A, "domain_b": DOMAIN_B},
                       out / "ckpt" / f"cyclegan_e{epoch}.pt")
            torch.save({"G_A2B": G_A2B.state_dict(), "G_B2A": G_B2A.state_dict(),
                        "epoch": epoch, "args": vars(args),
                        "domain_a": DOMAIN_A, "domain_b": DOMAIN_B},
                       out / "ckpt" / "cyclegan_last.pt")
        print(f"[epoch {epoch} done in {time.time()-t0:.0f}s]", flush=True)
    print("[train] complete", flush=True)


if __name__ == "__main__":
    main()
