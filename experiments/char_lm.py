"""Small char-level LM: cross-entropy vs POPLoss, matched seeds/steps/lr.

Corpus: the Python standard library's own source (always available, PSF-licensed), split by
file into train / held-out. Reports held-out bits per character (cross-entropy only, so the
objectives are compared on the same yardstick).

    python experiments/char_lm.py --betas 0 0.1 0.5 --seeds 0 1 2
"""
import argparse, glob, json, math, os, random, sys, sysconfig, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pop import POPLoss


def corpus(max_chars=4_000_000):
    files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
    random.Random(0).shuffle(files)
    text = {"train": [], "val": []}
    n = 0
    for i, f in enumerate(files):
        s = open(f, encoding="utf-8", errors="ignore").read()
        text["val" if i % 10 == 0 else "train"].append(s)
        n += len(s)
        if n > max_chars:
            break
    data = {k: "\n".join(v).encode("ascii", errors="replace") for k, v in text.items()}
    return {k: torch.tensor(list(v), dtype=torch.long) for k, v in data.items()}


class Block(torch.nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1, self.n2 = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)
        self.attn = torch.nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(d, 4 * d), torch.nn.GELU(), torch.nn.Linear(4 * d, d))

    def forward(self, x, mask):
        a = self.n1(x)
        x = x + self.attn(a, a, a, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class CharLM(torch.nn.Module):
    def __init__(self, d=192, h=6, layers=4, ctx=128):
        super().__init__()
        self.emb, self.pos = torch.nn.Embedding(128, d), torch.nn.Embedding(ctx, d)
        self.blocks = torch.nn.ModuleList(Block(d, h) for _ in range(layers))
        self.out = torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, 128))
        self.register_buffer("mask", torch.triu(torch.ones(ctx, ctx, dtype=torch.bool), 1))

    def forward(self, idx):
        t = idx.size(1)
        x = self.emb(idx) + self.pos(torch.arange(t, device=idx.device))
        for b in self.blocks:
            x = b(x, self.mask[:t, :t])
        return self.out(x)


def batch(data, bs, ctx, gen, device):
    i = torch.randint(0, len(data) - ctx - 1, (bs,), generator=gen)
    x = torch.stack([data[j:j + ctx] for j in i]).clamp(max=127)
    y = torch.stack([data[j + 1:j + ctx + 1] for j in i]).clamp(max=127)
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, data, ctx, device, n_batches=40, bs=64):
    model.eval()
    gen = torch.Generator().manual_seed(1234)          # same held-out windows for every run
    tot = 0.0
    for _ in range(n_batches):
        x, y = batch(data, bs, ctx, gen, device)
        tot += F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    model.train()
    return tot / n_batches / math.log(2)


def run(data, beta, seed, steps, device, ctx=128, bs=32, lr=1e-3):
    torch.manual_seed(seed)
    model = CharLM(ctx=ctx).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    loss_fn = POPLoss(beta=beta)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for _ in range(steps):
        x, y = batch(data["train"], bs, ctx, gen, device)
        loss = loss_fn(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    return {"beta": beta, "seed": seed, "steps": steps,
            "val_bpc": round(evaluate(model, data["val"], ctx, device), 4),
            "secs": round(time.time() - t0, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.1, 0.5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    data = corpus()
    print(f"device {device}; train {len(data['train']):,} chars, val {len(data['val']):,}", flush=True)
    rows = []
    for seed in args.seeds:
        for beta in args.betas:
            rows.append(run(data, beta, seed, args.steps, device))
            print(json.dumps(rows[-1]), flush=True)
    json.dump({"device": device, "python": sys.version.split()[0], "torch": torch.__version__,
               "rows": rows}, open(args.out, "w"), indent=1)
