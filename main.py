import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def quantize_9bit(tensor):
    tensor_scaled = torch.tanh(tensor)
    quantized = torch.round(tensor_scaled * 255).clamp(-256, 255)
    return quantized / 255.0


class AOC(nn.Module):
    def __init__(self, in_dim=28 * 28, hidden=256, out_dim=10, tol=1e-3, max_iters=200):
        super().__init__()
        self.W_IP = nn.Linear(in_dim, hidden, bias=True)
        self.W_OP = nn.Linear(hidden, out_dim, bias=True)
        # core AOC with 9-bit quantization
        self._W_raw = nn.Parameter(torch.empty(hidden, hidden))
        self._b_raw = nn.Parameter(torch.zeros(hidden))

        self._alpha_raw = nn.Parameter(torch.tensor(0.0))
        self._beta_raw = nn.Parameter(torch.tensor(0.0))

        self.tol = tol
        self.max_iters = max_iters

        self.initialize()

    @property
    def W(self):
        return quantize_9bit(self._W_raw)

    @property
    def b(self):
        return quantize_9bit(self._b_raw)

    @property
    def alpha(self):
        return torch.sigmoid(self._alpha_raw)

    @property
    def beta(self):
        return torch.sigmoid(self._beta_raw)

    def initialize(self):
        nn.init.kaiming_uniform_(self._W_raw, a=math.sqrt(5))
        with torch.no_grad():
            self._W_raw.mul_(0.1)
        for layer in (self.W_IP, self.W_OP):
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            if layer.bias is not None:
                fan_in = layer.weight.size(1)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(layer.bias, -bound, bound)
        nn.init.zeros_(self._b_raw)

    def forward(self, x, return_iters=False):
        B = x.size(0)
        x_proj = self.W_IP(x)
        s = self.b + x_proj
        b = self.b.unsqueeze(0)
        alpha, beta = self.alpha, self.beta

        iters = torch.zeros(B, device=x.device, dtype=torch.int32)
        converged = torch.zeros(B, device=x.device, dtype=torch.bool)

        for t in range(self.max_iters):
            if converged.all():
                break

            s_old = s
            s_next = alpha * s + beta * (torch.tanh(s) @ self.W.t()) + b + x_proj
            diff = (s_next - s_old).abs().amax(dim=1)
            newly_conv = diff < self.tol
            iters[~converged & newly_conv] = t + 1
            converged = converged | newly_conv
            s = s_next

        logits = self.W_OP(s)

        if return_iters:
            return logits, iters
        return logits


@dataclass
class Params:
    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: int = 256
    max_iters: int = 200
    tol: float = 1e-3
    seed: int = 42
    data_dir: str = "./data"


def flatten_tensor(t):
    return t.view(-1)


def get_loaders(hp: Params):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(flatten_tensor)])
    train_ds = datasets.MNIST(hp.data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(hp.data_dir, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=hp.batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader


def train(model, loader, opt, device):
    model.train()
    total, correct, total_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        total += x.size(0)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    iter_sum, iter_count = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, iters = model(x, return_iters=True)
        loss = F.cross_entropy(logits, y)

        total += x.size(0)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        iter_sum += iters.sum().item()
        iter_count += x.size(0)
    avg_iters = iter_sum / max(1, iter_count)
    return total_loss / total, correct / total, avg_iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--max_iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--save", type=str, default="model.pth")
    args = p.parse_args()

    hp = Params(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, hidden=args.hidden, tol=args.tol, max_iters=args.max_iters, seed=args.seed, data_dir=args.data_dir)
    torch.manual_seed(hp.seed), torch.cuda.manual_seed_all(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders(hp)
    model = AOC(in_dim=28 * 28, hidden=hp.hidden, out_dim=10, tol=hp.tol, max_iters=hp.max_iters).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)

    best_acc = 0.0
    for epoch in range(1, hp.epochs + 1):
        tr_loss, tr_acc = train(model, train_loader, opt, device)
        te_loss, te_acc, te_iters = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc * 100:.2f}% | test  loss {te_loss:.4f} acc {te_acc * 100:.2f}% | avg iters {te_iters:.1f} | alpha {model.alpha.item():.3f} beta {model.beta.item():.3f}")
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), args.save)

    print(f"Best test accuracy: {best_acc * 100:.2f}%")
    print(f"Saved best model to: {args.save}")


if __name__ == "__main__":
    main()
