"""학습 루프와 체크포인트 입출력.

Stage 1에서 wandb는 쓰지 않는다. 2D는 수 초, MNIST MLP도 1분 안쪽이라
추적 오버헤드가 이득보다 크다. 대신 학습 곡선을 체크포인트에 함께 저장한다.
학습 시간 축이 중요해지는 Stage 3(grokking)에서 wandb offline을 켠다.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .data import Dataset
from .models import MLP
from .utils import CKPT_DIR, get_device, load_checkpoint, save_checkpoint, set_seed


def run_name(dataset: str, hidden: list[int], seed: int) -> str:
    """체크포인트 이름. 세팅을 파일명만 보고 알 수 있게 한다."""
    return f"{dataset}_h{'x'.join(str(h) for h in hidden)}_s{seed}"


def ckpt_path(dataset: str, hidden: list[int], seed: int) -> Path:
    return CKPT_DIR / f"{run_name(dataset, hidden, seed)}.pt"


@torch.no_grad()
def evaluate(
    model: MLP, x: torch.Tensor, y: torch.Tensor, batch: int = 8192
) -> tuple[float, float]:
    """정확도와 평균 CE loss. 큰 텐서를 대비해 청크로 처리."""
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for i in range(0, len(x), batch):
        xb, yb = x[i : i + batch], y[i : i + batch]
        logits = model(xb)
        loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += len(xb)
    return correct / total, loss_sum / total


def train_model(
    ds: Dataset,
    hidden: list[int],
    seed: int = 0,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    log_every: int = 20,
    verbose: bool = True,
) -> tuple[MLP, dict]:
    """표준 Adam + cross-entropy 학습.

    weight_decay 기본값 0: Stage 1의 목적은 최고 성능이 아니라
    "전형적으로 학습된 MLP"를 얻는 것이다. 정규화를 걸면 폴리토프 구조가
    인위적으로 단순해져서 ε-path가 낙관적으로 나올 수 있다.
    정규화 효과 자체는 나중에 별도 축으로 볼 수 있다.
    """
    set_seed(seed)
    device = device or get_device()

    model = MLP(ds.input_dim, hidden, ds.n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_tr = ds.x_train.to(device)
    y_tr = ds.y_train.to(device)
    x_te = ds.x_test.to(device)
    y_te = ds.y_test.to(device)

    n = len(x_tr)
    history = {"epoch": [], "train_acc": [], "train_loss": [], "test_acc": [], "test_loss": []}
    g = torch.Generator(device="cpu").manual_seed(seed)

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            loss = F.cross_entropy(model(x_tr[idx]), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if ep % log_every == 0 or ep == epochs or ep == 1:
            tr_acc, tr_loss = evaluate(model, x_tr, y_tr)
            te_acc, te_loss = evaluate(model, x_te, y_te)
            history["epoch"].append(ep)
            history["train_acc"].append(tr_acc)
            history["train_loss"].append(tr_loss)
            history["test_acc"].append(te_acc)
            history["test_loss"].append(te_loss)
            if verbose:
                print(
                    f"  ep {ep:4d} | train {tr_acc:.4f} ({tr_loss:.4f})"
                    f" | test {te_acc:.4f} ({te_loss:.4f})"
                )

    return model, history


def save_model(
    model: MLP, ds: Dataset, seed: int, history: dict, extra: dict | None = None
) -> Path:
    """모델 + 복원에 필요한 config를 함께 저장.

    config를 같이 넣는 이유: 뒷단계가 hidden 구조를 모른 채 체크포인트만 보고
    모델을 복원할 수 있어야 앞단계 재실행을 피할 수 있다 (요구사항 6).
    """
    path = ckpt_path(ds.name, model.hidden, seed)
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "in_dim": model.in_dim,
            "hidden": model.hidden,
            "out_dim": model.out_dim,
            "dataset": ds.name,
            "seed": seed,
        },
        "history": history,
    }
    if extra:
        payload.update(extra)
    return save_checkpoint(path, payload)


def load_model(
    dataset: str, hidden: list[int], seed: int, device: torch.device | str = "cpu"
) -> tuple[MLP, dict]:
    """체크포인트에서 모델을 복원. 항상 map_location을 명시한다."""
    path = ckpt_path(dataset, hidden, seed)
    if not path.exists():
        raise FileNotFoundError(f"체크포인트 없음: {path}\n먼저 scripts/train.py를 실행하세요.")
    payload = load_checkpoint(path, device=device)
    cfg = payload["config"]
    model = MLP(cfg["in_dim"], cfg["hidden"], cfg["out_dim"])
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload
