"""데이터셋 — 2D 합성(two-moons / spiral)과 MNIST를 같은 인터페이스로.

Stage 0에서 확정한 세팅:
  A. 2D 합성 : 입력 2차원. 폴리토프를 평면에 직접 그려서 검산할 수 있는 유일한 세팅.
  B. MNIST   : 입력 784차원. A_r의 각 행이 28x28 이미지로 시각화된다.

입력은 항상 표준화한다. 표준화는 입력에 대한 affine 변환이므로 폴리토프 구조를
바꾸지 않고 좌표계만 바꾼다 — 그림을 그리기 좋은 범위로 옮기는 것이 목적이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .utils import DATA_DIR


@dataclass
class Dataset:
    """모든 데이터셋의 공통 형태. 텐서는 CPU float32로 유지한다.

    device 이동은 학습 루프에서 한다 — 데이터를 미리 GPU에 올려두면
    맥/리눅스 사이에서 코드가 갈라진다.
    """

    name: str
    x_train: torch.Tensor  # (N, D) float32
    y_train: torch.Tensor  # (N,)   int64
    x_test: torch.Tensor
    y_test: torch.Tensor
    input_dim: int
    n_classes: int
    input_shape: tuple[int, ...]  # 시각화용 원래 shape. 2D는 (2,), MNIST는 (28, 28)

    def __repr__(self) -> str:
        return (
            f"Dataset({self.name}, train={tuple(self.x_train.shape)}, "
            f"test={tuple(self.x_test.shape)}, classes={self.n_classes})"
        )


# ---------------------------------------------------------------- 표준화


def _standardize(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """훈련셋 통계로만 표준화한다 (테스트셋 통계를 쓰면 누수).

    std가 0인 차원(MNIST 가장자리의 항상-0 픽셀)은 1로 두어 0/0을 막는다.
    그 차원은 어차피 상수라 W1의 해당 열이 하는 일은 bias에 흡수된다.
    """
    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (x_train - mu) / sd, (x_test - mu) / sd


# ---------------------------------------------------------------- 2D 합성


def make_moons(n: int = 4000, noise: float = 0.15, seed: int = 0) -> Dataset:
    """two-moons: 초승달 두 개가 맞물린 2클래스 문제.

    경계가 매끄러운 곡선 하나뿐이라 적은 수의 affine 영역으로도 근사된다.
    → ε-path가 가파를 것으로 예상되는 쪽.
    """
    from sklearn.datasets import make_moons as _mm

    x, y = _mm(n_samples=n, noise=noise, random_state=seed)
    return _finish_2d("moons", x, y, n_classes=2, seed=seed)


def make_spiral(
    n: int = 4000, n_classes: int = 3, noise: float = 0.06, seed: int = 0
) -> Dataset:
    """spiral: 나선 팔 n_classes개가 서로 감긴 문제.

    경계가 여러 번 감기므로 많은 affine 영역을 강제로 요구한다.
    → ε-path가 완만할 것으로 예상되는 쪽. moons와의 대조가 목적이다.
    """
    rng = np.random.default_rng(seed)
    per = n // n_classes
    xs, ys = [], []
    for c in range(n_classes):
        # r을 sqrt로 뽑아 면적당 밀도를 균일하게 (안쪽에 점이 몰리는 것 방지)
        r = np.sqrt(rng.uniform(0.03, 1.0, per))
        theta = r * 3.6 * np.pi + c * (2 * np.pi / n_classes)
        theta = theta + rng.normal(0.0, noise, per)
        xs.append(np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1))
        ys.append(np.full(per, c))
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return _finish_2d("spiral", x, y, n_classes=n_classes, seed=seed)


def _finish_2d(
    name: str, x: np.ndarray, y: np.ndarray, n_classes: int, seed: int
) -> Dataset:
    """2D 데이터 공통 마무리: 셔플 → 80/20 분할 → 표준화 → 텐서화."""
    rng = np.random.default_rng(seed + 12345)
    perm = rng.permutation(len(x))
    x, y = x[perm], y[perm]

    n_tr = int(0.8 * len(x))
    x_tr, x_te = x[:n_tr], x[n_tr:]
    y_tr, y_te = y[:n_tr], y[n_tr:]
    x_tr, x_te = _standardize(x_tr, x_te)

    return Dataset(
        name=name,
        x_train=torch.tensor(x_tr, dtype=torch.float32),
        y_train=torch.tensor(y_tr, dtype=torch.long),
        x_test=torch.tensor(x_te, dtype=torch.float32),
        y_test=torch.tensor(y_te, dtype=torch.long),
        input_dim=2,
        n_classes=n_classes,
        input_shape=(2,),
    )


# ---------------------------------------------------------------- MNIST


def load_mnist(classes: tuple[int, ...] | None = None) -> Dataset:
    """MNIST를 784차원 벡터로 편다.

    classes를 주면 그 클래스만 남기고 라벨을 0..k-1로 재매핑한다.
    (예: classes=(3, 5) → 이진 분류. A_r이 1x784가 되어 영역당 템플릿 이미지 한 장.
     Stage 1이 막혔을 때 후퇴할 디버깅 세팅.)
    """
    from torchvision import datasets

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tr = datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
    te = datasets.MNIST(root=str(DATA_DIR), train=False, download=True)

    # ToTensor 대신 직접 편다 — transform 파이프라인을 거치지 않는 편이
    # "무엇이 입력인가"가 명확하고 재현이 단순하다.
    x_tr = tr.data.reshape(len(tr.data), -1).numpy().astype(np.float32) / 255.0
    x_te = te.data.reshape(len(te.data), -1).numpy().astype(np.float32) / 255.0
    y_tr = tr.targets.numpy()
    y_te = te.targets.numpy()

    name = "mnist"
    if classes is not None:
        remap = {c: i for i, c in enumerate(classes)}
        m_tr = np.isin(y_tr, classes)
        m_te = np.isin(y_te, classes)
        x_tr, y_tr = x_tr[m_tr], np.array([remap[v] for v in y_tr[m_tr]])
        x_te, y_te = x_te[m_te], np.array([remap[v] for v in y_te[m_te]])
        name = "mnist" + "".join(str(c) for c in classes)
        n_classes = len(classes)
    else:
        n_classes = 10

    x_tr, x_te = _standardize(x_tr, x_te)

    return Dataset(
        name=name,
        x_train=torch.tensor(x_tr, dtype=torch.float32),
        y_train=torch.tensor(y_tr, dtype=torch.long),
        x_test=torch.tensor(x_te, dtype=torch.float32),
        y_test=torch.tensor(y_te, dtype=torch.long),
        input_dim=784,
        n_classes=n_classes,
        input_shape=(28, 28),
    )


# ---------------------------------------------------------------- 디스패처

_2D_NAMES = {"moons", "spiral"}


def get_dataset(name: str, seed: int = 0, **kw) -> Dataset:
    """이름으로 데이터셋을 얻는다. 스크립트가 --dataset 하나만 받게 하기 위함."""
    if name == "moons":
        return make_moons(seed=seed, **kw)
    if name == "spiral":
        return make_spiral(seed=seed, **kw)
    if name == "mnist":
        return load_mnist()
    if name.startswith("mnist") and name[5:].isdigit():
        return load_mnist(classes=tuple(int(c) for c in name[5:]))
    raise ValueError(f"unknown dataset: {name!r} (known: moons, spiral, mnist, mnist35)")
