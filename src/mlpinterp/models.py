"""ReLU MLP와 그 piecewise-linear 구조 추출.

핵심 사실 (docs/concepts.md Part 1):
    활성 패턴 r을 고정하면 ReLU는 대각행렬 D(r)이 되고,
    신경망 전체가 하나의 affine 맵 x -> A_r·x + b_r 이 된다.

hidden 층이 L개일 때의 누적 규칙:
    A <- W1,  b <- b1
    for l in 1..L:
        A <- D_l·A          b <- D_l·b              (게이트: 꺼진 유닛의 행을 0으로)
        A <- W_{l+1}·A      b <- W_{l+1}·b + b_{l+1}

게이트가 하는 일은 "A의 행 중 꺼진 유닛에 해당하는 행을 통째로 0으로 만드는 것"이다.
ReLU는 정보를 자르는 게 아니라 선형맵의 행을 골라내는 스위치다.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """hidden 층 수를 인자로 받는 ReLU MLP.

    hidden=[64]      -> 1 hidden layer (영역 경계가 전역 초평면 = 직선)
    hidden=[64, 64]  -> 2 hidden layers (영역 경계가 꺾인다)

    이 둘의 대조가 Stage 0에서 정한 '깊이 축' 일반성 검증이다.
    """

    def __init__(self, in_dim: int, hidden: list[int], out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = list(hidden)
        self.out_dim = out_dim

        dims = [in_dim, *hidden, out_dim]
        # nn.Linear를 그대로 쓴다. 유효 affine 맵 계산에서 .weight/.bias를 직접 읽는다.
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )

    # ------------------------------------------------------------ 기본 forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)

    # ------------------------------------------------------------ 활성 패턴

    @property
    def n_hidden_units(self) -> int:
        """전체 ReLU 유닛 수. 활성 패턴의 비트 수이자 초평면 개수."""
        return sum(self.hidden)

    def pre_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        """각 hidden 층의 pre-activation z^l 을 순서대로 반환. (N, h_l)

        부호가 곧 활성 패턴이므로, 경계까지의 거리를 재고 싶을 때도 이 값을 쓴다.
        """
        zs = []
        h = x
        for layer in self.layers[:-1]:
            z = layer(h)
            zs.append(z)
            h = torch.relu(z)
        return zs

    def activation_pattern(self, x: torch.Tensor) -> torch.Tensor:
        """활성 패턴 r(x). (N, sum(hidden)) uint8, 층 순서로 이어붙인 비트.

        r_i = 1[z_i > 0]. 등호에서 0으로 두는 것이 ReLU와 정확히 일치한다
        (z=0이면 ReLU도 0, 게이트도 0 -> 곱해서 0으로 동일).
        """
        zs = self.pre_activations(x)
        return torch.cat([(z > 0).to(torch.uint8) for z in zs], dim=1)

    # ------------------------------------------------------------ 유효 affine 맵

    def effective_affine(
        self, pattern: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """활성 패턴들로부터 유효 affine 맵 (A_r, b_r)을 배치로 계산.

        Args:
            pattern: (N, sum(hidden)). 0/1. dtype은 무엇이든 float으로 캐스팅한다.

        Returns:
            A: (N, out_dim, in_dim)
            b: (N, out_dim)

        메모리 주의: MNIST(out=10, in=784)에서 N=60000이면 A만 1.9 GB.
        호출자가 청크로 끊어 부르는 것을 전제로 한다 (regions.py가 그렇게 한다).
        """
        device = self.layers[0].weight.device
        dtype = self.layers[0].weight.dtype
        pattern = pattern.to(device=device, dtype=dtype)
        n = pattern.shape[0]

        # 층별 게이트로 쪼갠다
        gates, off = [], 0
        for h in self.hidden:
            gates.append(pattern[:, off : off + h])
            off += h

        # A <- W1, b <- b1 (배치 차원 붙여서 시작)
        first = self.layers[0]
        A = first.weight.unsqueeze(0).expand(n, -1, -1)  # (N, h1, in)
        b = first.bias.unsqueeze(0).expand(n, -1)  # (N, h1)

        for l, gate in enumerate(gates):
            # 게이트: 꺼진 유닛의 행을 0으로
            A = A * gate.unsqueeze(-1)
            b = b * gate
            # 다음 층 가중치를 곱한다
            nxt = self.layers[l + 1]
            A = torch.einsum("oh,nhi->noi", nxt.weight, A)
            b = torch.einsum("oh,nh->no", nxt.weight, b) + nxt.bias

        return A, b

    def effective_affine_at(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """입력 x에서의 유효 affine 맵. 패턴 추출 + affine 계산을 합친 편의 함수."""
        return self.effective_affine(self.activation_pattern(x))

    # ------------------------------------------------------------ 검산

    @torch.no_grad()
    def verify_affine(
        self, x: torch.Tensor, atol: float = 1e-4, rtol: float = 1e-4
    ) -> dict:
        """A_r·x + b_r 이 model(x)와 같은지 확인한다.

        Stage 1 전체가 이 항등식 위에 서 있다. 여기서 틀리면 이후 클러스터링,
        ε-path, Stage 4 연결이 전부 무의미하므로 반드시 통과해야 한다.

        Returns: 최대/평균 오차와 통과 여부가 담긴 dict.
        """
        direct = self.forward(x)
        A, b = self.effective_affine_at(x)
        # (N, out, in) @ (N, in, 1) -> (N, out)
        recon = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1) + b

        err = (direct - recon).abs()
        scale = direct.abs().max().clamp(min=1e-12)
        return {
            "max_abs_err": err.max().item(),
            "mean_abs_err": err.mean().item(),
            "max_rel_err": (err.max() / scale).item(),
            "logit_scale": scale.item(),
            "passed": bool(torch.allclose(direct, recon, atol=atol, rtol=rtol)),
        }


# ---------------------------------------------------------------- 이론값


def max_regions_one_hidden(h: int, d: int) -> int:
    """1 hidden layer ReLU MLP의 영역 개수 상한 (Zaslavsky).

        R(h, d) <= sum_{i=0}^{d} C(h, i)

    d=2, h=64  -> 1 + 64 + 2016 = 2081        (완전 열거 가능)
    d=784, h=128 -> h <= d 이므로 2^128       (원천 불가능)

    2D 검산에서 "관측된 영역 수 <= 이 값"을 반드시 확인한다.
    """
    from math import comb

    return sum(comb(h, i) for i in range(min(d, h) + 1))
