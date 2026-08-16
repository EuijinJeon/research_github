"""활성 패턴에서 영역(region)과 유효 affine 맵을 뽑아낸다.

저장 정책 — 패턴이 정본(canonical), A_r은 파생물이다.
    MNIST에서 고유 영역 6만 개면 [A_r | b_r]은 60000 x 10 x 785 x 4B ~= 1.9 GB.
    반면 활성 패턴은 60000 x 128 uint8 = 7.7 MB.
    패턴만 있으면 A_r은 모델에서 언제든 정확히 재계산되므로 패턴만 저장한다.
    (요구사항 6은 '비싼 계산'인 패턴 추출을 저장하는 것으로 충족된다.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .models import MLP
from .utils import REGION_DIR, load_checkpoint, save_checkpoint


@dataclass
class RegionSet:
    """입력 집합에서 관측된 영역들.

    patterns[j] 가 j번째 고유 활성 패턴이고,
    region_of[i] = j 는 i번째 입력이 j번 영역에 속한다는 뜻이다.
    이 region_of 가 docs/concepts.md 0.3의 추상화 함수 α 의 1단계다
    (입력 -> 내부 상태 -> 영역 id).
    """

    dataset: str
    hidden: list[int]
    seed: int
    split: str
    patterns: torch.Tensor  # (R, H) uint8 — 고유 패턴
    region_of: torch.Tensor  # (N,) int64 — 입력 -> 영역 인덱스
    counts: torch.Tensor  # (R,) int64 — 영역별 입력 개수

    @property
    def n_regions(self) -> int:
        return self.patterns.shape[0]

    @property
    def n_inputs(self) -> int:
        return self.region_of.shape[0]

    @property
    def n_units(self) -> int:
        return self.patterns.shape[1]

    def __repr__(self) -> str:
        return (
            f"RegionSet({self.dataset} h={self.hidden} {self.split}: "
            f"{self.n_regions:,} regions / {self.n_inputs:,} inputs, "
            f"{self.n_regions / self.n_inputs:.3f} regions per input)"
        )


# ---------------------------------------------------------------- 추출


@torch.no_grad()
def extract_patterns(model: MLP, x: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """전체 입력의 활성 패턴을 (N, H) uint8로. 청크로 끊어 메모리를 통제."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(x), chunk):
        xb = x[i : i + chunk].to(device)
        out.append(model.activation_pattern(xb).cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def extract_regions(
    model: MLP,
    x: torch.Tensor,
    dataset: str,
    seed: int,
    split: str,
    chunk: int = 8192,
) -> RegionSet:
    """입력들을 고유 활성 패턴으로 묶는다.

    torch.unique(dim=0)는 정렬 기반이라 (60000, 128) uint8 정도는 가볍게 처리한다.
    return_inverse가 곧 '입력 -> 영역' 매핑이다.
    """
    patterns = extract_patterns(model, x, chunk=chunk)
    uniq, inverse, counts = torch.unique(
        patterns, dim=0, return_inverse=True, return_counts=True
    )
    return RegionSet(
        dataset=dataset,
        hidden=list(model.hidden),
        seed=seed,
        split=split,
        patterns=uniq,
        region_of=inverse.to(torch.int64),
        counts=counts.to(torch.int64),
    )


# ---------------------------------------------------------------- affine 맵


@torch.no_grad()
def affine_of_patterns(
    model: MLP, patterns: torch.Tensor, chunk: int = 256, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """패턴들의 유효 affine 맵 (A, b)를 청크로 계산해 CPU로 모은다.

    chunk 기본값 256이 작아 보이지만 MNIST 기준
    256 x 128 x 784 x 4B = 103 MB 의 중간 텐서가 잡히므로 적절하다.
    """
    model.eval()
    As, bs = [], []
    for i in range(0, len(patterns), chunk):
        A, b = model.effective_affine(patterns[i : i + chunk])
        As.append(A.to(dtype).cpu())
        bs.append(b.to(dtype).cpu())
    return torch.cat(As, dim=0), torch.cat(bs, dim=0)


def augmented_matrix(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[A_r | b_r] 증강 행렬. (R, out, in) + (R, out) -> (R, out, in+1)

    docs/concepts.md 1.2의 경고: A_r만으로 클러스터링하면
    '기울기는 같은데 절편이 다른' 서로 다른 함수를 하나로 합쳐버린다.
    거리 계산은 반드시 이 증강 행렬 위에서 한다.
    """
    return torch.cat([A, b.unsqueeze(-1)], dim=-1)


def flatten_augmented(M: torch.Tensor) -> torch.Tensor:
    """(R, out, in+1) -> (R, out*(in+1)). Frobenius 거리를 유클리드 거리로 쓰기 위함."""
    return M.reshape(M.shape[0], -1)


# ---------------------------------------------------------------- 저장/로드


def region_path(dataset: str, hidden: list[int], seed: int, split: str) -> Path:
    h = "x".join(str(v) for v in hidden)
    return REGION_DIR / f"{dataset}_h{h}_s{seed}_{split}.pt"


def save_regions(rs: RegionSet) -> Path:
    path = region_path(rs.dataset, rs.hidden, rs.seed, rs.split)
    return save_checkpoint(
        path,
        {
            "dataset": rs.dataset,
            "hidden": rs.hidden,
            "seed": rs.seed,
            "split": rs.split,
            "patterns": rs.patterns,
            "region_of": rs.region_of,
            "counts": rs.counts,
        },
    )


def load_regions(dataset: str, hidden: list[int], seed: int, split: str) -> RegionSet:
    path = region_path(dataset, hidden, seed, split)
    if not path.exists():
        raise FileNotFoundError(f"영역 파일 없음: {path}\n먼저 scripts/extract_regions.py를 실행하세요.")
    d = load_checkpoint(path)
    return RegionSet(**d)


# ---------------------------------------------------------------- 볼록성 검증


@torch.no_grad()
def segment_scan(
    model: MLP, xa: torch.Tensor, xb: torch.Tensor, steps: int = 2000
) -> list[dict]:
    """선분 위를 걸으며 활성 패턴을 기록하고 볼록성을 검증한다.

    원리: 볼록 집합과 직선의 교집합은 항상 하나의 연결된 구간이다.
    따라서 선분 위에서 어떤 활성 패턴도 두 번 나타나서는 안 된다.
    한 번 떠난 영역으로 되돌아왔다면 그 영역은 볼록이 아니다.

    이 테스트는 입력 차원에 무관하다 — 폴리토프를 그릴 수 없는 MNIST의
    784차원에서도 그대로 돈다.

    Args:
        xa, xb: (S, D) 선분 S개의 양 끝점.
        steps:  선분당 샘플 개수. 촘촘할수록 얇은 영역을 놓칠 확률이 준다.

    Returns:
        선분마다 {n_runs, n_unique, violations, ...} 딕셔너리.
        n_runs   = 선분이 지나간 영역의 개수 (연속 구간 수)
        n_unique = 그 중 서로 다른 패턴의 개수
        볼록이면 두 값이 같아야 한다.
    """
    model.eval()
    device = next(model.parameters()).device
    t = torch.linspace(0.0, 1.0, steps, device=device).unsqueeze(1)  # (T, 1)

    out = []
    for s in range(len(xa)):
        a = xa[s].to(device).unsqueeze(0)
        b = xb[s].to(device).unsqueeze(0)
        pts = a + t * (b - a)  # (T, D)
        pat = model.activation_pattern(pts)  # (T, H)

        # 연속 구간(run) 나누기: 이전 샘플과 패턴이 달라지는 지점이 경계
        changed = (pat[1:] != pat[:-1]).any(dim=1)
        run_start = torch.cat(
            [torch.ones(1, dtype=torch.bool, device=device), changed]
        )
        reps = pat[run_start]  # 각 run의 대표 패턴 (n_runs, H)
        n_runs = int(run_start.sum().item())
        n_unique = int(torch.unique(reps, dim=0).shape[0])

        out.append(
            {
                "n_runs": n_runs,
                "n_unique": n_unique,
                "violations": n_runs - n_unique,  # 볼록이면 0
            }
        )
    return out


def random_segment_endpoints(
    x: torch.Tensor, n_segments: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """데이터 포인트 쌍을 무작위로 골라 선분의 양 끝점으로 쓴다.

    데이터 사이를 잇는 선분을 쓰는 이유: 무작위 방향으로 쏘면 데이터가 없는
    영역만 지나갈 수 있다. 실제로 모델이 쓰는 영역들을 통과시키고 싶다.
    """
    g = torch.Generator().manual_seed(seed)
    ia = torch.randint(0, len(x), (n_segments,), generator=g)
    ib = torch.randint(0, len(x), (n_segments,), generator=g)
    return x[ia], x[ib]


# ---------------------------------------------------------------- 2D 전용


@torch.no_grad()
def grid_patterns(
    model: MLP,
    lim: float = 3.5,
    res: int = 600,
    chunk: int = 65536,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """2D 입력 공간을 격자로 훑어 영역 지도를 만든다.

    데이터 포인트에서만 영역을 보면 '데이터가 지나가는 영역'만 보인다.
    격자로 훑으면 그 상자 안의 (면적이 무시할 수 없는) 영역을 사실상 전부 본다.
    2D에서만 가능한 일이고, 그래서 2D가 검산 세팅인 것이다.

    Returns:
        xs: (res,) 격자 좌표
        region_map: (res, res) int64 — 각 격자점의 영역 id
        patterns: (R, H) uint8 — 고유 패턴
    """
    assert model.in_dim == 2, "2D 모델에서만 쓸 수 있습니다"
    device = next(model.parameters()).device

    xs = torch.linspace(-lim, lim, res)
    gy, gx = torch.meshgrid(xs, xs, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

    pat = []
    for i in range(0, len(pts), chunk):
        pat.append(model.activation_pattern(pts[i : i + chunk].to(device)).cpu())
    pat = torch.cat(pat, dim=0)

    uniq, inverse = torch.unique(pat, dim=0, return_inverse=True)
    return xs, inverse.reshape(res, res), uniq


def first_layer_lines(model: MLP) -> tuple[torch.Tensor, torch.Tensor]:
    """첫 층이 정의하는 직선들. w·x + b = 0.

    hidden 1층이면 이 직선들이 영역 경계 '전부'와 정확히 일치해야 한다.
    hidden 2층이면 이 직선들은 경계의 '일부'일 뿐이고,
    나머지 경계는 이 직선들 위에서 꺾인다. 그 차이를 그림으로 확인한다.
    """
    W = model.layers[0].weight.detach().cpu()  # (h1, 2)
    b = model.layers[0].bias.detach().cpu()  # (h1,)
    return W, b
