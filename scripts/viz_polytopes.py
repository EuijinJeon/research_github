"""2D 모델의 폴리토프 분할을 그리고, 이론과 코드가 맞는지 정량 검산한다.

Stage 1의 관문. 실행 전에 세운 예측:

  P1. h=[64]     : 영역 경계 격자점의 거의 100%가 W1의 64개 직선 중 하나 위에 있다.
                   (1층에서 경계는 전역 초평면뿐이므로)
  P2. h=[32,32]  : 상당 비율이 첫 층 32개 직선에서 멀리 떨어져 있다.
                   (둘째 층 경계는 첫 층 셀 안에서 꺾인 조각선이므로)
  P3. h=[64]     : 격자 영역 수 <= 2081 = sum_{i<=2} C(64, i)   (Zaslavsky)
  P4. h=[32,32]  : 영역 수가 첫 층 상한 sum_{i<=2} C(32,i) = 529 를 초과한다.

P1이 깨지면 activation_pattern 또는 first_layer_lines 구현이 틀린 것이다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlpinterp.data import get_dataset  # noqa: E402
from mlpinterp.models import MLP, max_regions_one_hidden  # noqa: E402
from mlpinterp.regions import first_layer_lines, grid_patterns  # noqa: E402
from mlpinterp.train import load_model  # noqa: E402
from mlpinterp.utils import (  # noqa: E402
    FIG_DIR,
    describe_device,
    ensure_dirs,
    get_device,
    setup_matplotlib,
)

LIM = 3.5
RES = 800


# ---------------------------------------------------------------- 검산


def boundary_mask(region_map: torch.Tensor) -> np.ndarray:
    """이웃 격자점과 영역 id가 다른 지점 = 경계 격자점."""
    r = region_map.numpy()
    m = np.zeros_like(r, dtype=bool)
    m[:, :-1] |= r[:, :-1] != r[:, 1:]  # 오른쪽 이웃과 다름
    m[:-1, :] |= r[:-1, :] != r[1:, :]  # 아래쪽 이웃과 다름
    return m


def distance_to_first_layer_lines(
    model: MLP, pts: np.ndarray
) -> np.ndarray:
    """각 점에서 첫 층 직선들까지의 최소 거리. |w·x + b| / ||w||."""
    W, b = first_layer_lines(model)
    W = W.numpy()
    b = b.numpy()
    norms = np.linalg.norm(W, axis=1, keepdims=True).clip(min=1e-12)  # (h1, 1)
    # (h1, N) = (h1, 2) @ (2, N) + (h1, 1)
    d = np.abs(W @ pts.T + b[:, None]) / norms
    return d.min(axis=0)


def check_boundaries(model: MLP, xs: torch.Tensor, region_map: torch.Tensor) -> dict:
    """경계 격자점이 첫 층 직선 위에 있는 비율을 잰다.

    격자 간격의 1.5배 이내면 '직선 위'로 친다. 경계는 격자점 사이 어딘가에
    있으므로 정확히 0이 나올 수는 없고, 격자 해상도만큼의 여유가 필요하다.
    """
    spacing = float(xs[1] - xs[0])
    tol = 1.5 * spacing

    m = boundary_mask(region_map)
    iy, ix = np.nonzero(m)
    pts = np.stack([xs.numpy()[ix], xs.numpy()[iy]], axis=1)

    dist = distance_to_first_layer_lines(model, pts)
    on_line = dist <= tol
    return {
        "n_boundary_pts": len(pts),
        "grid_spacing": spacing,
        "tol": tol,
        "frac_on_first_layer_line": float(on_line.mean()),
        "median_dist": float(np.median(dist)),
        "dist": dist,
        "tol_value": tol,
    }


# ---------------------------------------------------------------- 그림


def shuffled_colors(n_regions: int, seed: int = 0) -> np.ndarray:
    """인접 영역이 다른 색으로 보이도록 색을 섞는다.

    영역 id는 패턴 정렬 순서라 공간적 인접성과 무관하다.
    그냥 연속 컬러맵을 쓰면 서로 다른 영역이 같은 색으로 붙어 보인다.
    """
    rng = np.random.default_rng(seed)
    base = plt.get_cmap("turbo")(np.linspace(0, 1, max(n_regions, 2)))
    rng.shuffle(base)
    return base


def plot_setting(ax_row, model: MLP, ds, device, title_prefix: str) -> dict:
    xs, region_map, patterns = grid_patterns(model, lim=LIM, res=RES)
    n_grid_regions = int(region_map.max().item()) + 1

    colors = shuffled_colors(n_grid_regions)
    img = colors[region_map.numpy()]
    extent = (-LIM, LIM, -LIM, LIM)

    # --- (a) 폴리토프 분할 + 데이터
    ax = ax_row[0]
    ax.imshow(img, origin="lower", extent=extent, interpolation="nearest")
    xt = ds.x_train.numpy()
    yt = ds.y_train.numpy()
    ax.scatter(xt[:, 0], xt[:, 1], c=yt, cmap="gray", s=1.5, alpha=0.45,
               edgecolors="none")
    ax.set_title(f"{title_prefix}\n영역 {n_grid_regions:,}개 (격자 {RES}x{RES})", fontsize=9)

    # --- (b) 분할 위에 첫 층 직선 겹쳐 그리기
    ax = ax_row[1]
    ax.imshow(img, origin="lower", extent=extent, interpolation="nearest", alpha=0.35)
    W, b = first_layer_lines(model)
    W, b = W.numpy(), b.numpy()
    t = np.linspace(-LIM * 2, LIM * 2, 4)
    for i in range(len(W)):
        w0, w1 = W[i]
        if abs(w1) > abs(w0):  # y = -(w0 x + b) / w1
            ax.plot(t, -(w0 * t + b[i]) / w1, color="red", lw=0.8, alpha=0.9)
        else:  # x = -(w1 y + b) / w0
            ax.plot(-(w1 * t + b[i]) / w0, t, color="red", lw=0.8, alpha=0.9)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_title(f"첫 층 직선 {len(W)}개 겹침", fontsize=9)

    # --- (c) 결정 경계
    ax = ax_row[2]
    with torch.no_grad():
        gy, gx = torch.meshgrid(xs, xs, indexing="ij")
        pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1).to(device)
        pred = torch.cat(
            [model(pts[i : i + 65536]).argmax(1).cpu() for i in range(0, len(pts), 65536)]
        ).reshape(RES, RES)
    ax.imshow(pred.numpy(), origin="lower", extent=extent, cmap="Pastel1",
              interpolation="nearest")
    ax.scatter(xt[:, 0], xt[:, 1], c=yt, cmap="tab10", s=1.5, alpha=0.7,
               edgecolors="none")
    ax.set_title("결정 경계", fontsize=9)

    for ax in ax_row:
        ax.set_xticks([])
        ax.set_yticks([])

    stats = check_boundaries(model, xs, region_map)
    stats["n_grid_regions"] = n_grid_regions
    return stats


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ensure_dirs()
    device = get_device()
    print(f"device: {describe_device(device)}   격자 {RES}x{RES}, 범위 ±{LIM}\n")

    font = setup_matplotlib()
    print(f"font: {font or '(한글 폰트 없음 — 라벨이 깨질 수 있음)'}\n")

    for dsname in ("moons", "spiral"):
        ds = get_dataset(dsname, seed=args.seed)
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        results = {}

        for row, hidden in enumerate([[64], [32, 32]]):
            model, _ = load_model(dsname, hidden, args.seed, device=device)
            tag = f"{dsname} h={hidden}"
            stats = plot_setting(axes[row], model, ds, device, tag)
            results[tuple(hidden)] = stats

        fig.suptitle(f"{dsname}: ReLU MLP의 폴리토프 분할", fontsize=12)
        fig.tight_layout()
        out = FIG_DIR / f"polytopes_{dsname}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)

        # ---- 검산 결과 출력
        print(f"=== {dsname} ===")
        s1 = results[(64,)]
        s2 = results[(32, 32)]
        ub1 = max_regions_one_hidden(64, 2)
        ub_first = max_regions_one_hidden(32, 2)

        print(f"  격자 간격 {s1['grid_spacing']:.5f}, 판정 허용치 {s1['tol']:.5f}")
        print()
        print(f"  P1  h=[64]    경계점이 첫 층 직선 위에 있는 비율 : "
              f"{s1['frac_on_first_layer_line'] * 100:6.2f} %   "
              f"{'PASS' if s1['frac_on_first_layer_line'] > 0.99 else 'FAIL'}")
        print(f"  P2  h=[32,32] 같은 비율                          : "
              f"{s2['frac_on_first_layer_line'] * 100:6.2f} %   "
              f"{'PASS' if s2['frac_on_first_layer_line'] < 0.9 else 'FAIL'}"
              f"   (낮을수록 둘째 층 경계가 많다는 뜻)")
        print(f"  P3  h=[64]    영역 수 {s1['n_grid_regions']:,} <= {ub1:,}          "
              f"{'PASS' if s1['n_grid_regions'] <= ub1 else 'FAIL'}")
        print(f"  P4  h=[32,32] 영역 수 {s2['n_grid_regions']:,} >  {ub_first:,} (첫 층 상한)  "
              f"{'PASS' if s2['n_grid_regions'] > ub_first else 'FAIL'}")
        print()
        print(f"  경계점까지의 중앙 거리:  h=[64] {s1['median_dist']:.5f}"
              f"   |  h=[32,32] {s2['median_dist']:.5f}")
        print(f"  saved -> {out}")
        print()


if __name__ == "__main__":
    main()
