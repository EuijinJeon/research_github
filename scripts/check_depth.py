"""깊이가 영역 구조에 무엇을 하는지 검증한다.

실행 전 예측:

  P5. 모든 세팅(2D 1층/2층, MNIST 1층/2층)에서 선분당 볼록성 위반 = 0.
      영역은 깊이와 무관하게 볼록 다면체이기 때문.
      (검증법: 볼록 집합 ∩ 직선 = 하나의 구간. 선분 위에서 어떤 패턴도 두 번 나오면 안 됨)

  P6. h=[32,32]의 둘째 층 유닛의 z=0 선은 첫 층 직선과 만나는 지점에서만 꺾인다.
      꺾임의 원인이 첫 층 게이트 전환이므로, 꺾이는 '위치'까지 예측된다.

부수적으로 얻는 값: 데이터 포인트 두 개를 잇는 직선이 몇 개의 영역을 지나가는가.
이건 MNIST처럼 그림을 그릴 수 없는 곳에서 영역 밀도를 재는 방법이다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlpinterp.data import get_dataset  # noqa: E402
from mlpinterp.regions import (  # noqa: E402
    first_layer_lines,
    random_segment_endpoints,
    segment_scan,
)
from mlpinterp.train import load_model  # noqa: E402
from mlpinterp.utils import (  # noqa: E402
    FIG_DIR,
    describe_device,
    ensure_dirs,
    get_device,
    setup_matplotlib,
)

SETTINGS = [
    ("moons", [64]),
    ("moons", [32, 32]),
    ("spiral", [64]),
    ("spiral", [32, 32]),
    ("mnist", [128]),
    ("mnist", [64, 64]),
]

LIM = 3.5
RES = 700


# ---------------------------------------------------------------- P5


def run_convexity(seed: int, n_segments: int, steps: int, device) -> None:
    print("=" * 78)
    print("P5  볼록성: 선분 위에서 같은 패턴이 두 번 나타나면 위반")
    print("=" * 78)
    print(f"{'setting':22s} {'선분':>5s} {'평균 통과영역':>13s} {'최대':>6s} {'총 위반':>8s}  판정")
    print("-" * 78)

    for name, hidden in SETTINGS:
        model, _ = load_model(name, hidden, seed, device=device)
        ds = get_dataset(name, seed=seed)
        xa, xb = random_segment_endpoints(ds.x_test, n_segments, seed=seed)
        res = segment_scan(model, xa, xb, steps=steps)

        runs = np.array([r["n_runs"] for r in res])
        viol = sum(r["violations"] for r in res)
        tag = f"{name} h={'x'.join(map(str, hidden))}"
        verdict = "PASS" if viol == 0 else "FAIL"
        print(
            f"{tag:22s} {len(res):5d} {runs.mean():13.1f} {runs.max():6d} "
            f"{viol:8d}  {verdict}"
        )
    print()


# ---------------------------------------------------------------- P6


@torch.no_grad()
def zero_set_figure(dsname: str, seed: int, device) -> Path:
    """유닛 하나의 z=0 집합을 그린다. 첫 층은 직선, 둘째 층은 꺾인 선."""
    xs = torch.linspace(-LIM, LIM, RES)
    gy, gx = torch.meshgrid(xs, xs, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1).to(device)
    X, Y = gx.numpy(), gy.numpy()

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))

    for row, hidden in enumerate([[64], [32, 32]]):
        model, _ = load_model(dsname, hidden, seed, device=device)
        zs = [
            torch.cat(
                [model.pre_activations(pts[i : i + 65536])[l] for i in range(0, len(pts), 65536)]
            )
            for l in range(len(hidden))
        ]
        W1, b1 = first_layer_lines(model)
        W1, b1 = W1.numpy(), b1.numpy()
        t = np.linspace(-LIM * 2, LIM * 2, 4)

        # 어느 층의 어느 유닛을 보여줄지 고른다.
        # 상자 안에서 z가 부호를 바꾸는 유닛만 (영점 집합이 화면에 있는 유닛)
        picks = []
        for layer in range(len(hidden)):
            z = zs[layer]
            valid = ((z.max(0).values > 0) & (z.min(0).values < 0)).nonzero().flatten()
            n_take = 4 if len(hidden) == 1 else 2
            picks += [(layer, int(u)) for u in valid[:n_take]]

        for col, (layer, unit) in enumerate(picks[:4]):
            ax = axes[row, col]
            # 첫 층 직선들을 회색으로 깔아둔다 — 꺾이는 위치의 기준선
            for i in range(len(W1)):
                w0, w1 = W1[i]
                if abs(w1) > abs(w0):
                    ax.plot(t, -(w0 * t + b1[i]) / w1, color="0.75", lw=0.6, zorder=1)
                else:
                    ax.plot(-(w1 * t + b1[i]) / w0, t, color="0.75", lw=0.6, zorder=1)

            Z = zs[layer][:, unit].reshape(RES, RES).cpu().numpy()
            ax.contour(X, Y, Z, levels=[0.0], colors=["crimson"], linewidths=2.2,
                       zorder=3)
            ax.contourf(X, Y, Z, levels=[0.0, Z.max() + 1], colors=["#ffe9ec"],
                        zorder=0)

            ax.set_xlim(-LIM, LIM)
            ax.set_ylim(-LIM, LIM)
            ax.set_xticks([])
            ax.set_yticks([])
            kind = "첫 층" if layer == 0 else "둘째 층"
            ax.set_title(f"h={hidden}  {kind} 유닛 #{unit}", fontsize=10)

        axes[row, 0].set_ylabel(f"hidden={hidden}", fontsize=10)

    fig.suptitle(
        f"{dsname}: 유닛 하나의 영점 집합 z=0 (회색 = 첫 층 직선)\n"
        "첫 층 유닛은 전역 직선, 둘째 층 유닛은 회색 선 위에서만 꺾인다",
        fontsize=12,
    )
    fig.tight_layout()
    out = FIG_DIR / f"zero_sets_{dsname}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


@torch.no_grad()
def measure_bend_locations(dsname: str, seed: int, device) -> dict:
    """P6의 정량 버전.

    둘째 층 유닛의 z=0 선을 따라가며 '기울기가 바뀌는 지점'을 찾고,
    그 지점들이 첫 층 직선 위에 있는지 확인한다.

    구현: z=0 선 위의 점들을 격자에서 뽑고, 각 점에서 첫 층 직선까지의 거리를 잰다.
    꺾임점은 첫 층 직선 위에 있어야 하므로, 꺾임점만 모으면 거리가 ~0이어야 한다.
    꺾임점은 '영점 집합 위에서 첫 층 활성 패턴이 바뀌는 곳'으로 찾는다.
    """
    model, _ = load_model(dsname, [32, 32], seed, device=device)
    xs = torch.linspace(-LIM, LIM, RES)
    spacing = float(xs[1] - xs[0])
    gy, gx = torch.meshgrid(xs, xs, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1).to(device)

    zs = [
        torch.cat(
            [model.pre_activations(pts[i : i + 65536])[l] for i in range(0, len(pts), 65536)]
        )
        for l in range(2)
    ]
    z2 = zs[1].reshape(RES, RES, -1).cpu()  # 둘째 층 pre-activation
    pat1 = (zs[0] > 0).reshape(RES, RES, -1).cpu()  # 첫 층 패턴

    W1, b1 = first_layer_lines(model)
    norms = W1.norm(dim=1, keepdim=True).clamp(min=1e-12)

    xs_np = xs.numpy()
    bend_d, plain_d = [], []

    for unit in range(z2.shape[-1]):
        Z = z2[..., unit]
        # z=0 을 가로지르는 격자 이웃쌍 = 영점 집합 위의 점
        cross = (Z[:, :-1] * Z[:, 1:]) < 0
        iy, ix = torch.nonzero(cross, as_tuple=True)
        if len(iy) == 0:
            continue
        # 그 지점에서 첫 층 패턴도 함께 바뀌는가? -> 꺾임점
        p_left = pat1[iy, ix]
        p_right = pat1[iy, ix + 1]
        is_bend = (p_left != p_right).any(dim=1)

        px = torch.tensor(xs_np[ix.numpy()], dtype=torch.float32)
        py = torch.tensor(xs_np[iy.numpy()], dtype=torch.float32)
        P = torch.stack([px, py], dim=1)  # (M, 2)
        d = ((W1 @ P.T + b1[:, None]).abs() / norms).min(dim=0).values

        bend_d.append(d[is_bend])
        plain_d.append(d[~is_bend])

    bend = torch.cat(bend_d) if bend_d else torch.zeros(0)
    plain = torch.cat(plain_d) if plain_d else torch.zeros(0)
    tol = 1.5 * spacing
    return {
        "n_bend": len(bend),
        "n_plain": len(plain),
        "bend_frac_on_line": float((bend <= tol).float().mean()) if len(bend) else float("nan"),
        "plain_median_dist": float(plain.median()) if len(plain) else float("nan"),
        "bend_median_dist": float(bend.median()) if len(bend) else float("nan"),
        "tol": tol,
    }


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--segments", type=int, default=200)
    ap.add_argument("--steps", type=int, default=4000)
    args = ap.parse_args()

    ensure_dirs()
    device = get_device()
    setup_matplotlib()
    print(f"device: {describe_device(device)}\n")

    run_convexity(args.seed, args.segments, args.steps, device)

    print("=" * 78)
    print("P6  둘째 층 유닛의 z=0 선은 첫 층 직선 위에서만 꺾이는가")
    print("=" * 78)
    for dsname in ("moons", "spiral"):
        out = zero_set_figure(dsname, args.seed, device)
        m = measure_bend_locations(dsname, args.seed, device)
        verdict = "PASS" if m["bend_frac_on_line"] > 0.99 else "FAIL"
        print(f"  [{dsname}] 허용치 {m['tol']:.5f}")
        print(f"    꺾임점 {m['n_bend']:,}개 중 첫 층 직선 위: "
              f"{m['bend_frac_on_line'] * 100:6.2f} %   {verdict}")
        print(f"    꺾임점 중앙거리   {m['bend_median_dist']:.5f}")
        print(f"    비꺾임점 중앙거리 {m['plain_median_dist']:.5f}  "
              f"(대조군 — 크게 나와야 정상)")
        print(f"    saved -> {out}")
        print()


if __name__ == "__main__":
    main()
