"""ReLU 유닛 1개에서 64개까지, 한 걸음씩 늘려가며 직접 확인한다.

docs/walkthrough_stage1.md 와 짝을 이룬다. 문서를 읽으며 이걸 실행하면
모든 숫자를 눈으로 확인할 수 있다.

    python scripts/walkthrough.py

출력:
  STEP 1  ReLU 유닛 하나 = 직선 하나 = 반으로 접기        (손으로 계산 가능한 숫자)
  STEP 2  유닛 2개 = 4조각, 각 조각의 유효 선형맵          (손으로 계산 가능한 숫자)
  STEP 3  유닛 3개 = 8조각이 아니라 7조각. 불가능한 코드 찾기
  STEP 4  유닛을 늘리며 조각 수 세기: 실측 vs 2^h vs 공식
  STEP 5  왜 MNIST에서는 2^128 이 되는가
  그림    artifacts/figures/walkthrough_buildup.png
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlpinterp.models import MLP  # noqa: E402
from mlpinterp.train import load_model  # noqa: E402
from mlpinterp.utils import FIG_DIR, ensure_dirs, setup_matplotlib  # noqa: E402

RULE = "─" * 74


def head(n: int, title: str) -> None:
    print(f"\n{RULE}\nSTEP {n}  {title}\n{RULE}")


# ================================================================ STEP 1


def step1() -> None:
    head(1, "ReLU 유닛 하나 = 직선 하나 = 공간을 반으로 접기")

    # 손으로 따라갈 수 있게 아주 단순한 숫자를 쓴다.
    w = torch.tensor([1.0, 0.0])
    b = torch.tensor(-0.5)
    print(f"  유닛: z(x) = w·x + b,  w = {w.tolist()}, b = {b.item()}")
    print(f"        즉  z(x) = 1·x1 + 0·x2 - 0.5 = x1 - 0.5")
    print(f"  꺼짐/켜짐 경계:  z = 0  ->  x1 = 0.5  ->  세로 직선 하나")
    print()

    pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 5.0], [2.0, -3.0]])
    print(f"  {'x':>14s} {'z = x1-0.5':>12s} {'r = 1[z>0]':>11s} {'ReLU(z)':>9s}")
    for p in pts:
        z = (w @ p + b).item()
        print(f"  {str(p.tolist()):>14s} {z:12.2f} {int(z > 0):11d} {max(z, 0.0):9.2f}")

    print()
    print("  핵심: x2 는 z 에 전혀 영향이 없다 (w의 둘째 성분이 0).")
    print("        그래서 경계가 '세로 직선'이다. w 가 직선의 방향을 정하고 b 가 위치를 정한다.")
    print("        유닛 하나 = 평면을 두 조각으로 나누는 칼질 한 번.")


# ================================================================ STEP 2


def step2() -> None:
    head(2, "유닛 2개 = 4조각. 각 조각에서 신경망은 서로 다른 '하나의 행렬'")

    torch.manual_seed(0)
    model = MLP(in_dim=2, hidden=[2], out_dim=1)
    # 손으로 검산할 수 있게 가중치를 직접 박는다.
    with torch.no_grad():
        model.layers[0].weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        model.layers[0].bias.copy_(torch.tensor([-0.5, -0.5]))
        model.layers[1].weight.copy_(torch.tensor([[2.0, 3.0]]))
        model.layers[1].bias.copy_(torch.tensor([1.0]))

    print("  W1 = [[1, 0],     b1 = [-0.5, -0.5]      유닛0 경계: x1 = 0.5")
    print("        [0, 1]]                            유닛1 경계: x2 = 0.5")
    print("  W2 = [[2, 3]],    b2 = [1.0]")
    print()
    print("  두 칼질이 평면을 4조각으로 나눈다. 각 조각의 코드 r = (r0, r1):")
    print()

    probes = {
        "(0,0)  둘 다 꺼짐": [0.0, 0.0],
        "(1,0)  유닛0만 켜짐": [2.0, 0.0],
        "(0,1)  유닛1만 켜짐": [0.0, 2.0],
        "(1,1)  둘 다 켜짐": [2.0, 2.0],
    }
    print(f"  {'조각':>20s} {'x':>12s} {'A_r':>16s} {'b_r':>8s} {'검산':>18s}")
    for label, xv in probes.items():
        x = torch.tensor([xv])
        A, bb = model.effective_affine_at(x)
        direct = model(x).item()
        recon = (A[0] @ x[0]).item() + bb[0].item()
        print(
            f"  {label:>20s} {str(xv):>12s} {str([round(v,1) for v in A[0,0].tolist()]):>16s}"
            f" {bb[0,0].item():8.2f} {f'{direct:.2f} == {recon:.2f}':>18s}"
        )

    print()
    print("  읽는 법:")
    print("    코드 (0,0) -> D = diag(0,0) -> A_r = W2·D·W1 = [0, 0].  상수 함수.")
    print("    코드 (1,0) -> D = diag(1,0) -> W2의 0번 열만 살아남음 -> A_r = [2, 0].")
    print("    코드 (0,1) -> D = diag(0,1) -> W2의 1번 열만 살아남음 -> A_r = [0, 3].")
    print("    코드 (1,1) -> D = diag(1,1) -> 둘 다        -> A_r = [2, 3].")
    print()
    print("  ★ ReLU는 정보를 '자르는' 게 아니라 W2의 '열을 골라내는 스위치'다.")
    print("    켜진 유닛에 해당하는 열만 최종 선형맵에 기여한다.")


# ================================================================ STEP 3


def _realized_codes(W: torch.Tensor, b: torch.Tensor, lim: float = 6.0) -> set:
    """평면을 촘촘히 훑어 실제로 나타나는 코드를 모은다."""
    g = torch.linspace(-lim, lim, 3000)
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    codes = ((pts @ W.T + b) > 0).to(torch.uint8)
    return {tuple(c.tolist()) for c in torch.unique(codes, dim=0)}


def _all3() -> list[tuple[int, int, int]]:
    return [((i >> 2) & 1, (i >> 1) & 1, i & 1) for i in range(8)]


def step3() -> None:
    head(3, "유닛 3개 = 8조각? 아니다. 7조각이다. (단, 조건이 붙는다)")

    W = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    # ---- 3a. 일반위치 (general position)
    b_gp = torch.tensor([0.0, 0.0, -1.0])
    print("  [3a] 세 직선:  x1 = 0,   x2 = 0,   x1 + x2 = 1")
    print("       코드는 3비트니까 2^3 = 8 가지가 '있을 수 있다'. 실제로는?")
    print()
    seen = _realized_codes(W, b_gp)
    print(f"  {'코드':>12s}  실현 가능?")
    for c in _all3():
        print(f"  {str(c):>12s}  {'O' if c in seen else 'X  <- 불가능!'}")
    missing = [c for c in _all3() if c not in seen]
    print()
    print(f"  실현된 코드: {len(seen)}개 / 8개.  불가능한 코드: {missing}")
    print()
    print("  왜 불가능한가 — 산수로:")
    print("    코드 (0,0,1) 은 'x1<=0 이고 x2<=0 인데 x1+x2>1' 을 요구한다.")
    print("    x1<=0, x2<=0 이면 x1+x2<=0 일 수밖에 없다. 1보다 클 수 없다. 모순.")
    print("    반면 (1,1,0) 은 가능하다: x=(0.2, 0.2) -> x1>0, x2>0, x1+x2=0.4 < 1.  OK")
    print()
    print(f"  공식 1 + h + C(h,2) = 1 + 3 + 3 = 7.  실측 {len(seen)}.  일치.")

    # ---- 3b. 퇴화 (degenerate): 세 직선이 한 점에서 만남
    b_dg = torch.tensor([0.0, 0.0, 0.0])
    seen_dg = _realized_codes(W, b_dg)
    missing_dg = [c for c in _all3() if c not in seen_dg]
    print()
    print("  [3b] 그런데 셋째 직선을 x1 + x2 = 0 으로 바꾸면? (b를 -1에서 0으로)")
    print(f"       실현된 코드: {len(seen_dg)}개.  불가능: {missing_dg}")
    print()
    print("       조각이 7개가 아니라 6개다! 왜냐하면 이제 세 직선이 전부")
    print("       원점을 지나기 때문이다 (concurrent). 한 점에서 만나는 세 직선은")
    print("       평면을 부채꼴 6개로 가른다 — 교점이 3개가 아니라 1개로 뭉쳤다.")
    print()
    print("  ★ 공식 1 + h + C(h,2) 는 '상한'이고, 등호는 일반위치에서만 성립한다.")
    print("    일반위치 = 세 직선이 한 점에서 만나지 않고, 두 직선이 평행하지도 않음.")
    print("    학습된 가중치는 거의 항상 일반위치라 실무에서는 등호에 가깝다.")
    print("    (실제로 STEP 4에서 h=3일 때 학습된 모델도 정확히 7이 나온다.)")
    print()
    print("  ★★ 이 STEP의 진짜 교훈: 코드는 3비트지만 '기하학적으로 실현 가능한'")
    print("     코드는 더 적다. 평면이 좁아서 직선들이 서로를 제약하기 때문이다.")
    print("     이 제약의 정확한 값이 Zaslavsky 공식이고, STEP 5에서 보듯")
    print("     차원이 높아지면 제약이 사라져서 2^h 에 도달한다.")


# ================================================================ STEP 4


def step4(model: MLP) -> list[tuple[int, int]]:
    head(4, "유닛을 늘리며 조각 수 세기 — 실측 vs 2^h vs 공식")

    print("  실제 학습된 moons h=[64] 모델의 첫 층에서, 앞에서부터 h개 유닛만 켜고 센다.")
    print("  격자 1200x1200, 범위 ±3.5 (상자 안에서 센 값이라 이론값보다 작을 수 있다).")
    print()

    lim, res = 3.5, 1200
    g = torch.linspace(-lim, lim, res)
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

    W = model.layers[0].weight.detach().cpu()
    b = model.layers[0].bias.detach().cpu()
    z_all = pts @ W.T + b  # (N, 64)
    on = (z_all > 0).to(torch.uint8)

    print(f"  {'h':>4s} {'실측 조각수':>12s} {'2^h':>24s} {'공식 1+h+C(h,2)':>18s}")
    counts = []
    for h in [1, 2, 3, 4, 5, 8, 16, 32, 64]:
        n = int(torch.unique(on[:, :h], dim=0).shape[0])
        formula = 1 + h + comb(h, 2)
        two_h = 2**h
        two_str = f"{two_h:,}" if h <= 32 else f"{two_h:.3e}"
        counts.append((h, n))
        print(f"  {h:4d} {n:12,d} {two_str:>24s} {formula:18,d}")

    print()
    print("  읽는 법:")
    print("    h=1,2 까지는 2^h 와 같다. h=3 에서 처음 벌어진다 (7 < 8).")
    print("    h=64 에서 2^h 는 1.8e19 인데 실제 조각은 2천 개 남짓. 16자릿수 차이다.")
    print("    '코드가 64비트니까 영역이 2^64개'는 완전히 틀린 직관이다.")
    print()
    print("  실측이 공식보다 조금 작은 이유: 우리는 ±3.5 상자 안만 셌다.")
    print("    상자 밖에서만 나타나는 조각들이 빠졌다. 상자를 키우면 공식에 수렴한다.")
    return counts


# ================================================================ STEP 5


def step5() -> None:
    head(5, "그런데 MNIST 에서는 왜 2^128 이 되는가")

    print("  공식:  조각 수 <= sum_{i=0}^{d} C(h, i)     (d = 입력 차원, h = 유닛 수)")
    print()
    print(f"  {'세팅':>22s} {'d':>5s} {'h':>5s} {'상한':>22s}")
    for label, d, h in [("2D 합성", 2, 64), ("MNIST", 784, 128)]:
        ub = sum(comb(h, i) for i in range(min(d, h) + 1))
        s = f"{ub:,}" if ub < 10**12 else f"{ub:.3e}"
        print(f"  {label:>22s} {d:5d} {h:5d} {s:>22s}")
    print()
    print("  합의 범위가 i=0..min(d,h) 라는 데 주목.")
    print("    d=2, h=64  -> i=0,1,2 세 항만.        1 + 64 + 2016 = 2081")
    print("    d=784, h=128 -> i=0..128 전부 -> sum_i C(128,i) = 2^128. 상한이 무의미해진다.")
    print()
    print("  직관: STEP 3에서 코드가 제약된 이유는 '2차원 평면이 좁아서' 직선들이")
    print("        서로를 밀어냈기 때문이다. 784차원에서는 128개 초평면이 서로를")
    print("        전혀 제약하지 않는다 — 공간이 남아돈다. 모든 부호 조합이 실현된다.")
    print()
    print("  ★ 그래서 우리가 실측한 MNIST의 R/N = 1.000 (입력 6만개 = 영역 6만개) 은")
    print("    이상한 결과가 아니라 당연한 결과다. 이건 Stage 1의 출발점이")
    print("    K = 60,000 이라는 뜻이고, 여기서 K를 줄일 수 있는지가 실질적 질문이다.")


# ================================================================ 그림


def buildup_figure(model: MLP) -> Path:
    lim, res = 3.5, 900
    g = torch.linspace(-lim, lim, res)
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

    W = model.layers[0].weight.detach().cpu()
    b = model.layers[0].bias.detach().cpu()
    z_all = pts @ W.T + b
    on = (z_all > 0).to(torch.uint8)
    t = np.linspace(-lim * 2, lim * 2, 4)

    hs = [1, 2, 3, 4, 8, 64]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
    rng = np.random.default_rng(0)

    for ax, h in zip(axes.flat, hs):
        uniq, inv = torch.unique(on[:, :h], dim=0, return_inverse=True)
        n = uniq.shape[0]
        cols = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, max(n, 2)))
        rng.shuffle(cols)
        ax.imshow(
            cols[inv.reshape(res, res).numpy()],
            origin="lower",
            extent=(-lim, lim, -lim, lim),
            interpolation="nearest",
        )
        for i in range(h):
            w0, w1 = W[i].tolist()
            if abs(w1) > abs(w0):
                ax.plot(t, -(w0 * t + b[i].item()) / w1, color="k", lw=1.0, alpha=0.8)
            else:
                ax.plot(-(w1 * t + b[i].item()) / w0, t, color="k", lw=1.0, alpha=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        two = f"{2**h:,}" if h <= 32 else f"{2**h:.1e}"
        ax.set_title(f"유닛 {h}개 → 조각 {n:,}개   (2^{h} = {two})", fontsize=10)

    fig.suptitle(
        "ReLU 유닛을 하나씩 켜면 평면이 어떻게 쪼개지는가\n"
        "코드는 h비트지만 조각 수는 2^h 보다 훨씬 느리게 는다",
        fontsize=13,
    )
    fig.tight_layout()
    out = FIG_DIR / "walkthrough_buildup.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ================================================================ main


def main() -> None:
    ensure_dirs()
    setup_matplotlib()
    torch.set_printoptions(precision=3, sci_mode=False)

    step1()
    step2()
    step3()

    model, _ = load_model("moons", [64], 0, device="cpu")
    step4(model)
    step5()

    out = buildup_figure(model)
    print(f"\n{RULE}\n그림 저장: {out}\n{RULE}")


if __name__ == "__main__":
    main()
