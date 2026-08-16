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


def toy_model() -> MLP:
    """STEP 2~6에서 계속 쓰는 4조각짜리 장난감 신경망.

    손으로 검산할 수 있게 가중치를 직접 박는다.
        W1 = [[1,0],[0,1]]   b1 = [-0.5,-0.5]   -> 경계 x1=0.5, x2=0.5
        W2 = [[2,3]]         b2 = [1.0]
    """
    model = MLP(in_dim=2, hidden=[2], out_dim=1)
    with torch.no_grad():
        model.layers[0].weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        model.layers[0].bias.copy_(torch.tensor([-0.5, -0.5]))
        model.layers[1].weight.copy_(torch.tensor([[2.0, 3.0]]))
        model.layers[1].bias.copy_(torch.tensor([1.0]))
    return model


def step2() -> None:
    head(2, "유닛 2개 = 4조각. 각 조각에서 신경망은 서로 다른 '하나의 행렬'")

    model = toy_model()

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


# ================================================================ STEP 6


def _partitions(items: list[int]):
    """집합의 모든 분할을 생성한다. 원소 4개면 15가지 (벨 수 B4=15)."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in _partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1 :]
        yield [[first]] + p


def step6() -> None:
    head(6, "조각을 합쳐보기 — 4조각짜리 신경망의 ε-path 전체를 손으로")

    model = toy_model()

    # 상자 안을 훑어 4조각과 각 조각의 (A_r, b_r), 점 개수를 얻는다
    g = torch.linspace(-2.0, 3.0, 400)
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    X = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    with torch.no_grad():
        y_true = model(X).squeeze(1)
        pat = model.activation_pattern(X)
    uniq, inv = torch.unique(pat, dim=0, return_inverse=True)
    R = uniq.shape[0]
    A, b = model.effective_affine(uniq)  # (R,1,2), (R,1)

    print(f"  상자 [-2,3]^2 를 400x400 격자로 훑음. 조각 {R}개.")
    print(f"  출력 범위: {y_true.min():.2f} ~ {y_true.max():.2f}\n")
    print(f"  {'조각':>10s} {'코드':>8s} {'A_r':>12s} {'b_r':>7s} {'점 개수':>8s}")
    for r in range(R):
        cnt = int((inv == r).sum())
        code = "".join(str(int(v)) for v in uniq[r])
        print(f"  {r:>10d} {code:>8s} {str([round(v,1) for v in A[r,0].tolist()]):>12s}"
              f" {b[r,0].item():7.2f} {cnt:8,d}")

    print()
    print("  '합친다' = 여러 조각이 하나의 (A, b)를 공유하게 만드는 것.")
    print("  대표값은 두 가지로 만들 수 있고, 둘은 다른 답을 준다:")
    print("    (a) centroid  : 합칠 조각들의 [A|b]를 점 개수로 가중평균")
    print("    (b) 최소제곱  : 합친 점들에 대해 A·x+b 를 직접 최소제곱 피팅")
    print()

    # ---- 각 K에 대해 '모든 분할'을 훑어 최적 분할을 찾는다 (R=4라 15가지뿐)
    def err_of(partition, mode: str) -> float:
        pred = torch.empty_like(y_true)
        for block in partition:
            mask = torch.isin(inv, torch.tensor(block))
            if mode == "centroid":
                w = torch.tensor([float((inv == r).sum()) for r in block])
                w = w / w.sum()
                Ab = (w[:, None] * A[block, 0]).sum(0)  # (2,)
                bb = (w * b[block, 0]).sum()
            else:  # 최소제곱
                Xs = X[mask]
                ys = y_true[mask]
                M = torch.cat([Xs, torch.ones(len(Xs), 1)], dim=1)  # (n,3)
                sol = torch.linalg.lstsq(M, ys.unsqueeze(1)).solution.squeeze(1)
                Ab, bb = sol[:2], sol[2]
            pred[mask] = X[mask] @ Ab + bb
        return (pred - y_true).abs().mean().item()

    all_parts = list(_partitions(list(range(R))))
    print(f"  조각이 {R}개뿐이라 가능한 분할 {len(all_parts)}가지를 '전부' 훑어")
    print(f"  각 K마다 진짜 최적 분할을 찾을 수 있다.\n")
    print(f"  {'K':>3s} {'ε (centroid)':>14s} {'ε (최소제곱)':>14s}   최적 분할 (centroid 기준)")
    for K in range(R, 0, -1):
        cands = [p for p in all_parts if len(p) == K]
        best = min(cands, key=lambda p: err_of(p, "centroid"))
        e_c = err_of(best, "centroid")
        best_ls = min(cands, key=lambda p: err_of(p, "lstsq"))
        e_l = err_of(best_ls, "lstsq")
        shown = " | ".join("".join(str(i) for i in sorted(bl)) for bl in sorted(best, key=min))
        print(f"  {K:3d} {e_c:14.4f} {e_l:14.4f}   {shown}")

    print()
    print("  ★ 최적 분할을 읽어보자 — 이게 이 STEP에서 가장 재미있는 부분이다.")
    print("      K=2 의 최적 분할은  {0,2} | {1,3}  이다.")
    print("      조각 0=00, 2=10 은 '유닛1이 꺼진' 것들이고,")
    print("      조각 1=01, 3=11 은 '유닛1이 켜진' 것들이다.")
    print("      즉 최적 분할은 '유닛1의 상태'로 묶었다. 유닛0의 상태는 무시했다.")
    print()
    print("      왜? W2 = [2, 3] 이라 유닛1의 가중치(3)가 유닛0(2)보다 크기 때문이다.")
    print("      유닛0을 틀리는 비용보다 유닛1을 틀리는 비용이 크다.")
    print("      그래서 '유닛0은 뭉개도 되고 유닛1은 구분해야 한다'가 최적이 된다.")
    print()
    print("      ★★ 이게 바로 'causal importance'의 가장 단순한 형태다.")
    print("         우리는 중요도를 손으로 정의한 적이 없는데, 그냥 ε을 최소화했더니")
    print("         알고리즘이 '더 중요한 유닛'을 알아서 찾아냈다.")
    print("         Stage 2의 SPD가 학습으로 하려는 일이 개념적으로 이것이다.")
    print()
    print("  읽는 법:")
    print("    K=4 -> ε=0. 당연하다. 안 합쳤으니 원본과 완전히 같다.")
    print("    K를 줄일수록 ε이 커진다. 이 표가 바로 ε-path 다.")
    print("    최소제곱이 centroid보다 항상 낫거나 같다 — 점들에 직접 맞추니까.")
    print("    ★ 그런데 centroid 가 '[A|b]를 클러스터링한다'는 우리 얘기와 맞는 방식이다.")
    print("      최소제곱은 A_r 을 아예 안 쓰고 (x, f(x)) 만 쓴다 — 그건 다른 문제다.")
    print("      우리는 '파라미터를 합친다'를 하려는 것이므로 centroid 쪽이 맞다.")
    print()
    print("  ★★ 결정적으로 중요한 한계: 여기서는 분할 15가지를 '전부' 훑었다.")
    print("     조각이 60,000개면 분할의 수가 우주의 원자 수를 아득히 넘는다.")
    print("     실제로는 greedy(agglomerative)로 근사할 수밖에 없고,")
    print("     따라서 우리가 그릴 ε-path 는 '진짜 최적'보다 위에 있는 상계다.")


# ================================================================ STEP 7


def step7() -> None:
    head(7, "'비슷하다'를 어떻게 재는가 — 세 척도가 서로 다른 답을 준다")

    print("  출력 1개, 입력 2차원인 조각 세 개를 상상하자 (b는 전부 0으로 둔다):")
    print()
    A0 = torch.tensor([10.0, 0.0])
    Ac = torch.tensor([1.0, 0.0])
    Ad = torch.tensor([7.07, 7.07])
    fmt = lambda t: str([round(v, 2) for v in t.tolist()])  # noqa: E731
    print(f"    기준  A  = {fmt(A0):>14s}")
    print(f"    후보  C  = {fmt(Ac):>14s}   <- A와 '방향'이 완전히 같다 (둘 다 x1축)")
    print(f"    후보  D  = {fmt(Ad):>14s}   <- A와 '크기'가 같다 (노름 10), 방향은 45도")
    print()
    print("  질문: A 에 더 가까운 것은 C 인가 D 인가?")
    print()

    def cos_dist(u, v):
        return 1 - (u @ v / (u.norm() * v.norm())).item()

    print(f"  {'척도':>28s} {'A~C':>10s} {'A~D':>10s}   더 가까운 쪽")
    fc, fd = (A0 - Ac).norm().item(), (A0 - Ad).norm().item()
    print(f"  {'Frobenius (성분 차이)':>28s} {fc:10.3f} {fd:10.3f}   {'C' if fc < fd else 'D'}")
    cc, cd = cos_dist(A0, Ac), cos_dist(A0, Ad)
    print(f"  {'cosine (방향 차이)':>28s} {cc:10.3f} {cd:10.3f}   {'C' if cc < cd else 'D'}")
    print()
    print("  ★ 벌써 답이 갈린다. Frobenius는 D, cosine은 C 라고 한다.")
    print("    cosine 입장에서 C는 A와 '거리 0' 이다 — 방향이 똑같으니까.")
    print("    Frobenius 입장에서 C는 A와 9만큼 떨어져 있다 — 크기가 10배 다르니까.")
    print()
    print("  그럼 logit 거리는? 이건 '실제로 출력이 얼마나 다른가'를 잰다.")
    print("  그런데 이 값은 데이터가 어디 있느냐에 따라 달라진다:")
    print()

    dists = {
        "데이터가 x1축 위 (1,0)": torch.tensor([[1.0, 0.0]]),
        "데이터가 x2축 위 (0,1)": torch.tensor([[0.0, 1.0]]),
        "데이터가 대각선 (0.7,0.7)": torch.tensor([[0.707, 0.707]]),
    }
    print(f"  {'데이터 분포':>28s} {'A~C':>10s} {'A~D':>10s}   더 가까운 쪽")
    for label, xs in dists.items():
        lc = ((xs @ (A0 - Ac)).abs()).mean().item()
        ld = ((xs @ (A0 - Ad)).abs()).mean().item()
        print(f"  {label:>28s} {lc:10.3f} {ld:10.3f}   {'C' if lc < ld else 'D'}")

    print()
    print("  ★★ 데이터가 x2축 위에 있으면 A와 C는 출력이 '완전히 같다' (거리 0).")
    print("     x1 성분이 다른데도 상관없다 — 데이터에 x1 성분이 없으니까.")
    print("     반대로 x1축 위에 있으면 A와 C가 가장 멀다.")
    print("     같은 세 행렬인데 데이터 분포가 답을 뒤집는다.")
    print()
    print("  그래서 세 척도는 이렇게 정리된다:")
    print("    Frobenius : 파라미터가 얼마나 다른가.        분포 무관. 계산 쌈.")
    print("    cosine    : 판단 '방향'이 얼마나 다른가.     크기를 버림. 분포 무관.")
    print("    logit     : 실제 행동이 얼마나 다른가.       분포 의존. 우리 목표와 일치.")
    print()
    print("  ★ 우리 문제 설정은 err(⟦P⟧, f_θ) ≤ ε 이다. 이 err 이 곧 행동 차이다.")
    print("    따라서 '옳은' 거리는 logit 거리이고, 나머지 둘은 계산이 싼 대리 지표다.")
    print("    Stage 1에서 셋을 다 재보는 이유: 대리 지표가 얼마나 쓸 만한지 알아야")
    print("    MNIST 규모에서 무엇을 쓸지 정할 수 있기 때문이다.")


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
    step6()
    step7()

    out = buildup_figure(model)
    print(f"\n{RULE}\n그림 저장: {out}\n{RULE}")


if __name__ == "__main__":
    main()
