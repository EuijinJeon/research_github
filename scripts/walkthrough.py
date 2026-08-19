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
  STEP 6  4조각 ε-path 전체를 전수탐색으로 (+ centroid vs refit)
  STEP 7  '비슷하다'를 재는 세 척도가 서로 다른 답을 준다
  STEP 8  네 번째 척도(부호벡터 해밍) — 가장 싸지만 유닛 중요도를 못 본다
  STEP 9  절편 가중치 λ 는 자유 손잡이가 아니다 — greedy 를 깨뜨린다
  STEP 10 앵커: 클러스터링 코드가 정답을 복원하는가 (+ 확률적 선택 γ=0.2)
  STEP 11 대조군: 랜덤 배정을 이기는가 (dense/chance 종점)
  ─ 여기서 방향을 튼다 ────────────────────────────────────────────
  STEP 12 잠깐 — '묶기' 는 애초에 맞는 목표였나 (아니었다. 근거 셋)
  STEP 13 조견표는 'k=1' 이다 — 4조각은 유닛 기여 2개가 만든 것
  STEP 14 깊이가 합성성을 깨뜨린다 (MNIST 1층 vs 2층) — Stage 2 의 존재 이유
  그림    artifacts/figures/walkthrough_buildup.png

읽는 순서 안내:
  STEP 6~11 은 '비슷한 조각끼리 묶는다' 는 **가설을 따라간 기록**이다.
  그 목표는 STEP 12 에서 기각된다. 그래도 6~11 을 남겨두는 이유는
  거기서 배운 **검증 방법**(앵커·손잡이·대조군)이 대상이 바뀌어도 그대로 쓰이기 때문이다.
  결론만 보려면 STEP 12 -> 13 -> 14.
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

from mlpinterp.data import get_dataset  # noqa: E402
from mlpinterp.models import MLP  # noqa: E402
from mlpinterp.symbols import legend  # noqa: E402
from mlpinterp.train import load_model  # noqa: E402
from mlpinterp.utils import FIG_DIR, ensure_dirs, setup_matplotlib  # noqa: E402

RULE = "─" * 74


def head(n: int, title: str) -> None:
    print(f"\n{RULE}\nSTEP {n}  {title}\n{RULE}")


# ================================================================ STEP 0
#
# 문제 정의. STEP 1 이전에 "무엇을 / 왜 / 어떤 파라미터로" 를 세워둔다.
# 이 절이 없으면 STEP 1~4의 '평면·직선·칼질' 어휘가 일반적 서술로 오해되고,
# STEP 5에서 d=784를 만났을 때 쌓아온 직관이 무너진다.


def _count_regions_exact(W: torch.Tensor, b: torch.Tensor) -> int:
    """초평면 배치가 만드는 영역 수를 '정확히' 센다 (d=1,2,3 공용).

    방법: 가능한 부호벡터 2^h 개를 전부 훑으며 각각이 실현 가능한지 LP로 판정한다.
    STEP 3에서 손으로 '(0,0,1)은 모순이라 불가능'을 보인 것과 같은 판정을 기계가 한다.

    ⚠️ 처음에는 난수 샘플링으로 셌는데 d=2에서 16이 나와야 할 자리에 14가 나왔다.
       상자를 키워도 16 → 15 → 14 로 오락가락했다. 잘린 게 아니라 얇은 영역을
       난수가 놓친 것이었다. '실측이 공식보다 작으면 상자 탓'이라는 설명이
       항상 맞는 게 아니라는 뜻이라 판정 자체를 정확한 방법으로 바꿨다.
    """
    from itertools import product

    import numpy as np
    from scipy.optimize import linprog

    Wn = W.numpy().astype(float)
    bn = b.numpy().astype(float)
    h, d = Wn.shape
    n_ok = 0
    for signs in product([0, 1], repeat=h):
        # 변수 [x (d개), t]. 부등식 s_i·(w_i·x + b_i) ≥ t 를 만족하며 t를 최대화한다.
        # 최적 t > 0 이면 '두께가 있는' 영역이 실제로 존재한다는 뜻.
        s = np.where(np.array(signs) > 0, 1.0, -1.0)
        res = linprog(
            c=np.concatenate([np.zeros(d), [-1.0]]),
            A_ub=np.hstack([-(s[:, None] * Wn), np.ones((h, 1))]),
            b_ub=s * bn,
            bounds=[(-50, 50)] * d + [(0, 1)],
            method="highs",
        )
        if res.status == 0 and -res.fun > 1e-9:
            n_ok += 1
    return n_ok


def _pad(s: str, width: int, right: bool = False) -> str:
    """터미널 표시폭 기준으로 채운다.

    한글은 터미널에서 2칸을 차지하는데 파이썬 format은 1칸으로 세어 열이 밀린다.
    한글이 섞인 표에서는 이 보정이 없으면 표가 읽히지 않는다.
    """
    from unicodedata import east_asian_width

    disp = sum(2 if east_asian_width(ch) in "WF" else 1 for ch in s)
    fill = " " * max(width - disp, 0)
    return fill + s if right else s + fill


# 0-3의 표와 그림이 '같은 배치'를 써야 한다. 이 seed는 임의로 고른 게 아니라,
# 교점이 전부 원점 근처(반경 1.5 안)에 모여 상자 하나에 16조각이 다 들어오는 것을
# 골랐다. seed=7은 두 직선이 거의 평행해 교점이 x=9.9까지 뻗어서, 그림에
# 조각 14개만 보이는데 표에는 16이라 적히는 어긋남이 생겼다.
DIM_SEED = 39


def _general_position(h: int, d: int, seed: int = DIM_SEED) -> tuple[torch.Tensor, torch.Tensor]:
    """일반위치에 가까운 초평면 h개. 절편을 0에서 떨어뜨려 동시교차를 피한다."""
    gen = torch.Generator().manual_seed(seed)
    W = torch.randn(h, d, generator=gen)
    W = W / W.norm(dim=1, keepdim=True)
    b = torch.randn(h, generator=gen) * 0.6
    return W, b


def step0_task(models: dict) -> None:
    head(0, "문제 정의 — 무엇을, 왜, 어떤 파라미터로 뜯어보는가")

    print("  [0-1] 우리가 뜯어볼 함수는 어디서 오는가")
    print()
    print("  해석하려면 먼저 '해석당할 함수'가 있어야 한다. 그래서 MLP를 학습시킨다.")
    print("  ★ 핵심: 우리는 이 태스크를 잘 푸는 게 목적이 아니다.")
    print("    태스크는 '뜯어볼 함수를 만들어내는 도구'일 뿐이다. 그래서 일부러")
    print("    이미 잘 풀리는 쉬운 문제를 골랐다 — 문제가 어려우면 나중에")
    print("    '모델이 못 배운 것'과 '우리 설명이 실패한 것'이 뒤섞이기 때문이다.")
    print()
    cols = [("태스크", 9), ("입력 x", 27), ("정답 y", 20), ("d", 5), ("N(train)", 10)]
    print("  " + "".join(_pad(c, w) for c, w in cols) + _pad("test acc", 9, right=True))
    for key, (model, ds, acc) in models.items():
        xin = {
            "moons": "평면 위의 점 (x₁, x₂)",
            "spiral": "평면 위의 점 (x₁, x₂)",
            "mnist": "28×28 이미지를 편 784벡터",
        }[ds.name]
        yout = {
            "moons": "초승달 2개 중 하나",
            "spiral": "나선 팔 3개 중 하나",
            "mnist": "숫자 10개 중 하나",
        }[ds.name]
        cells = [key, xin, yout, str(ds.input_dim), f"{len(ds.x_train):,}"]
        print(
            "  " + "".join(_pad(v, w) for v, (_, w) in zip(cells, cols))
            + _pad(f"{acc:.4f}", 9, right=True)
        )
    print()
    print("  세 태스크가 서로 다른 역할을 맡는다:")
    print("    moons  — 경계가 매끄러운 곡선 하나. 적은 조각으로 근사될 것 같은 쪽.")
    print("    spiral — 경계가 여러 번 감긴다. 많은 조각을 강제하는 쪽. moons와 대조.")
    print("    mnist  — d=784. '2차원이라서 됐던 것'이 아님을 확인하는 쪽.")


def step0_params(models: dict) -> None:
    print()
    print(RULE)
    print("  [0-2] 파라미터 — 각 기호가 무엇이고 무엇을 정하는가")
    print(RULE)
    legend("d", "h", "W1", "b1", "W2", "b2", "A_r", "b_r", "N")
    print("  1 hidden layer MLP:")
    print()
    print("      z = W1·x + b1        (h개 유닛의 pre-activation)")
    print("      a = ReLU(z)          (음수를 0으로)")
    print("      f = W2·a  + b2       (C_out개 클래스의 logit)")
    print()

    m2, ds2, _ = models["moons"]
    keys = ["moons"] + (["mnist"] if "mnist" in models else [])
    hdr = "".join(_pad(f"{k} h={models[k][0].hidden}", 16, right=True) for k in keys)
    print("  " + _pad("기호", 11) + _pad("정체", 26) + _pad("shape", 13) + hdr)

    def row(sym, what, shape, vals):
        print(
            "  " + _pad(sym, 11) + _pad(what, 26) + _pad(shape, 13)
            + "".join(_pad(v, 16, right=True) for v in vals)
        )

    def shp(t) -> str:
        return str(tuple(t.shape))

    row("d", "입력 차원", "—", [str(models[k][1].input_dim) for k in keys])
    row("h", "hidden 유닛 수", "—", [str(sum(models[k][0].hidden)) for k in keys])
    row("C_out", "출력 차원 (클래스 수)", "—", [str(models[k][1].n_classes) for k in keys])
    row("W1", "첫 층 가중치", "(h, d)", [shp(models[k][0].layers[0].weight) for k in keys])
    row("b1", "첫 층 절편", "(h,)", [shp(models[k][0].layers[0].bias) for k in keys])
    row("W2", "출력층 가중치", "(C_out, h)", [shp(models[k][0].layers[-1].weight) for k in keys])
    row("b2", "출력층 절편", "(C_out,)", [shp(models[k][0].layers[-1].bias) for k in keys])
    row("θ", "전체 파라미터 개수", "—",
        [f"{sum(p.numel() for p in models[k][0].parameters()):,}" for k in keys])
    row("N", "학습 데이터 개수", "—", [f"{len(models[k][1].x_train):,}" for k in keys])
    print("  " + "─" * 72)
    row("A_r", "영역 r의 유효 선형맵", "(C_out, d)",
        [f"({models[k][1].n_classes}, {models[k][1].input_dim})" for k in keys])
    row("b_r", "영역 r의 유효 절편", "(C_out,)", [f"({models[k][1].n_classes},)" for k in keys])
    row("[A_r|b_r]", "증강 행렬 = 조견표 한 줄", "(C_out, d+1)",
        [f"{models[k][1].n_classes*(models[k][1].input_dim+1):,}개 숫자" for k in keys])

    print()
    print("  ★ 각 파라미터가 '무엇을 정하는가' — 이게 STEP 1~2의 예고편이다:")
    print("      W1의 한 행 w_i  ->  유닛 i가 긋는 칼날의 '방향' (초평면의 법선)")
    print("      b1의 한 성분    ->  그 칼날의 '위치' (원점에서 얼마나 밀려났나)")
    print("      h              ->  칼질 횟수. 공간을 몇 번 자르는가")
    print("      W2             ->  '켜진 유닛들'을 어떻게 섞어 최종 출력을 만드는가")
    print("      d              ->  ★ 같은 h로 몇 조각이 나오는지를 정한다 (0-3)")
    print()
    print("  ★★ [A_r|b_r] 줄을 주목. MNIST에서 조견표 한 줄이 7,850개 숫자다.")
    print("     원본 파라미터가 101,770개인데 조견표가 60,000줄이면 4.7억 개 —")
    print("     원본보다 4,600배 크다. '설명'이라 부를 수 없다. 이게 Stage 1의 출발점이다.")


def step0_dimension() -> None:
    print()
    print(RULE)
    print("  [0-3] 왜 하필 2차원인가 — 편의가 아니라 실험 설계다")
    print(RULE)
    legend("d", "h", "C(h, i)", "1 + h + C(h, 2)", "2^h")
    print("  초평면 h개가 d차원 공간을 몇 조각으로 자르는가 (Zaslavsky):")
    print()
    print("      조각 수  ≤  Σ_{i=0}^{min(d,h)} C(h, i)")
    print()
    print("  합의 '위 끝'이 d 라는 데 전부가 걸려 있다.")
    print("  ★ 초평면 5개를 '똑같이' 두고 d만 바꿔보자. 코드는 셋 다 5비트다.")
    print()
    h = 5
    shapes = {1: "직선 위의 '점'", 2: "평면 위의 '직선'", 3: "공간 속의 '평면'"}
    dcols = [("d", 4), ("초평면의 모습", 20), ("2^h", 6), ("공식", 30), ("공식값", 8),
             ("실측 (LP 전수판정)", 20)]
    print("  " + "".join(_pad(c, w) for c, w in dcols))
    for d in (1, 2, 3):
        W, b = _general_position(h, d)
        got = _count_regions_exact(W, b)
        terms = "1 + " + " + ".join(f"C(5,{i})" for i in range(1, d + 1))
        formula = sum(comb(h, i) for i in range(d + 1))
        cells = [str(d), shapes[d], str(2**h), terms, str(formula), str(got)]
        print("  " + "".join(_pad(v, w) for v, (_, w) in zip(cells, dcols)))
    print()
    print("  ★ 셋 다 '코드는 5비트, 2^h = 32가지'로 똑같다. 그런데 실제 영역은 6 / 16 / 26.")
    print("    코드 길이(h)가 아니라 d 가 영역 수를 정한다.")
    print("    d가 1 늘 때마다 합에 항이 하나씩 붙기 때문이다.")
    print("    그리고 d ≥ h 가 되면 합이 끝까지 가서 Σ_i C(h,i) = 2^h — 상한이 무너진다.")
    print("    (실측은 부호벡터 32가지를 LP로 하나씩 판정한 값이다. 공식과 정확히 일치한다.)")
    print()
    print("  ★ d=2 가 우리에게 해주는 세 가지:")
    print("     (1) 그릴 수 있다.     폴리토프를 종이에 그려서 눈으로 검산할 수 있다.")
    print("     (2) 끊긴다.           합이 세 항(i=0,1,2)에서 멈춰 h의 2차식이 된다.")
    print("                           h=64 여도 조각이 2,081개뿐 — 전부 열거 가능하다.")
    print("     (3) 비교할 수 있다.   '샘플링으로 본 영역'과 '실제 존재하는 영역 전체'를")
    print("                           직접 대조할 수 있는 유일한 세팅이다.")
    print()
    print("  d=784 에서는 셋 다 깨진다. 그릴 수 없고, 상한이 2^128 이고, 열거가 불가능하다.")
    print()
    print("  ★★ 그래서 앞으로 이 문서가 '직선', '평면', '칼질', '조각'이라고 쓰면")
    print("     그건 전부 d=2 에서만 통하는 말이다. 일반형은 '초평면'과 '영역'이다.")
    print("     STEP 1~4는 d=2 세계이고, STEP 5에서 이 가정을 일부러 깬다.")


def step0_objective() -> None:
    print()
    print(RULE)
    print("  [0-4] 우리가 재는 두 숫자 — Ω 와 ε")
    print(RULE)
    legend("Ω", "ε", "K", "ε-path", "P", "⟦P⟧", "err")
    print("  프로젝트 전체가 푸는 문제는 한 줄이다:")
    print()
    print("      P* = argmin Ω(P)      s.t.   err(⟦P⟧, f_θ) ≤ ε")
    print()
    print("  말로: '오차 ε까지 봐준다고 할 때, 이 신경망의 가장 단순한 설명은")
    print("        무엇이고 그 단순함은 얼마인가?'")
    print()
    print("  Stage 1은 설명 P를 '구역별 조견표'로 고정한다:")
    print()
    print("      입력 x -> x가 속한 구역 k를 찾는다 -> 그 줄의 A_k·x + b_k 를 계산한다")
    print()
    for sym, what in [
        ("P", "조견표 (K줄, 각 줄이 [A_k|b_k])"),
        ("Ω(P)", "K — 조견표의 줄 수. ★ 이걸 최소화한다"),
        ("ε", "합친 조견표와 원본 f_θ의 출력 차이"),
        ("k(x)", "x가 어느 줄인지. ← 이건 아직 원본 신경망이 알려준다"),
    ]:
        print("  " + _pad(sym, 8) + what)
    print()
    print("  K를 줄이면 ε이 커진다. 그 맞바꿈 곡선이 ε-path 이고, 그게 산출물이다.")
    print("  ★ 이 문서(STEP 1~7)는 그 곡선을 그리기 위한 재료를 하나씩 쌓는 과정이다.")


# ---------------------------------------------------------------- 그림 ①


def task_figure(models: dict) -> Path:
    """우리가 뜯어볼 함수 3개를 눈으로. '입력이 뭐고 출력이 뭔가'를 먼저 보여준다."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    m_moons, ds_moons, _ = models["moons"]
    m_spiral, ds_spiral, _ = models["spiral"]

    # (a) 입력 데이터 자체 — 아직 모델 이야기가 아니다
    ax = axes[0]
    xs = ds_moons.x_train.numpy()
    ys = ds_moons.y_train.numpy()
    for c, col in zip(range(2), ["#4C72B0", "#DD8452"]):
        m = ys == c
        ax.scatter(xs[m, 0], xs[m, 1], s=4, c=col, label=f"정답 y = {c}", alpha=0.75)
    # 한글 폰트에 아래첨자 글리프가 없어 x₁ 이 두부로 깨진다. mathtext 로 그린다.
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(markerscale=3, fontsize=9, loc="upper right")
    ax.set_title(
        "(a) 태스크 moons — 입력과 정답\n입력 x = 평면 위의 점 $(x_1, x_2)$,  d = 2",
        fontsize=10,
    )

    # (b),(c) 학습이 끝난 f_θ — 이것이 '해석당할 함수'
    # 배경(모델의 예측)과 점(정답)을 같은 컬러맵·같은 스케일로 칠한다.
    # 서로 다른 맵을 쓰면 '이 색 영역 = 이 클래스'라는 대응이 안 읽힌다.
    lim, res = 3.5, 400
    g = torch.linspace(-lim, lim, res)
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    for ax, (mdl, ds, acc), tag in [
        (axes[1], models["moons"], "(b)"),
        (axes[2], models["spiral"], "(c)"),
    ]:
        with torch.no_grad():
            pred = mdl(grid).argmax(1).reshape(res, res).numpy()
        ax.imshow(
            pred, origin="lower", extent=(-lim, lim, -lim, lim),
            cmap="tab10", vmin=0, vmax=9, alpha=0.30, interpolation="nearest",
        )
        xs, ys = ds.x_train.numpy(), ds.y_train.numpy()
        ax.scatter(xs[:, 0], xs[:, 1], s=2.5, c=ys, cmap="tab10", vmin=0, vmax=9, alpha=0.9)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{tag} 학습된 $f_θ$ — {ds.name} h={mdl.hidden}\n"
            f"{sum(mdl.hidden)}유닛, test {acc:.3f}.  ★ 이 함수를 뜯는다",
            fontsize=10,
        )

    # (d) MNIST — 입력이 784차원이라 (b),(c)처럼 그릴 수 없다는 것이 요점
    ax = axes[3]
    if "mnist" in models:
        _, ds_m, acc_m = models["mnist"]
        tiles = ds_m.x_train[:9].reshape(9, 28, 28).numpy()
        mont = np.vstack([np.hstack(tiles[i * 3 : i * 3 + 3]) for i in range(3)])
        # 표준화된 값이라 극단 픽셀 몇 개가 범위를 늘려 흐릿해진다. 분위수로 자른다.
        ax.imshow(
            mont, cmap="gray_r", interpolation="nearest",
            vmin=np.percentile(mont, 2), vmax=np.percentile(mont, 99.5),
        )
        for k in (28, 56):
            ax.axhline(k - 0.5, color="w", lw=1.5)
            ax.axvline(k - 0.5, color="w", lw=1.5)
        ax.set_title(
            f"(d) 태스크 MNIST — d = 784\n"
            f"test {acc_m:.3f}.  ★ 입력이 784차원이라 (b),(c)처럼 못 그린다",
            fontsize=10,
        )
    else:
        ax.text(0.5, 0.5, "MNIST 없음\n(python scripts/train.py)", ha="center", va="center")
        ax.set_title("(d) 태스크: MNIST", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle(
        "STEP 0-1  우리가 '해석'할 대상은 이 함수들이다\n"
        "태스크를 잘 푸는 것이 목적이 아니라, 뜯어볼 함수를 얻는 것이 목적이다",
        fontsize=13,
    )
    fig.tight_layout()
    out = FIG_DIR / "step0_task.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- 그림 ②


def _draw_layer(ax, x, n_show, n_real, label, color, y_span=2.6):
    """세로로 늘어선 노드 한 층. n_real이 크면 가운데를 '생략 점 3개'로 대신한다.

    ⋮ (U+22EE) 는 한글 폰트에 글리프가 없어 두부로 깨진다. 직접 찍는다.
    """
    ys = np.linspace(y_span / 2, -y_span / 2, n_show)
    for i, y in enumerate(ys):
        if n_real > n_show and i == n_show // 2:
            for dy in (-0.13, 0.0, 0.13):
                ax.plot([x], [y + dy], marker="o", ms=2.6, color="#555", zorder=3)
            continue
        ax.add_patch(plt.Circle((x, y), 0.16, fc=color, ec="#333", lw=1.0, zorder=3))
    ax.text(x, -y_span / 2 - 0.42, label, ha="center", va="top", fontsize=10)
    return ys


def architecture_figure(models: dict) -> Path:
    """파라미터가 어디에 붙어 있는지. shape을 그림 위에 직접 적는다."""
    keys = ["moons"] + (["mnist"] if "mnist" in models else [])
    fig, axes = plt.subplots(1, len(keys), figsize=(7.5 * len(keys), 5.4))
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        mdl, ds, _ = models[key]
        d, h, c = ds.input_dim, sum(mdl.hidden), ds.n_classes
        x_in, x_hd, x_ou = 0.0, 2.2, 4.4
        ax.set_xlim(-0.95, 5.35)
        ax.set_ylim(-3.05, 2.5)
        ax.set_aspect("equal")  # 없으면 노드가 원이 아니라 타원으로 찌그러진다
        ax.axis("off")

        yin = _draw_layer(ax, x_in, min(d, 5), d, f"입력 x\nd = {d}", "#BBD5EA")
        yhd = _draw_layer(ax, x_hd, min(h, 7), h, f"ReLU 유닛\nh = {h}", "#F3C77B")
        you = _draw_layer(ax, x_ou, min(c, 5), c, f"출력 logit\nC_out = {c}", "#A8D5A2")

        for xa, xb, ya, yb in [(x_in, x_hd, yin, yhd), (x_hd, x_ou, yhd, you)]:
            for a in ya:
                for b_ in yb:
                    ax.plot([xa, xb], [a, b_], color="#999", lw=0.35, alpha=0.5, zorder=1)

        for xm, txt, note in [
            (
                (x_in + x_hd) / 2,
                f"W1  ({h}, {d})\nb1  ({h},)",
                "각 행 $w_i$ = 초평면의 '방향'\n$b1$ 성분 = 그 '위치'",
            ),
            (
                (x_hd + x_ou) / 2,
                f"W2  ({c}, {h})\nb2  ({c},)",
                "켜진 유닛의 열만\n살아남는다 (STEP 2)",
            ),
        ]:
            ax.text(xm, 1.72, txt, ha="center", va="bottom", fontsize=11,
                    color="#B5471B", weight="bold")
            ax.text(xm, -2.42, note, ha="center", va="top", fontsize=9, color="#444")
        n_par = sum(p.numel() for p in mdl.parameters())
        ax.set_title(
            f"{ds.name}  hidden={mdl.hidden}   (θ 총 {n_par:,}개)\n"
            f"영역 하나의 조견표 줄 [A_r|b_r] = ({c}, {d + 1}) = {c * (d + 1):,}개 숫자",
            fontsize=11,
        )

    fig.suptitle(
        "STEP 0-2  파라미터가 어디에 붙어 있는가\n"
        "W1·b1 이 공간을 자르고, W2·b2 가 살아남은 유닛을 섞는다",
        fontsize=13,
    )
    fig.tight_layout()
    out = FIG_DIR / "step0_architecture.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- 그림 ③


def dimension_figure() -> Path:
    """같은 초평면 5개가 d=1,2,3 에서 몇 조각을 만드는가 + 성장 곡선."""
    h = 5
    fig = plt.figure(figsize=(14, 8.6))

    # (a) d=1 : 직선 위의 점 5개 -> 6조각
    ax = fig.add_subplot(2, 3, 1)
    W1d, b1d = _general_position(h, 1)
    n1 = _count_regions_exact(W1d, b1d)
    xlim1 = 1.6
    cuts = np.sort((-b1d / W1d[:, 0]).numpy())
    edges = np.concatenate([[-xlim1], cuts, [xlim1]])
    cols = plt.get_cmap("turbo")(np.linspace(0.08, 0.92, len(edges) - 1))
    for i in range(len(edges) - 1):
        ax.axvspan(edges[i], edges[i + 1], ymin=0.42, ymax=0.58, color=cols[i])
    for cpt in cuts:
        ax.plot([cpt, cpt], [0.34, 0.66], color="k", lw=2.2)
    ax.set_xlim(-xlim1, xlim1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("$x_1$")
    ax.set_title(f"d = 1 :  초평면 = '점'\n점 5개 → 조각 {n1}개", fontsize=11)

    # (b) d=2 : 평면 위의 직선 5개 -> 16조각
    ax = fig.add_subplot(2, 3, 2)
    lim, res = 2.6, 1400
    W2d, b2d = _general_position(h, 2)
    n2 = _count_regions_exact(W2d, b2d)
    g = torch.linspace(-lim, lim, res)
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    code = ((pts @ W2d.T + b2d) > 0).to(torch.uint8)
    uniq, inv = torch.unique(code, dim=0, return_inverse=True)
    cols = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, uniq.shape[0]))
    np.random.default_rng(1).shuffle(cols)
    ax.imshow(
        cols[inv.reshape(res, res).numpy()],
        origin="lower", extent=(-lim, lim, -lim, lim), interpolation="nearest",
    )
    t = np.linspace(-lim * 2, lim * 2, 4)
    for i in range(h):
        w0, w1 = W2d[i].tolist()
        if abs(w1) > abs(w0):
            ax.plot(t, -(w0 * t + b2d[i].item()) / w1, color="k", lw=1.4)
        else:
            ax.plot(-(w1 * t + b2d[i].item()) / w0, t, color="k", lw=1.4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(
        f"d = 2 :  초평면 = '직선'   ★ STEP 1~4가 사는 곳\n직선 5개 → 조각 {n2}개",
        fontsize=11,
    )

    # (c) d=3 : 공간 속의 평면 5개 -> 26조각
    ax = fig.add_subplot(2, 3, 3, projection="3d")
    W3d, b3d = _general_position(h, 3)
    n3 = _count_regions_exact(W3d, b3d)
    span = np.linspace(-3, 3, 2)
    P, Q = np.meshgrid(span, span)
    for i in range(h):
        w = W3d[i].numpy()
        bb = b3d[i].item()
        k = int(np.argmax(np.abs(w)))
        axis = [0, 1, 2]
        axis.remove(k)
        coords = [None, None, None]
        coords[axis[0]], coords[axis[1]] = P, Q
        coords[k] = -(w[axis[0]] * P + w[axis[1]] * Q + bb) / w[k]
        ax.plot_surface(*coords, alpha=0.30, color=plt.get_cmap("turbo")(i / h), lw=0)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(f"d = 3 :  초평면 = '평면'\n평면 5개 → 조각 {n3}개", fontsize=11)

    # (d) 성장 곡선 — 여기가 이 그림의 결론
    ax = fig.add_subplot(2, 1, 2)
    hs = np.arange(1, 129)
    for d, style, col in [
        (1, "-", "#6BAED6"), (2, "-", "#B5471B"), (3, "-", "#74A97B"),
        (784, "--", "#555555"),
    ]:
        vals = [sum(comb(int(hh), i) for i in range(min(d, int(hh)) + 1)) for hh in hs]
        lab = f"d = {d}" + ("   ← 2D 합성 (STEP 1~4)" if d == 2 else "")
        lab += "   ← MNIST. 2^h 와 같아진다 (STEP 5)" if d == 784 else ""
        ax.plot(hs, vals, style, color=col, lw=2.2 if d in (2, 784) else 1.4, label=lab)
    ax.axvline(64, color="#B5471B", lw=0.8, ls=":", alpha=0.7)
    ax.axvline(128, color="#555555", lw=0.8, ls=":", alpha=0.7)
    ax.annotate(
        "h=64, d=2\n조각 2,081개\n(열거 가능)",
        xy=(64, 2081), xytext=(70, 3e6), fontsize=9.5, color="#B5471B",
        arrowprops=dict(arrowstyle="->", color="#B5471B", lw=1.2),
    )
    # 위첨자 유니코드(10³⁸)는 한글 폰트에 글리프가 없어 두부로 깨진다. mathtext 로.
    ax.annotate(
        "h=128, d=784\n조각 $3.4\\times10^{38}$개\n(원천 불가능)",
        xy=(128, 2.0**128), xytext=(74, 1e30), fontsize=9.5, color="#333",
        arrowprops=dict(arrowstyle="->", color="#333", lw=1.2),
    )
    ax.set_yscale("log")
    ax.set_xlabel("h  (ReLU 유닛 = 초평면의 개수)")
    ax.set_ylabel("영역 수 상한  (로그 눈금)")
    ax.set_title(
        "같은 h 라도 d 가 영역 수를 정한다 — 이것이 '2차원을 고른' 이유다",
        fontsize=11.5,
    )
    ax.legend(fontsize=9.5, loc="upper left")
    ax.grid(alpha=0.25, which="both")

    fig.suptitle(
        "STEP 0-3  왜 하필 2차원인가\n"
        "d=2 는 편의가 아니라 실험 설계다: 그릴 수 있고, 공식이 h² 에서 끊기고, 전부 셀 수 있다",
        fontsize=13,
    )
    fig.tight_layout()
    out = FIG_DIR / "step0_dimension.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


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

    legend("h", "d", "2^h", "1+h+C(h,2)", "일반위치")
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

    legend("d", "h", "C(h, i)", "R", "N", "R/N")
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

    print("  ⚠️ 먼저 알아둘 것: 이 STEP 부터 STEP 11 까지는 **가설을 따라간 기록**이다.")
    print("     '비슷한 선형함수끼리 묶어 설명을 줄인다' 는 목표는 **STEP 12 에서 기각된다.**")
    print("     그래도 남겨두는 이유는 두 가지다:")
    print("       (1) 여기서 배우는 검증 방법(앵커·손잡이·대조군)은 대상이 바뀌어도 그대로 쓴다")
    print("       (2) 어떻게 기각되는지를 보려면 무엇을 기각하는지 먼저 알아야 한다")
    print()

    legend("K", "\u03a9", "\u03b5", "\u03b5-path", "A_r", "b_r", "centroid")
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
    print()
    print("  ⚠️ 여기 원래 이렇게 적혀 있었다 — 그리고 그 판단은 2026-08-19 에 뒤집혔다:")
    print('      "최소제곱은 A_r 을 안 쓰고 (x, f(x)) 만 쓴다 — 그건 다른 문제다.')
    print('       우리는 파라미터를 합치려는 것이므로 centroid 쪽이 맞다."')
    print()
    print("    선행연구 원문을 읽고 나서 알게 된 것: **최소제곱이 곧 refit 이고,**")
    print("    affine map 을 클러스터링한 유일한 선행연구(Aletheia)가 바로 그걸 한다.")
    print("    논문 Algorithm 1 의 마지막 단계가 '클러스터마다 GLM 을 다시 적합' 이다.")
    print("    즉 최소제곱은 '다른 문제'가 아니라 **이 분야의 표준 절차**였다.")
    print()
    print("    두 방식의 관계를 정확히 적으면 이렇다:")
    print("      centroid : 대표값을 [A_r|b_r] '들'로부터 만든다.  파라미터만 쓴다.")
    print("      refit    : 대표값을 그 클러스터의 (x, f(x)) 로부터 다시 적합한다.")
    print("      → 어느 쪽이 '맞다'가 아니라, **둘은 서로 다른 손잡이**다.")
    print("        클러스터링(어느 조각을 묶나)과 대표값 만들기(묶고 나서 뭘 쓰나)는 별개다.")
    print("        우리 Ω = K 정의에는 후자가 안 들어 있다 — 그래서 공짜처럼 보이지만 아니다.")
    print()
    print("    → 방침: **refit 있음/없음 두 곡선을 나란히 그린다.** 하나를 고르지 않는다.")
    print("      (한때 'MNIST 는 영역당 1점이라 refit 불가' 라고 적었던 것도 틀렸다.")
    print("       refit 은 영역이 아니라 클러스터 단위라, K=100 이면 클러스터당 600점이다.)")
    print()
    print("  ★★ 결정적으로 중요한 한계: 여기서는 분할 15가지를 '전부' 훑었다.")
    print("     조각이 60,000개면 분할의 수가 우주의 원자 수를 아득히 넘는다.")
    print("     실제로는 greedy(agglomerative)로 근사할 수밖에 없고,")
    print("     따라서 우리가 그릴 ε-path 는 '진짜 최적'보다 위에 있는 상계다.")


# ================================================================ STEP 7


def step7() -> None:
    head(7, "'비슷하다'를 어떻게 재는가 — 세 척도가 서로 다른 답을 준다")

    legend("Frobenius", "cosine", "logit \uac70\ub9ac", "[A_r | b_r]")
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


# ================================================================ STEP 8
#
# 2026-08-19 선행연구 원문 확인으로 추가된 절.
# polytope lens 원문에서 Frobenius 는 각주의 지나가는 제안일 뿐이었고,
# 그들이 실제로 쓴 것은 '부호벡터 위의 해밍 거리' 였다. 척도가 넷이 됐다.


def _toy_pieces() -> tuple:
    """STEP 6과 똑같은 4조각을 뽑아 (코드, A, b, 점, 조각라벨) 로 돌려준다.

    STEP 8~11이 전부 이 넷을 공유하므로 한 곳에서 만든다.
    """
    model = toy_model()
    g = torch.linspace(-2.0, 3.0, 400)
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    X = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    with torch.no_grad():
        y = model(X).squeeze(1)
        pat = model.activation_pattern(X)
    uniq, inv = torch.unique(pat, dim=0, return_inverse=True)
    A, b = model.effective_affine(uniq)          # (R,1,2), (R,1)
    return uniq, A[:, 0], b[:, 0], X, y, inv     # A:(R,2)  b:(R,)


def _piece_table(codes, A, b, inv) -> None:
    print(f"  {'조각':>6s} {'코드 r':>8s} {'A_r':>14s} {'b_r':>8s} {'점 개수':>9s}")
    for r in range(len(codes)):
        code = "".join(str(int(v)) for v in codes[r])
        av = "[" + ", ".join(f"{v:.0f}" for v in A[r].tolist()) + "]"
        print(f"  {r:>6d} {code:>8s} {av:>14s} {b[r].item():8.1f} {int((inv==r).sum()):9,d}")


# ---- 네 가지 척도. 전부 '조각 두 개 -> 실수 하나' 로 통일한다 ----------


def _d_hamming(codes, A, b, X, y, inv, i, j, lam=1.0) -> float:
    """부호벡터 r 끼리 다른 비트 수. [A|b] 를 만들 필요조차 없다 — 가장 싸다."""
    return float((codes[i] != codes[j]).sum())


def _d_euclid(codes, A, b, X, y, inv, i, j, lam=1.0) -> float:
    """증강행렬 [A_r | lam*b_r] 의 유클리드 거리. 출력이 1개라 Frobenius 와 같은 값."""
    u = torch.cat([A[i], lam * b[i : i + 1]])
    v = torch.cat([A[j], lam * b[j : j + 1]])
    return (u - v).norm().item()


def _d_cosine(codes, A, b, X, y, inv, i, j, lam=1.0) -> float:
    """A_r 의 방향 차이. 크기와 절편을 통째로 버린다."""
    u, v = A[i], A[j]
    if u.norm() < 1e-9 or v.norm() < 1e-9:
        return 1.0                                # 영벡터는 방향이 없다
    return 1.0 - (u @ v / (u.norm() * v.norm())).item()


def _d_logit(codes, A, b, X, y, inv, i, j, lam=1.0) -> float:
    """두 조각의 '실제 출력 차이'. 그 두 조각에 속한 점들 위에서만 잰다."""
    m = (inv == i) | (inv == j)
    xs = X[m]
    dA, db = A[i] - A[j], b[i] - b[j]
    return (xs @ dA + db).abs().mean().item()


METRICS = [
    ("해밍 (부호벡터 r)", _d_hamming),
    ("유클리드 ([A|b])", _d_euclid),
    ("cosine (A 방향)", _d_cosine),
    ("logit (실제 출력)", _d_logit),
]


def step8() -> None:
    head(8, "척도가 넷이 됐다 — 그리고 가장 싼 것이 가장 중요한 것을 못 본다")

    legend("A_r", "b_r", "r", "[A_r | b_r]", "K", "ε")
    codes, A, b, X, y, inv = _toy_pieces()
    R = len(codes)

    print("  STEP 6의 그 4조각을 그대로 쓴다. 이번엔 '어느 둘이 가까운가'만 본다.\n")
    _piece_table(codes, A, b, inv)
    print()
    print("  STEP 7에서 척도가 셋이라고 했다. 2026-08-19 선행연구 원문 확인에서")
    print("  네 번째가 나왔다 — polytope lens 가 실제로 쓴 것은 Frobenius 가 아니라")
    print("  **부호벡터 r 위의 해밍 거리** 였다 (Frobenius 는 각주의 지나가는 제안이었다).")
    print("  해밍은 [A_r|b_r] 를 만들 필요조차 없어 넷 중 가장 싸다. MNIST에서 특히 그렇다.")
    print()

    # ---- 쌍거리 표 ----
    pairs = [(i, j) for i in range(R) for j in range(i + 1, R)]
    print(f"  {'척도':>20s} " + " ".join(f"{f'{i}-{j}':>7s}" for i, j in pairs) + "   최근접쌍")
    nearest = {}
    for name, fn in METRICS:
        vals = [fn(codes, A, b, X, y, inv, i, j) for i, j in pairs]
        lo = min(vals)
        tie = [f"{i}{j}" for (i, j), v in zip(pairs, vals) if abs(v - lo) < 1e-9]
        nearest[name] = tie
        print(f"  {name:>20s} " + " ".join(f"{v:7.3f}" for v in vals)
              + f"   {'/'.join(tie)}")
    print()
    print("  ★ 해밍 줄을 보라. 1.000 이 네 번 나온다 — 네 쌍이 전부 동점이다.")
    print("    (0,1) (0,2) (1,3) (2,3) 이 구별되지 않는다.")
    print("    유닛0을 뒤집든 유닛1을 뒤집든 '비트 하나'로 똑같이 세기 때문이다.")
    print()
    # ---- 진짜 질문: 척도가 '옳은 분할'을 고르는가 ----
    print("  " + "-" * 70)
    print("  그런데 최근접쌍은 중간 결과일 뿐이다. 진짜 질문은 이것이다:")
    print("    '이 척도로 묶으면 STEP 6의 전수탐색 최적 {0,2}|{1,3} 이 나오는가?'")
    print()
    print("  ⚠️ 여기서 한 번 틀렸다. 남겨둘 가치가 있는 실패다.")
    print("     처음에 블록 비용을 '블록 안 쌍거리의 평균'으로 잡고 블록마다 더했다.")
    print("     그랬더니 넷 다 1+3 분할(예: 0 | 123)을 골랐다 — 척도와 무관하게.")
    print("     이유: 홑원소 블록은 쌍이 없어 비용이 그냥 0 이다.")
    print("     즉 '조각 하나를 떼어내고 나머지를 다 뭉치기'가 언제나 공짜로 싸 보인다.")
    print("     척도의 성질이 아니라 **비용함수의 결함**이었다.")
    print()
    print("  고친 방법: 블록당 (쌍거리제곱의 합 ÷ 블록 크기) 를 쓴다.")
    print("    이건 K-means 목적함수(블록 중심까지의 제곱거리 합)와 정확히 같은 값이고,")
    print("    항등식  Σ_{i<j} d²_ij / n_b  =  Σ_i ‖x_i − 중심‖²  로 쌍거리만으로 계산된다.")
    print("    → 거리만 있으면 되므로 **해밍·cosine·logit 에도 그대로 쓸 수 있다.**")
    print("      (K-means 자체는 유클리드 전용이지만, 이 '값'은 척도를 안 가린다.)")
    print("    블록이 커지면 나누는 n_b 보다 쌍의 수가 빨리 늘어 비용이 제대로 오른다.")
    print()

    parts2 = _k_partitions(list(range(R)), 2)

    def metric_cost(partition, fn) -> float:
        """블록당 쌍거리제곱합 / 블록크기. 블록 중심까지의 제곱거리 합과 같다."""
        tot = 0.0
        for blk in partition:
            if len(blk) < 2:
                continue
            ds = [fn(codes, A, b, X, y, inv, i, j) ** 2
                  for a, i in enumerate(blk) for j in blk[a + 1:]]
            tot += sum(ds) / len(blk)
        return tot

    def eps_centroid(partition) -> float:
        """STEP 6과 같은 방식: 대표값 = 점 개수 가중평균, ε = 평균 절대오차."""
        pred = torch.empty_like(y)
        for blk in partition:
            m = torch.isin(inv, torch.tensor(blk))
            w = torch.tensor([float((inv == r).sum()) for r in blk])
            w = w / w.sum()
            Ab = (w[:, None] * A[blk]).sum(0)
            bb = (w * b[blk]).sum()
            pred[m] = X[m] @ Ab + bb
        return (pred - y).abs().mean().item()

    shown = lambda p: " | ".join("".join(str(i) for i in bl)  # noqa: E731
                                for bl in sorted(p, key=min))
    eps = {shown(p): eps_centroid(p) for p in parts2}
    best_eps = min(eps, key=eps.get)

    hdr = f"  {'K=2 분할':>10s} {'진짜 ε':>9s}"
    for name, _ in METRICS:
        hdr += f" {name.split()[0]:>9s}"
    print(hdr)
    costs = {name: {shown(p): metric_cost(p, fn) for p in parts2}
             for name, fn in METRICS}
    for p in sorted(parts2, key=lambda q: eps[shown(q)]):
        k = shown(p)
        row = f"  {k:>10s} {eps[k]:9.4f}"
        for name, _ in METRICS:
            row += f" {costs[name][k]:9.3f}"
        print(row + ("   <- 진짜 최적" if k == best_eps else ""))
    print()
    print(f"  진짜 최적(ε 최소)은 {best_eps} 이다. STEP 6의 전수탐색 결과와 같다.")
    print("  (1+3 분할 네 개의 ε 이 전부 같은 값인 것은 이 장난감이 대칭이라 그렇다.)")
    print()
    print(f"  {'척도':>20s} {'고른 분할':>16s} {'맞았나':>7s}   비고")
    verdict = {}
    for name, _ in METRICS:
        c = costs[name]
        lo = min(c.values())
        picks = sorted([k for k, v in c.items() if abs(v - lo) < 1e-9])
        ok = (len(picks) == 1 and picks[0] == best_eps)
        verdict[name] = ok
        note = "" if len(picks) == 1 else f"{len(picks)}개 동점 — 고르지 못한다"
        print(f"  {name:>20s} {'/'.join(picks):>16s} {'O' if ok else 'X':>7s}   {note}")
    print()
    print("  ★ 예측 P7 이 맞았다. 해밍은 {0,2}|{1,3} 과 {0,1}|{2,3} 에 **똑같은 값**을 준다.")
    print("    두 분할 다 '한 비트로 가르기' 라서, 어느 비트인지를 구별할 수단이 없다.")
    print("    해밍이 보는 것은 부호벡터뿐이고, W2 = [2,3] 은 부호벡터에 안 들어 있다.")
    print()
    print("    이게 왜 중요한가: STEP 6의 ★★ 는 '중요도를 정의한 적 없는데 ε 최소화만으로")
    print("    유닛1이 더 중요하다는 게 나왔다' 였다. 그 발견은 **W2 를 보는 척도에서만** 나온다.")
    print("    → 해밍은 넷 중 가장 싸지만, 정확히 그 성질을 잃는다.")
    print("      MNIST 에서 싸다는 이유로 해밍을 고르면 무엇을 포기하는지가 이것이다.")
    print()
    print("  ★★ 덤으로 나온 것: cosine 도 틀린다 — 그것도 동점이 아니라 확실하게.")
    print("     cosine 은 절편 b_r 을 통째로 버리고 A_r 의 방향만 본다.")
    print("     조각 0 의 A_r = [0,0] 은 방향이 아예 없어서(영벡터) 거리가 정의되지 않는다.")
    print("     실모델에서도 A_r 이 작은 조각마다 같은 일이 생긴다.")
    print()
    print("  정리:")
    print("    유클리드 · logit : 진짜 최적을 고른다.  (logit 은 ε 과 단위가 같으니 당연)")
    print("    해밍             : 고르지 못한다 (동점).  가장 싸지만 유닛 중요도를 못 본다.")
    print("    cosine           : 틀린 것을 고른다.     절편을 버리는 대가.")
    print("    → 대리 지표로 쓸 만한 것은 **유클리드 하나**다. 이게 D1 의 첫 실측 답이다.")


def _k_partitions(items: list[int], k: int) -> list[list[list[int]]]:
    """원소를 정확히 k개 블록으로 나누는 모든 분할. R=4, k=2 면 7가지."""
    return [p for p in _partitions(items) if len(p) == k]


def _agglomerative(codes, A, b, X, y, inv, fn, lam=1.0, linkage="average",
                   stop_k=2) -> tuple:
    """계층 병합. 매 단계 가장 가까운 블록 쌍을 합친다.

    STEP 9~11이 공유한다. 반환: (stop_k 시점의 분할, 병합 이력)
    이력의 각 항목은 (합친 뒤 블록 수, 합쳐진 조각들, 그때의 거리).
    """
    blocks = [[i] for i in range(len(codes))]
    hist, out = [], None
    while len(blocks) > 1:
        best = None
        for p in range(len(blocks)):
            for q in range(p + 1, len(blocks)):
                ds = [fn(codes, A, b, X, y, inv, i, j, lam)
                      for i in blocks[p] for j in blocks[q]]
                d = sum(ds) / len(ds) if linkage == "average" else min(ds)
                if best is None or d < best[0] - 1e-12:
                    best = (d, p, q)
        d, p, q = best
        merged = sorted(blocks[p] + blocks[q])
        blocks = [bl for k, bl in enumerate(blocks) if k not in (p, q)] + [merged]
        hist.append((len(blocks), merged, d))
        if len(blocks) == stop_k:
            out = [sorted(bl) for bl in blocks]
    return out, hist


def _show(part) -> str:
    """분할을 '02 | 13' 꼴 문자열로."""
    return " | ".join("".join(str(i) for i in sorted(bl))
                      for bl in sorted(part, key=min))



# ================================================================ STEP 9
#
# Aletheia 논문 §5.1 은 "계수와 절편의 유클리드 거리" 라고만 쓴다 — 가중치가 없다.
# 구현체에만 all_bias_weight 라는 미문서화 손잡이가 있었다.
# 그러면 그 손잡이를 돌리면 무슨 일이 생기는가? 여기서 직접 돌려본다.


def step9() -> None:
    head(9, "절편 가중치 λ 는 자유 손잡이가 아니다 — 돌리면 앵커가 깨진다")

    legend("[A_r | b_r]", "b_r", "K", "ε")
    codes, A, b, X, y, inv = _toy_pieces()
    R = len(codes)
    pairs = [(i, j) for i in range(R) for j in range(i + 1, R)]

    print("  거리를 [A_r | b_r] 위에서 잰다고 했다. 그런데 A 와 b 는 단위가 다르다.")
    print("  A 성분이 크면 b 차이가 거리에서 사라진다 — 교훈 L2 에서 피하려던 바로 그 문제다.")
    print("  그래서 절편에 가중치를 걸 수 있다:  [A_r | λ·b_r].")
    print()
    print("  ⚠️ 출처를 정확히 해두자. **λ 는 Aletheia 논문에 없다.**")
    print("     논문 §5.1 은 '계수와 절편의 유클리드 거리' 라고만 쓰고 가중치도 표준화도")
    print("     언급하지 않는다. λ 는 구현체(all_bias_weight)에만 있는 미문서화 손잡이다.")
    print("     → λ=1 이 '선행연구 재현' 이고, λ≠1 은 **우리 추가분**이다.")
    print()
    print("  손으로 먼저 적어보자. 조각 넷의 [A|b] 차이는 이렇다:\n")
    print(f"  {'쌍':>5s} {'ΔA':>10s} {'Δb':>7s}   d² = ‖ΔA‖² + λ²·Δb²")
    for i, j in pairs:
        dA = A[i] - A[j]
        db = (b[i] - b[j]).item()
        na = (dA @ dA).item()
        av = "[" + ", ".join(f"{v:.0f}" for v in dA.tolist()) + "]"
        print(f"  {f'{i}-{j}':>5s} {av:>10s} {db:7.1f}   d² = {na:5.2f} + {db*db:.2f}·λ²")
    print()
    print("  이 여섯 줄만 보면 교차점을 손으로 풀 수 있다:")
    print("    (0,2) 와 (1,2):   4 + 1.00·λ²  =  13 + 0.25·λ²   ->  λ² = 12,  λ = 3.464")
    print("    (0,1) 과 (1,2):   9 + 2.25·λ²  =  13 + 0.25·λ²   ->  λ² =  2,  λ = 1.414")
    print()
    print("  즉 λ 를 키우면 어느 순간 순위가 뒤집힌다. 정말 그런지 훑어보자.\n")

    lams = [0.0, 0.5, 1.0, 1.4, 1.5, 2.0, 3.0, 3.4, 3.5, 5.0, 10.0]
    parts2 = _k_partitions(list(range(R)), 2)
    shown = lambda p: " | ".join("".join(str(i) for i in bl)  # noqa: E731
                                for bl in sorted(p, key=min))

    def wcss(partition, lam) -> float:
        tot = 0.0
        for blk in partition:
            if len(blk) < 2:
                continue
            ds = [_d_euclid(codes, A, b, X, y, inv, i, j, lam) ** 2
                  for a, i in enumerate(blk) for j in blk[a + 1:]]
            tot += sum(ds) / len(blk)
        return tot

    print(f"  {'λ':>6s} " + " ".join(f"{f'{i}-{j}':>7s}" for i, j in pairs)
          + f" {'최근접':>6s} {'전수탐색 K=2':>12s} {'greedy K=2':>11s} {'첫 병합':>7s}")
    for lam in lams:
        vals = [_d_euclid(codes, A, b, X, y, inv, i, j, lam) for i, j in pairs]
        lo = min(vals)
        near = "/".join(f"{i}{j}" for (i, j), v in zip(pairs, vals) if abs(v - lo) < 1e-6)
        c = {shown(p): wcss(p, lam) for p in parts2}
        mv = min(c.values())
        pick = "/".join(sorted(k for k, v in c.items() if abs(v - mv) < 1e-6))
        gpart, ghist = _agglomerative(codes, A, b, X, y, inv, _d_euclid, lam)
        gpick = _show(gpart)
        first = "".join(str(i) for i in ghist[0][1])
        flag = "" if gpick == "02 | 13" else "  <- greedy 깨짐"
        print(f"  {lam:6.1f} " + " ".join(f"{v:7.3f}" for v in vals)
              + f" {near:>6s} {pick:>12s} {gpick:>11s} {first:>7s}{flag}")
    print()
    # 교차점을 수치로 정확히 찾아 손계산과 대조한다
    def nearest_is_02_13(lam) -> bool:
        vals = {(i, j): _d_euclid(codes, A, b, X, y, inv, i, j, lam) for i, j in pairs}
        lo = min(vals.values())
        return all(abs(vals[p] - lo) > 1e-9 or p in {(0, 2), (1, 3)} for p in vals)

    lo_l, hi_l = 0.0, 20.0
    for _ in range(60):
        mid = (lo_l + hi_l) / 2
        if nearest_is_02_13(mid):
            lo_l = mid
        else:
            hi_l = mid
    print(f"  이분법으로 찾은 최근접쌍 교차점: λ = {lo_l:.4f}   (손계산 √12 = {12**0.5:.4f})")
    print()
    print("  ★ 예측 P8 은 **절반만 맞았다.** 맞은 쪽부터:")
    print("    λ 가 3.464 를 넘으면 최근접쌍이 (1,2) 로 바뀐다. 교차점이 손계산과 소수 4자리까지 같다.")
    print("    (1,2) 는 코드 01 과 10 — **두 유닛을 동시에 가로지르는 병합**이다.")
    print("    이유: (1,2) 는 ΔA 가 크지만(‖ΔA‖²=13) Δb 가 작다(0.25).")
    print("    λ 를 키우면 Δb 항이 지배하므로 **절편만 비슷하면 계수가 아무리 달라도** 가깝다고 나온다.")
    print()
    print("  ⚠️ 틀린 쪽: '앵커가 깨진다' 고 예측했는데, **전수탐색은 λ=10 에서도 안 깨진다.**")
    print("     표의 '전수탐색 K=2' 열이 끝까지 02 | 13 이다. 왜 그런지 파고들면 이렇다:")
    print()
    print("       K=2 균형 분할은 네 조각을 **둘씩 짝지어야** 한다.")
    print("       (1,2) 를 한 블록으로 쓰면 나머지 (0,3) 이 **강제**된다.")
    print("       그런데 (0,3) 은 여섯 쌍 중 **가장 먼** 쌍이다 (Δb=2.5 라 λ 에 가장 민감).")
    print("       03 | 12 의 비용 = 13 + 3.25·λ²  vs  02 | 13 의 비용 = 4 + λ².")
    print("       λ 가 아무리 커도 후자가 작다. **전수탐색은 이 짝의 강제를 미리 본다.**")
    print()
    print("  ★★ 그래서 진짜 결론은 예측보다 날카롭다 — **λ 의 위험은 greedy 의 위험이다.**")
    print("     표의 'greedy K=2' 열을 보라. λ=3.5 부터 012 | 3 으로 바뀐다.")
    print("     greedy 는 첫 수로 (1,2) 를 합치고, 그 순간 (0,3) 이 강제되는 미래를 못 본다.")
    print("     한 번 합치면 무를 수 없기 때문이다.")
    print()
    print("     → STEP 6 에서 '우리 ε-path 는 상계다' 라고 추상적으로 적었던 것의 **구체적 실례**다.")
    print("       상계라는 사실이 손해로 나타나는 방식이 이것이다: 손잡이를 잘못 돌리면")
    print("       **전수탐색이라면 안 틀렸을 자리에서 greedy 만 틀린다.**")
    print()
    print("  ★★★ 그리고 이것이 앵커가 필수인 이유다.")
    print("     ε 만 보고 있으면 이 사고를 알아챌 수 없다 — λ 를 바꿨으니 ε 값도 바뀌는 게 당연하고,")
    print("     '조금 나빠졌네' 로 보일 뿐이다. **정답을 아는 장난감**에서만 '틀렸다' 가 보인다.")
    print()
    print("  반대쪽도 보자 — λ=0 (절편을 아예 버리기) 은 이 장난감에서 앵커를 통과한다.")
    print("    그렇다고 안전하다는 뜻은 아니다. 교훈 L2 의 상황(기울기는 같은데 절편이 다른")
    print("    두 조각)이 이 장난감에는 없을 뿐이다. 실모델에서는 λ=0 이 그 둘을 합쳐버린다.")
    print()
    print("  → 방침: **λ=1 을 기준선으로 고정하고**, λ 를 바꿀 때는")
    print("    'ε 이 얼마나 변하나' 가 아니라 **'앵커를 아직 통과하나'** 를 먼저 본다.")


# ================================================================ STEP 10
#
# param-decomp 핸드북이 **필수 절차**로 규정한 것:
#   "메커니즘이 이미 알려진 타깃을 먼저 분해해 방법이 그것을 복원하는지 확인하라.
#    이 앵커는 필수다 — 학습 부족과 잘못된 재구성 목적함수는 둘 다 '깨끗해 보이는 null'을
#    만들 수 있어서, 앵커 없는 타깃은 '구조가 정말 없음'과 '방법이 잘못 설정됨'을 구분 못 한다."
# 우리는 앵커를 이미 갖고 있다 — STEP 6의 전수탐색 정답 {0,2}|{1,3}.


def step10() -> None:
    head(10, "앵커 — 클러스터링 코드가 정답을 복원하는지 먼저 확인한다")

    legend("K", "ε", "[A_r | b_r]", "centroid")
    codes, A, b, X, y, inv = _toy_pieces()
    R = len(codes)
    TRUTH = "02 | 13"

    print("  선행연구가 요구하는 순서는 이것이다:")
    print("    실모델에 돌리기 **전에**, 정답을 아는 장난감에서 정답이 나오는지 확인한다.")
    print("    통과 못 하면 거기서 멈춘다. 실모델 결과를 해석하지 않는다.")
    print()
    print(f"  우리 앵커: STEP 6의 전수탐색 최적  {TRUTH}  (= 유닛1의 상태로 묶기)")
    print("  시험 대상: average linkage 계층 병합, λ=1, 척도 넷.\n")

    print(f"  {'척도':>20s} {'병합 순서':>22s} {'K=2 결과':>10s} {'앵커':>6s}")
    passed = []
    for name, fn in METRICS:
        part, hist = _agglomerative(codes, A, b, X, y, inv, fn, lam=1.0)
        seq = " -> ".join("".join(str(i) for i in mg) for _, mg, _ in hist)
        got = _show(part)
        ok = got == TRUTH
        passed.append((name, ok))
        print(f"  {name:>20s} {seq:>22s} {got:>10s} {'통과' if ok else '실패':>6s}")
    print()
    print("  ★ STEP 8의 결론이 그대로 재현된다 — 이번엔 '분할 전수 비교'가 아니라")
    print("    '실제 알고리즘을 돌려서' 확인했다는 점이 다르다. 두 경로가 같은 답을 준다.")
    print()
    for name, ok in passed:
        if not ok:
            print(f"    {name} 는 앵커를 통과하지 못한다 -> 실모델에 쓰지 않는다.")
    print()
    print("  → **유클리드와 logit 만 다음 단계로 보낸다.** 이게 앵커의 용도다.")
    print("    ε 만 보고 있었다면 해밍도 cosine 도 '조금 더 나쁜 곡선'으로 보였을 뿐이다.")
    print()
    # ---------------- VPD 의 확률적 선택 -------------------------------
    print("  " + "-" * 70)
    print("  STEP 9 에서 본 것: greedy 는 첫 수를 무를 수 없어 틀릴 수 있다.")
    print("  VPD 논문이 그 대응책을 적어뒀다 — **일부러 최선이 아닌 쌍을 고른다.**")
    print()
    print("    후보 쌍을 비용 오름차순으로 세우고 순위 J 에 확률을 준다:")
    print("      P(J) ∝ exp(−γ·J),   논문 설정 γ = 0.2")
    print("    역CDF 로 뽑는다:  J = ⌊ −log(1 − u(1 − e^{−γN})) / γ ⌋,  u ~ U(0,1)")
    print()

    gamma, N = 0.2, R * (R - 1) // 2          # 첫 병합에서 후보 쌍 6개
    w = torch.tensor([float(torch.exp(torch.tensor(-gamma * J))) for J in range(N)])
    prob = w / w.sum()
    print(f"  후보가 {N}개일 때(=조각 4개의 첫 병합) 순위별 확률:")
    print("    " + " ".join(f"J={J}:{prob[J]*100:5.1f}%" for J in range(N)))
    print()
    w_inf = 1.0 / (1.0 - float(torch.exp(torch.tensor(-gamma))))
    print(f"  논문이 인용한 18.1% 는 후보가 '아주 많을 때'의 값이다:")
    print(f"    가중치 합 -> 1/(1−e^{{−0.2}}) = {w_inf:.3f},  P(J=0) = {100/w_inf:.1f}%")
    print(f"    후보가 {N}개뿐이면 합이 {w.sum():.3f} 이라 P(J=0) = {prob[0]*100:.1f}% 로 커진다.")
    print("    → γ 의 효과는 **후보 수에 따라 달라진다.** 조각이 많을수록 더 과감해진다.")
    print()

    # 역CDF 샘플링이 정말 저 분포를 주는지 확인
    torch.manual_seed(0)
    u = torch.rand(200_000)
    J = torch.floor(-torch.log(1 - u * (1 - torch.exp(torch.tensor(-gamma * N)))) / gamma)
    emp = torch.bincount(J.long(), minlength=N).float() / len(u)
    print("  역CDF 샘플링 20만 번으로 검산:")
    print("    " + " ".join(f"J={j}:{emp[j]*100:5.1f}%" for j in range(N)))
    print(f"    이론값과 최대 차이 {float((emp[:N]-prob).abs().max())*100:.2f}%p — 식이 맞다.")
    print()

    # ---- 그래서 STEP 9 의 사고를 구해내는가 ----
    print("  " + "-" * 70)
    print("  이제 진짜 시험: STEP 9 에서 greedy 가 틀렸던 λ=5 에 확률적 선택을 넣어본다.")
    print()

    def stochastic_once(lam, gen) -> tuple:
        """확률적 계층 병합 1회. greedy 와 같지만 최선 대신 순위추첨으로 고른다."""
        blocks = [[i] for i in range(R)]
        while len(blocks) > 2:
            cand = []
            for p in range(len(blocks)):
                for q in range(p + 1, len(blocks)):
                    ds = [_d_euclid(codes, A, b, X, y, inv, i, j, lam)
                          for i in blocks[p] for j in blocks[q]]
                    cand.append((sum(ds) / len(ds), p, q))
            cand.sort()
            n = len(cand)
            uu = torch.rand(1, generator=gen).item()
            jj = int(-torch.log(torch.tensor(1 - uu * (1 - torch.exp(torch.tensor(-gamma * n)))))
                     / gamma)
            jj = min(jj, n - 1)
            _, p, q = cand[jj]
            merged = blocks[p] + blocks[q]
            blocks = [bl for k, bl in enumerate(blocks) if k not in (p, q)] + [merged]
        return [sorted(bl) for bl in blocks]

    def eps_of(part) -> float:
        pred = torch.empty_like(y)
        for blk in part:
            m = torch.isin(inv, torch.tensor(blk))
            ww = torch.tensor([float((inv == r).sum()) for r in blk])
            ww = ww / ww.sum()
            pred[m] = X[m] @ (ww[:, None] * A[blk]).sum(0) + (ww * b[blk]).sum()
        return (pred - y).abs().mean().item()

    for lam in (1.0, 5.0):
        gpart, _ = _agglomerative(codes, A, b, X, y, inv, _d_euclid, lam)
        gen = torch.Generator().manual_seed(0)
        runs = [stochastic_once(lam, gen) for _ in range(200)]
        hits = sum(1 for r in runs if _show(r) == TRUTH)
        best = min(runs, key=eps_of)
        print(f"  λ={lam:.0f}:  greedy = {_show(gpart):>9s} (ε={eps_of(gpart):.4f})")
        print(f"        확률적 200회 중 정답 {hits}회 ({hits/2:.1f}%),"
              f" 최선 = {_show(best):>9s} (ε={eps_of(best):.4f})")
    print()
    print("  ★ λ=5 에서 greedy 는 **항상** 틀리지만, 확률적 선택은 여러 번 중 몇 번은 맞힌다.")
    print("    그리고 우리는 ε 을 볼 수 있으므로 **여러 번 돌려 가장 좋은 것을 취하면 된다.**")
    print("    이게 VPD 가 말한 '국소최소 탈출' 의 전부다 — 구현은 실제로 10줄이었다.")
    print()
    print("  ★★ 단, 오해하지 말 것: 확률적 선택은 **greedy 의 손실을 줄일 뿐 없애지 않는다.**")
    print("     조각이 4개라 후보가 6개뿐이니 200회면 공간을 사실상 다 훑는다.")
    print("     조각이 1,400개면 첫 병합 후보만 979,300개다. 같은 비율의 탐색은 불가능하다.")
    print("     → **우리 ε-path 는 여전히 상계다.** 확률적 선택은 그 상계를 낮출 뿐이다.")
    print("       그래서 보고할 때 손잡이(γ, linkage, λ, 재시작 횟수)를 함께 적어야 한다.")


# ================================================================ STEP 11
#
# param-decomp 핸드북의 공정성 조건:
#   "...and includes the dense and chance endpoints. Merely beating the architectural-unit
#    count is not evidence of useful minimality. If a bad or untrained baseline passes the
#    stated bar, strengthen the bar before interpreting."
# Aletheia Table 3 의 SLFN 열이 그 실례다 — FL-Net 과 크기가 같고 초기화만 랜덤인 망.


def step11() -> None:
    head(11, "대조군 — 클러스터링이 '아무렇게나 묶기'를 이기는가")

    legend("K", "ε", "ε-path", "centroid")
    codes, A, b, X, y, inv = _toy_pieces()
    R = len(codes)

    def eps_of(part) -> float:
        pred = torch.empty_like(y)
        for blk in part:
            m = torch.isin(inv, torch.tensor(blk))
            w = torch.tensor([float((inv == r).sum()) for r in blk])
            w = w / w.sum()
            pred[m] = X[m] @ (w[:, None] * A[blk]).sum(0) + (w * b[blk]).sum()
        return (pred - y).abs().mean().item()

    print("  ε-path 를 그렸다고 하자. 곡선이 아래로 내려간다. 그래서 뭐가 증명됐나?")
    print("  **아무것도 아니다** — 비교 대상이 없으면 곡선은 해석할 수 없다.")
    print()
    print("  선행연구가 요구하는 것은 두 가지다:")
    print("    (1) 양 끝점을 포함할 것 — K=R (안 합침, ε=0) 과 K=1 (전부 하나).")
    print("    (2) 나쁜 기준선이 통과하면 기준을 강화할 것.")
    print()
    print("  가장 엄한 기준선은 이것이다: **모델·데이터·K·대표값 방식이 전부 같고")
    print("  오직 '어느 조각을 어느 블록에 넣는가' 만 다른 배정.** 다른 변수가 안 섞인다.")
    print("  (Aletheia Table 3 의 SLFN 열이 딱 이 형태다 — FL-Net 과 크기가 같고 초기화만 랜덤.)")
    print()

    print(f"  {'K':>3s} {'분할 수':>7s} {'최적 ε':>9s} {'랜덤 배정 평균 ε':>16s} "
          f"{'최악 ε':>9s} {'이득(배)':>9s}")
    rows = []
    for K in range(R, 0, -1):
        parts = _k_partitions(list(range(R)), K)
        es = [eps_of(p) for p in parts]
        lo, mean, hi = min(es), sum(es) / len(es), max(es)
        gain = float("inf") if lo < 1e-12 else mean / lo
        rows.append((K, len(parts), lo, mean, hi, gain))
        g = "—" if lo < 1e-12 else f"{gain:9.2f}"
        print(f"  {K:>3d} {len(parts):>7d} {lo:9.4f} {mean:16.4f} {hi:9.4f} {g:>9s}")
    print()
    print("  읽는 법:")
    print("    K=4 : 안 합쳤으니 ε=0. **dense 종점.** 분할이 하나뿐이라 랜덤도 같다.")
    print("    K=1 : 전부 한 덩어리. **chance 종점.** 분할이 하나뿐이라 여기도 랜덤과 같다.")
    print("    → 양 끝점에서는 클러스터링이 할 일이 없다. **가운데에서만 이득이 생긴다.**")
    print("      곡선의 의미는 전부 그 가운데 구간에 있다.")
    print()

    K2 = [r for r in rows if r[0] == 2][0]
    print(f"  ★ K=2 를 보자. 최적 ε={K2[2]:.4f}, 랜덤 평균 ε={K2[3]:.4f}, 최악 ε={K2[4]:.4f}.")
    print(f"    클러스터링이 아무렇게나 묶기보다 **{K2[5]:.2f}배** 낫다.")
    print("    이 배수가 곧 '클러스터링이 실제로 한 일' 의 크기다.")
    print()
    print("  ★★ 왜 '랜덤 배정' 이 '학습 안 된 모델' 보다 엄한 기준선인가:")
    print("     학습 안 된 모델을 쓰면 바뀌는 것이 너무 많다 — 가중치도, 조각 개수도,")
    print("     [A_r|b_r] 의 스케일도 전부 다르다. 곡선이 달라져도 무엇 때문인지 모른다.")
    print("     랜덤 배정은 **오직 배정만** 바꾼다. 차이가 나면 그건 배정 때문이다.")
    print("     둘 다 두면 좋지만, 순서는 랜덤 배정이 먼저다.")
    print()

    # 랜덤 '라벨링' 과 랜덤 '분할' 이 같은지 확인 — R=4, K=2 에서는 같다
    print("  덤: '랜덤 배정' 을 어떻게 뽑을지도 정해야 한다. 두 가지가 있다.")
    print("    (a) 조각마다 라벨을 K개 중 균등하게 뽑는다   (b) 분할을 균등하게 뽑는다")
    lab = {}
    for mask in range(1 << R):
        g0 = [i for i in range(R) if not (mask >> i) & 1]
        g1 = [i for i in range(R) if (mask >> i) & 1]
        if not g0 or not g1:
            continue
        lab[_show([g0, g1])] = lab.get(_show([g0, g1]), 0) + 1
    print(f"    R=4, K=2 에서 (a)는 라벨링 {sum(lab.values())}가지가 분할 {len(lab)}개로 모이고,")
    print(f"    각 분할이 정확히 {set(lab.values())} 번씩 나온다 -> **(a)와 (b)가 일치한다.**")
    print("    R 이나 K 가 커지면 (a)는 균형 분할을 선호해 둘이 갈라진다.")
    print("    → 실모델에서는 **어느 쪽을 썼는지 명시**해야 한다. 우리는 (a)를 쓴다")
    print("      ('배정만 랜덤' 이라는 말과 뜻이 같으므로).")
    print()
    print("  ▶ 이 STEP 의 결론 한 줄:")
    print("    ε-path 는 혼자서는 아무 말도 하지 않는다. **같은 그림에 랜덤 배정 곡선을 겹쳐야**")
    print("    비로소 '이 모델이 압축된다' 를 말할 수 있다. 안 겹치면 그냥 곡선 하나다.")


# ================================================================ STEP 12
#
# 2026-08-19. STEP 6~11 을 다 만들고 나서 물었어야 할 질문을 늦게 물었다:
# "비슷한 선형함수끼리 묶는다" 는 목표 자체가 맞는가?
# 답은 아니오였고, 근거가 셋이다. 이 STEP 은 그 셋을 숫자로 보인다.


def _region_radius(model, X, n: int = 60, seed: int = 0) -> torch.Tensor:
    """데이터 점에서 무작위 방향으로 영역을 벗어날 때까지의 거리.

    영역이 '얼마나 넓은 정의역을 갖는가' 를 재는 가장 단순한 방법이다.
    이분법 40회면 배정밀도 한계까지 좁혀진다.
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X), generator=g)[:n]
    out = []
    with torch.no_grad():
        for i in idx:
            x0 = X[i : i + 1]
            p0 = model.activation_pattern(x0)
            d = torch.randn(1, X.shape[1], generator=g)
            d = d / d.norm()
            lo, hi = 0.0, 1.0
            while hi < 1e4 and (model.activation_pattern(x0 + hi * d) == p0).all():
                hi *= 2
            for _ in range(40):
                mid = (lo + hi) / 2
                if (model.activation_pattern(x0 + mid * d) == p0).all():
                    lo = mid
                else:
                    hi = mid
            out.append(lo)
    return torch.tensor(out), idx


def step12() -> None:
    head(12, "잠깐 — '비슷한 선형함수끼리 묶기' 는 애초에 맞는 목표였나")

    legend("[A_r | b_r]", "A_r", "R", "N", "K")
    print("  STEP 6~11 에서 우리는 '어떻게 잘 묶을까' 를 여섯 걸음에 걸쳐 파고들었다.")
    print("  척도를 넷 비교하고, λ 를 조이고, 앵커를 세우고, 대조군을 놓았다.")
    print("  **그런데 '묶는다는 목표 자체가 맞는가' 는 한 번도 안 물었다.**")
    print("  늦게 물었더니 답이 아니오였다. 근거가 셋이고, 셋 다 숫자로 나온다.\n")

    # ---------- 근거 1: ReLU 의존 ----------
    print("  " + "=" * 70)
    print("  근거 1 — 이 렌즈는 ReLU 밖으로 못 나간다. **저자들이 직접 적어뒀다.**")
    print()
    print("    polytope lens (Black et al. 2022) 의 미해결 질문 목록 첫 항목:")
    print()
    print("      \"Fuzzy polytope boundaries with other activations — 오늘날 많은 신경망,")
    print("       특히 대규모 언어모델은 GELU·softmax 같은 매끄러운 활성함수를 쓰는데,")
    print("       그러면 **폴리토프가 사실 폴리토프가 아니게 된다** — 모서리가 휘거나 뭉개진다.\"")
    print()
    print("    트랜스포머로의 확장도 미해결로 남겨뒀다.")
    print("    즉 조각/폴리토프라는 어휘 전체가 **ReLU 계열 전용**이고,")
    print("    그 사실이 2022년에 이미 공개돼 있었다.")
    print()

    # ---------- 근거 2: 조각이 너무 작다 ----------
    print("  " + "=" * 70)
    print("  근거 2 — 조각이 작아지면 [A_r|b_r] 는 '함수' 가 아니라 '한 점의 야코비안' 이다.")
    print()
    print("    '이 영역에서 신경망은 A_r·x + b_r 로 계산한다' 가 뜻을 가지려면")
    print("    그 영역이 **뭔가를 담을 만큼 넓어야** 한다. 실제로 재보자:")
    print("    데이터 점에서 무작위 방향으로 영역을 벗어날 때까지의 거리 vs 최근접이웃 거리.\n")

    rows = []
    for name, hidden in [("moons", [64]), ("spiral", [64]), ("mnist", [128])]:
        try:
            model, _ = load_model(name, hidden, 0, device="cpu")
            ds = get_dataset(name, seed=0)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {name}: {type(e).__name__}")
            continue
        X = ds.x_train
        r, idx = _region_radius(model, X)
        nn = torch.cdist(X[idx], X).topk(2, largest=False).values[:, 1]
        with torch.no_grad():
            pat = model.activation_pattern(X)
        cnt = torch.unique(pat, dim=0, return_counts=True)[1]
        rows.append((name, hidden, float(r.median()), float(nn.median()),
                     float(cnt.float().mean())))

    print(f"  {'세팅':>16s} {'영역 반경':>10s} {'최근접이웃':>11s} {'비':>8s} {'영역당 점':>9s}")
    for name, hidden, rad, nnd, per in rows:
        print(f"  {name + ' h=' + str(hidden):>16s} {rad:10.4f} {nnd:11.4f} "
              f"{rad / nnd:7.2f}배 {per:9.1f}")
    print()
    twod = [r for r in rows if r[0] != "mnist"]
    mn = [r for r in rows if r[0] == "mnist"]
    if twod and mn:
        lo2 = min(r[2] / r[3] for r in twod)
        hi2 = max(r[2] / r[3] for r in twod)
        rm = mn[0][2] / mn[0][3]
        print("  ★ 2D 와 MNIST 가 **정반대**다.")
        print(f"    2D: 영역이 데이터 간격보다 {lo2:.1f}~{hi2:.1f}배 넓다"
              " -> [A_r|b_r] 가 진짜로 이웃을 기술한다.")
        print(f"    MNIST: 영역이 데이터 간격의 **{rm:.2f}배**(1/{1/rm:.0f}) 다"
              " -> 영역 안에 자기 점 하나뿐이고")
        print("           다른 데이터는 **절대 안 들어온다.**")
        print()
        print("    즉 MNIST 에서 [A_r|b_r] 는 '영역 위의 선형함수' 가 아니라")
        print(f"    **그 점 하나에서의 야코비안**이다. {10 * 785:,}개 숫자로 점 하나를 설명하는 셈이고,")
        print("    **설명이 설명 대상보다 길다.**")
        print()
        print("    (표본 60개라 값이 조금 흔들린다. 표본 200개·전체 데이터 기준으로는")
        print("     MNIST 비가 0.067 이었다 — 어느 쪽이든 결론은 같다.)")
    print()
    print("    -> MNIST 영역 클러스터링 = **점별 야코비안 클러스터링**.")
    print("       I3 에서 '영역 클러스터링 = 데이터 클러스터링' 이라고 완곡하게 적었던 것의")
    print("       정확한 뜻이 이것이었다. D3(6만 개를 Pruner 로) 은 해석이 안 된다.")
    print()
    # ---------- 근거 3: 계보 안에서 갈린다 ----------
    print("  " + "=" * 70)
    print("  근거 3 — 인용수가 '영역' 계보 **안에서** 갈린다. 병합만 죽었다.")
    print()
    print(f"  {'무엇을 하는 논문인가':>22s} {'논문':>22s} {'연도':>5s} {'인용':>6s}")
    cite = [
        ("영역 세기 · 이론", "Hanin & Rolnick", 2019, 98),
        ("영역별 피처 귀속", "OpenBox", 2018, 89),
        ("영역 **병합**", "Aletheia", 2020, 20),
        ("영역 **클러스터링**", "polytope lens", 2022, 5),
        ("영역 **클러스터링**", "Clustering-Based Interp.", 2021, 2),
    ]
    for what, who, yr, c in cite:
        print(f"  {what:>22s} {who:>22s} {yr:>5d} {c:>6d}")
    print()
    print("  (OpenAlex, 2026-08 조회. arXiv DOI 레코드는 출판본과 분리돼 낮게 나오므로")
    print("   제목 검색으로 정본을 잡았다.)")
    print()
    print("  ★ '영역' 이 죽은 게 아니라 **'병합' 이 죽었다.** 4~50배 차이는 잡음이 아니다.")
    print("    그리고 2024~2026 논문들(AffineLens, Expressivity Saturation, Region Seeding)도")
    print("    전부 **세기만 하고 병합하지 않는다.**")
    print()

    # ---------- 그래서 무엇이 남는가 ----------
    print("  " + "=" * 70)
    print("  그래서 STEP 6~11 은 버리는가? — **아니다. 무엇이 남는지가 중요하다.**")
    print()
    print(f"  {'STEP 6~11 에서 한 것':>26s}  {'목표가 틀렸으니 버림?':>12s}")
    keep = [
        ("병합 대상으로서의 조각 클러스터", "버림"),
        ("거리 척도 네 개 비교", "**남음** — 무엇을 잴지 고르는 문제는 어디에나 있다"),
        ("λ 같은 손잡이의 경계 찾기", "**남음** — STEP 9 가 보인 '조용히 틀림' 은 일반 현상"),
        ("앵커(정답 아는 데서 먼저 검증)", "**남음** — 선행연구가 필수로 규정"),
        ("랜덤 배정 대조군", "**남음** — 곡선 하나는 아무 말도 안 한다"),
        ("greedy = 상계임을 명시", "**남음** — 알고리즘 탓/모델 탓 구분"),
        ("비용함수 함정(홑원소 블록)", "**남음** — 대칭·퇴화 케이스 함정"),
    ]
    for k, v in keep:
        print(f"  {k:>26s}  {v}")
    print()
    print("  ▶ **틀린 것은 '무엇을 묶을까' 였지 '어떻게 검증할까' 가 아니었다.**")
    print("    param-decomp 핸드북이 가르치는 것이 정확히 뒤쪽이고, 그건 대상이 바뀌어도 그대로 쓴다.")
    print()
    print("  ▶ 그리고 STEP 8~11 은 이미 균열을 보여주고 있었다 — 우리가 못 읽었을 뿐이다:")
    print("      STEP 8  네 척도 중 둘이 앵커를 통과 못 했다")
    print("      STEP 9  손잡이를 조금 돌리자 greedy 가 조용히 틀렸다")
    print("      STEP 11 랜덤 배정 대비 이득이 겨우 1.67배였다")
    print("    **방법이 대상을 감당 못 한다는 신호가 계속 나오고 있었다.**")
    print()
    print("  ▶ 다음 STEP 13~14 가 '그럼 무엇이 이겼나' 를 보인다.")
    print("    미리 한 줄로: **묶는 게 아니라 쪼개는 것**이었다.")


# ================================================================ STEP 13
#
# 2026-08-19 문헌 조사 후속. polytope lens 원문이 우리 조견표를 'k=1' 이라 규정하고
# "decomposable description 을 찾는 데 명백히 최적이 아니다" 라고 적어뒀다.
# 2026년 MFA 논문이 'k>1' 로 SAE 를 크게 이겼다.
# 새 결과를 내려는 게 아니라, 왜 그 계열이 이겼는지를 우리 4조각에서 확인한다.


def step13() -> None:
    head(13, "조견표는 'k=1' 이다 — 4조각은 사실 유닛 기여 2개가 만든 것")

    legend("A_r", "b_r", "[A_r | b_r]", "r", "K", "Ω", "ε")
    codes, A, b, X, y, inv = _toy_pieces()

    print("  STEP 6~11 에서 우리는 조각 4개를 '서로 다른 4개' 로 다뤘다.")
    print("  묶을지 말지만 고민했지 **그 4개가 어디서 왔는지**는 안 물었다.")
    print()
    print("  선행연구가 우리 방식에 붙인 이름이 있다 — polytope lens 원문:")
    print()
    print("    \"클러스터링은 k-sparse 피처를 찾는 것으로 볼 수 있는데, 거기서 k = 1 이다.")
    print("     N개 클러스터를 찾는 것은 N개 기저 방향 중 **한 번에 하나만 켜질 수 있는**")
    print("     과완비 기저를 찾는 것과 같다. 이건 신경망의 분해 가능한 설명을 찾는 데")
    print("     **명백히 최적이 아니다** — 이상적으로는 k > 1 을 허용해야 한다.\"")
    print()
    print("  우리 조견표는 입력 하나당 정확히 한 줄을 쓴다. 그게 k=1 이다.")
    print("  그러면 k>1 은 무엇인가? 4조각에서 바로 보인다.\n")

    # ---- 항등식에서 유도 ----
    print("  " + "-" * 70)
    print("  STEP 2 에서 얻은 항등식을 다시 보자:")
    print()
    print("      A_r = W2 · D(r) · W1        b_r = W2 · D(r) · b1 + b2")
    print()
    print("  여기서 D(r) = diag(r) 이다. 즉 **r 에 대해 선형**이다.")
    print("  선형이면 쪼갤 수 있다 — 유닛 하나씩 떼어서 더하면 된다:")
    print()
    print("      [A_r | b_r]  =  base  +  Σ_i  r_i · c_i")
    print()
    print("      base = [0 | b2]                         (전부 꺼진 상태)")
    print("      c_i  = [ W2[:,i]⊗W1[i,:] | W2[:,i]·b1[i] ]   (유닛 i 가 켜질 때 더해지는 것)")
    print()
    print("  손으로 확인해 보자. 우리 토이는 W1 = I, b1 = [-.5,-.5], W2 = [2,3], b2 = 1.\n")

    base = torch.cat([A[0], b[0:1]])                 # r=00 조각이 곧 base
    c0 = torch.cat([A[2], b[2:3]]) - base            # r=10 - base
    c1 = torch.cat([A[1], b[1:2]]) - base            # r=01 - base
    fmt = lambda t: "[" + ", ".join(f"{v:5.1f}" for v in t.tolist()) + "]"  # noqa: E731
    print(f"    base            = {fmt(base)}      (조각 0 = 코드 00)")
    print(f"    c_0 (유닛0 기여) = {fmt(c0)}      W2[0]=2 만큼 W1 의 0행이 더해진다")
    print(f"    c_1 (유닛1 기여) = {fmt(c1)}      W2[1]=3 만큼 W1 의 1행이 더해진다")
    print()
    print(f"  {'조각':>5s} {'코드':>5s} {'base + r0·c0 + r1·c1':>26s} {'실제 [A_r|b_r]':>22s} {'오차':>9s}")
    worst = 0.0
    for r in range(len(codes)):
        r0, r1 = float(codes[r][0]), float(codes[r][1])
        pred = base + r0 * c0 + r1 * c1
        real = torch.cat([A[r], b[r : r + 1]])
        err = float((pred - real).abs().max())
        worst = max(worst, err)
        code = "".join(str(int(v)) for v in codes[r])
        print(f"  {r:>5d} {code:>5s} {fmt(pred):>26s} {fmt(real):>22s} {err:9.1e}")
    print()
    print(f"  ★ 최대 오차 {worst:.1e} — **정확하다.** 근사가 아니라 항등식이다.")
    print("    조각 4개는 독립적인 넷이 아니었다. **숫자 2개(r_0, r_1)가 만들어낸 것**이다.")
    print()
    # ---- 같은 Ω 예산, 다른 결과 ----
    print("  " + "-" * 70)
    print("  이제 STEP 6 과 나란히 놓아보자. **둘 다 '2개' 를 쓴다.**")
    print()

    def eps_of(part) -> float:
        pred = torch.empty_like(y)
        for blk in part:
            m = torch.isin(inv, torch.tensor(blk))
            w = torch.tensor([float((inv == r).sum()) for r in blk])
            w = w / w.sum()
            pred[m] = X[m] @ (w[:, None] * A[blk]).sum(0) + (w * b[blk]).sum()
        return (pred - y).abs().mean().item()

    eps_k1 = eps_of([[0, 2], [1, 3]])

    # k>1 재구성: base + Σ r_i c_i 로 모든 점을 복원
    pred = torch.empty_like(y)
    for r in range(len(codes)):
        m = inv == r
        v = base + float(codes[r][0]) * c0 + float(codes[r][1]) * c1
        pred[m] = X[m] @ v[:2] + v[2]
    eps_k2 = (pred - y).abs().mean().item()

    print(f"  {'설명 형태':>24s} {'쓰는 것':>18s} {'Ω':>4s} {'ε':>9s}")
    print(f"  {'k=1  조견표 (STEP 6)':>24s} {'대표 [A|b] 2줄':>18s} {2:>4d} {eps_k1:9.4f}")
    print(f"  {'k>1  유닛 기여':>24s} {'base + c_0, c_1':>18s} {2:>4d} {eps_k2:9.4f}")
    print()
    print("  ★★ 같은 2개인데 한쪽은 ε=1.2531, 다른 쪽은 ε=0 이다.")
    print()
    print("    왜 이런 차이가 나나: k=1 은 입력마다 **줄 하나를 고른다**. 2줄뿐이면")
    print("    조각 4개를 2개 값으로 뭉갤 수밖에 없다.")
    print("    k>1 은 입력마다 **조합을 만든다**. 2개 부품으로 4가지 조합이 나온다.")
    print("    부품 h개 -> 조합 2^h 가지. **지수 대 선형**이고, 이게 전부다.")
    print()
    print("  이걸 일반화하면:")
    print(f"    조각(영역) 수      최대 2^h        - 우리 토이 4,  moons h=64 에서 1,378")
    print(f"    유닛 기여 수       h + 1          - 우리 토이 3,  moons h=64 에서 65")
    print()
    print("  ▶ **영역 클러스터링이 왜 파라미터 분해에 밀렸는지가 여기 있다.**")
    print("    우리가 뭘 잘못해서가 아니라, D(r) 이 선형이라는 항등식이")
    print("    k>1 쪽에 공짜로 지수적 이득을 주기 때문이다.")
    print()
    print("  ▶ 그리고 이건 APD 의 첫 번째 성질 **faithfulness** 그 자체다:")
    print("    '컴포넌트들의 합이 원본 파라미터를 정확히 복원한다.'")
    print("    우리는 그걸 학습 없이, 항등식만으로 얻었다. (단, 1층에서만 — STEP 14)")
    print()
    print("  ▶ I4 도 다시 읽힌다. STEP 6 에서 '유닛1이 더 중요하다' 가 나왔는데,")
    print(f"    이제 이유가 눈에 보인다 — c_1 = {fmt(c1)} 이 c_0 = {fmt(c0)} 보다 크다.")
    print("    'causal importance' 는 결국 **‖c_i‖ 가 얼마나 큰가** 였다.")
    print()
    print("  ⚠️ 바뀌지 않는 것: **라우팅**. k>1 로 가도 r 을 알려면 첫 층을 돌려야 한다.")
    print("     Ω 는 줄었지만 오라클 호출은 그대로다. Q1 은 k 와 무관한 별개 문제다.")


# ================================================================ STEP 14
#
# STEP 13 의 가법 분해는 1층에서 유도했다. 깊어지면 어떻게 되는가?
# A_r = W3·D2·W2·D1·W1 — r 에 대해 층을 가로질러 '곱' 이 된다. 가법이 아니다.
# 총 유닛 수를 맞춰 학습해둔 두 MNIST 모델이 이 질문의 대조군으로 이미 준비돼 있다.


def _additive_residual(model, patterns: torch.Tensor, m: int = 1500,
                       seed: int = 0) -> tuple:
    """[A_r|b_r] 를 base + Σ r_i·c_i 로 최소제곱 맞춤한 뒤 상대 잔차를 돌려준다.

    1층이면 항등식이라 잔차가 부동소수점 수준이어야 한다.
    최소제곱으로 푸는 이유: base·c_i 를 가중치에서 직접 만들지 않고
    '데이터가 그 구조에 맞는지' 만 묻기 위해서다 (구조를 가정하지 않는 검사).
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(patterns), generator=g)[:m]
    P = patterns[idx]
    with torch.no_grad():
        A, b = model.effective_affine(P)
    M = torch.cat([A.reshape(len(P), -1), b], dim=1).double()      # (m, C_out·(d+1))
    R = P.double()
    X = torch.cat([torch.ones(len(R), 1, dtype=torch.float64), R], dim=1)  # (m, 1+U)
    sol = torch.linalg.lstsq(X, M).solution
    rel = ((X @ sol - M).norm() / M.norm()).item()
    return rel, M.shape[1], P.shape[1], len(P)


def step14() -> None:
    head(14, "깊이가 합성성을 깨뜨린다 — 그리고 그게 SPD 가 존재하는 이유다")

    legend("A_r", "b_r", "[A_r | b_r]", "r", "C_out", "Ω", "ε")

    print("  STEP 13 의 가법 분해는 **1층에서** 유도했다. 2층이면 어떻게 되나?")
    print()
    print("      1층:  A_r = W2 · D(r) · W1                 -> D(r) 이 한 번만 들어간다")
    print("      2층:  A_r = W3 · D2(r) · W2 · D1(r) · W1   -> D 가 두 번, 사이에 W2 가 낀다")
    print()
    print("  2층에서는 r 의 성분들이 **곱해진다**. r_i·r_j 항이 생기므로 가법이 아니다.")
    print("  그런데 이건 계산으로 확인할 문제다. 우리에겐 대조군이 이미 있다:")
    print("  **총 유닛 수를 128 로 맞춰 학습해둔 MNIST 모델 두 개** (stage1_log §1).")
    print("  유닛 수가 같으니 차이가 나면 원인은 **깊이 하나로 특정된다.**\n")

    print("  검사 방법: [A_r|b_r] 를 base + Σ r_i·c_i 로 **최소제곱 맞춤**하고 잔차를 본다.")
    print("  (가중치에서 c_i 를 직접 만들지 않는다 — 구조를 가정하지 않고 '맞는지' 만 묻는다.)\n")

    print(f"  {'모델':>18s} {'유닛':>5s} {'영역':>9s} {'[A|b] 크기':>11s} "
          f"{'자유도':>7s} {'상대 잔차':>11s}  판정")
    got = {}
    for hidden in ([128], [64, 64]):
        tag = "x".join(map(str, hidden))
        try:
            model, _ = load_model("mnist", hidden, 0, device="cpu")
            d = torch.load(f"artifacts/regions/mnist_h{tag}_s0_train.pt",
                           map_location="cpu")
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] mnist h={hidden}: {type(e).__name__} — {e}")
            continue
        rel, dim, U, m = _additive_residual(model, d["patterns"])
        got[tag] = (rel, dim, U, len(d["patterns"]))
        ok = "가법 성립" if rel < 1e-6 else "**깨짐**"
        print(f"  {'mnist h=' + str(hidden):>18s} {U:5d} {len(d['patterns']):9,d} "
              f"{dim:11,d} {1 + U:7d} {rel:11.2e}  {ok}")
    print()

    if "128" in got and "64x64" in got:
        r1, dim, U, R1 = got["128"]
        r2 = got["64x64"][0]
        print(f"  ★ 1층은 {r1:.1e} — **부동소수점 수준**이다. 항등식이 실수 가중치에서도 성립한다.")
        print(f"    2층은 {r2:.1e} — 여섯 자릿수 차이로 깨진다. **유닛 수는 똑같이 128 인데도.**")
        print()
        print(f"  숫자로 본 압축 (1층, ε=0 을 유지하면서):")
        print(f"    k=1 조견표로 ε=0 을 내려면 줄이 {R1:,}개 필요하다  "
              f"-> {R1 * dim:,} 개 숫자")
        print(f"    k>1 유닛 기여는 {1 + U}개면 된다                    "
              f"-> {(1 + U) * dim:,} 개 숫자")
        print(f"    **{R1 / (1 + U):.0f}배 압축, 오차 0.**")
        print()
    print("  ▶ 이 STEP 의 핵심 — **깊이 ≥ 2 에서는 '공짜 분해' 가 없다.**")
    print("    1층에서는 분해가 항등식으로 그냥 나온다. 2층부터는 나오지 않는다.")
    print("    그러면 어떻게 하나? **분해를 학습한다.**")
    print("    → 그게 APD → SPD → VPD 가 하는 일이고, **Stage 2 의 존재 이유**다.")
    print("      Stage 2 를 '다음에 할 다른 주제' 가 아니라 **여기서 막힌 것의 해법**으로 읽어야 한다.")
    print()
    print("  ▶ I2 (깊이 축) 에 새 의미가 붙는다.")
    print("    깊이는 통제 변수가 아니라 **합성성을 깨뜨리는 변수**다.")
    print("    spiral 에서 2층이 1층을 이긴 것(95.5% -> 100%)과 같은 원인일 수 있다:")
    print("    층을 가로지르는 곱이 표현력을 주고, **그 대가로 분해 가능성을 가져간다.**")
    print("    표현력과 해석가능성의 교환이 이 한 줄에 들어 있다.")
    print()
    print("  ▶ 출구가 둘 있고, 둘 다 실재하는 연구 갈래다.")
    print()
    print("      (A) **분해를 학습한다** — APD -> SPD -> VPD (Stage 2).")
    print("          활성함수는 그대로 두고, 어떤 파라미터 조각이 함께 쓰이는지를 학습으로 찾는다.")
    print()
    print("      (B) **분해가 정확해지도록 활성함수를 바꾼다** — bilinear MLP")
    print("          (Pearce et al., **ICLR 2025 Spotlight**, arXiv:2410.08417).")
    print("          element-wise 비선형성을 아예 없애면 MLP 가 3차 텐서로 **완전히** 표현되고,")
    print("          고유분해가 층의 계산과 **정확히 등가**인 분해를 준다.")
    print("          데이터 없이 **가중치만으로** 된다 -> STEP 12 의 근거 2(작은 정의역)가 통째로 사라진다.")
    print("          그리고 GELU/softmax 문제도 없다 -> 근거 1도 사라진다.")
    print()
    print("      그 논문 서론이 우리 프로젝트 제목과 같은 문장으로 시작한다:")
    print("        \"신경망에서 MLP 가 어떻게 계산하는지에 대한 메커니즘 수준의 이해는 아직 없다.")
    print("         ... MLP 는 그동안 해석가능성 연구에서 **분해 불가능한 부품으로 취급돼 왔다.**\"")
    print()
    print("  ▶ 왜 2D 로는 이 실험을 못 하나 — I3 에 붙는 세 번째 근거")
    print("    moons 의 증강행렬은 C_out×(d+1) = 2×3 = 6 개 숫자뿐이다.")
    print("    자유도가 h+1 = 65 여도 6 을 넘을 수 없으니 **구속력이 없다.**")
    print("    시험이 성립하려면 C_out×(d+1) > h 여야 하고, MNIST 는 7,850 > 128 이다.")
    print("    → '2D 는 그리기 쉬워서' 가 아니라 **잴 수 있는 것이 서로 다르다.**")


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


def _load_setting(name: str, hidden: list[int]) -> tuple | None:
    """(model, dataset, test_acc) 를 얻는다. 체크포인트가 없으면 None.

    STEP 0은 '실제 학습된 모델'의 shape과 정확도를 보여주는 것이 요점이라
    임의의 숫자를 지어내지 않는다. 없으면 그 세팅만 빠진다.
    """
    try:
        model, _ = load_model(name, hidden, 0, device="cpu")
        ds = get_dataset(name, seed=0)
        with torch.no_grad():
            acc = (model(ds.x_test).argmax(1) == ds.y_test).float().mean().item()
        return model, ds, acc
    except Exception as e:  # noqa: BLE001 — 없는 세팅은 조용히 건너뛴다
        print(f"  [skip] {name} h={hidden}: {type(e).__name__} — {e}")
        return None


def main() -> None:
    ensure_dirs()
    setup_matplotlib()
    torch.set_printoptions(precision=3, sci_mode=False)

    models = {}
    for key, hidden in [("moons", [64]), ("spiral", [32, 32]), ("mnist", [128])]:
        got = _load_setting(key, hidden)
        if got is not None:
            models[key] = got

    step0_task(models)
    step0_params(models)
    step0_dimension()
    step0_objective()

    step1()
    step2()
    step3()

    model = models["moons"][0]
    step4(model)
    step5()
    step6()
    step7()
    step8()
    step9()
    step10()
    step11()
    step12()
    step13()
    step14()

    outs = [
        task_figure(models),
        architecture_figure(models),
        dimension_figure(),
        buildup_figure(model),
    ]
    print(f"\n{RULE}\n그림 저장:")
    for o in outs:
        print(f"  {o}")
    print(RULE)


if __name__ == "__main__":
    main()
