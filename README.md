# 학습된 MLP의 웨이트를 이해 가능한 형태로 번역하기

---

# 여기부터 보세요

## 1. 명령 하나 — 개념 전체

```bash
.venv/bin/python scripts/walkthrough.py
```

유닛 1개에서 시작해 조각 합치기까지 7단계. **모든 숫자가 손으로 검산됩니다.**
읽는 버전: [`docs/walkthrough_stage1.md`](docs/walkthrough_stage1.md)

| STEP | 무엇을 보이나 | 핵심 한 줄 |
|---|---|---|
| 1 | ReLU 유닛 하나 = 칼질 한 번 | `w`가 칼날 방향, `b`가 위치 |
| 2 | 유닛 2개 = 4조각, 조각마다 행렬 하나 | **ReLU는 `W2`의 열을 골라내는 스위치** |
| 3 | 유닛 3개 = 8조각이 아니라 7조각 | 기하학적으로 불가능한 코드가 있다 |
| 4 | 유닛 64개 → 조각 1,378개 (`2^64` 아님) | **`2^h`는 완전히 틀린 직관** |
| 5 | MNIST에서는 왜 다시 `2^128`인가 | 784차원은 공간이 남아돌아 제약이 사라짐 |
| 6 | 4조각의 ε-path 전체 (전수 탐색) | **ε 최소화만으로 causal importance가 나옴** |
| 7 | 세 거리 척도가 다른 답을 준다 | logit이 기준, 나머지는 대리 지표 |

**기호가 헷갈릴 때** — 정의 원본은 [`docs/symbols.md`](docs/symbols.md) 한 곳에 있습니다.
그리고 HTML 버전을 만들면 **기호에 마우스를 올렸을 때 정의가 뜹니다**:

```bash
uv run python scripts/build_docs.py && xdg-open artifacts/docs_html/walkthrough_stage1.html
```

터미널만 볼 때도 각 표 앞에 그 표의 기호 정의가 먼저 출력됩니다.
문서·HTML 툴팁·터미널 범례 셋 다 `docs/symbols.md` 하나를 읽으므로 정의가 어긋날 수 없습니다.

## 2. 그림 여덟 장 — 문제 정의부터 증거까지

`artifacts/figures/` (레포에 포함되어 있음, 재생성 불필요)

**문제 정의** (walkthrough STEP 0 — 먼저 읽을 것)

| 그림 | 무엇을 보여주나 |
|---|---|
| `step0_task.png` | **어떤 태스크인가.** moons / spiral / MNIST의 입력·정답과 학습된 결정 경계 |
| `step0_architecture.png` | **파라미터가 어디에 붙어 있는가.** W1·b1·W2·b2의 shape와 역할 |
| `step0_dimension.png` | **왜 하필 2차원인가.** 초평면 5개를 그대로 두고 d만 바꾸면 6 → 16 → 26조각 |

**증거**

| 그림 | 무엇을 보여주나 |
|---|---|
| `walkthrough_buildup.png` | 유닛 1→64개일 때 평면이 쪼개지는 과정 |
| `polytopes_moons.png`, `polytopes_spiral.png` | 학습된 모델의 실제 폴리토프 + 첫 층 직선 겹침 + 결정 경계 |
| `zero_sets_moons.png`, `zero_sets_spiral.png` | 1층 유닛은 전역 직선, 2층 유닛은 꺾인 선 |

## 3. 검증 결과 — 예측을 먼저 적고 맞췄는지

[`docs/stage1_log.md`](docs/stage1_log.md). 예측 10개 중 9개 PASS, 1개 부분 PASS.

**구조 검증 (실모델 6개)**

| | 예측 | 결과 |
|---|---|---|
| P1 | 1층 영역 경계는 `W1`의 직선 위에만 있다 | **100.00 %** |
| P2 | 2층은 그 비율이 낮다 (나머지는 둘째 층 경계) | 51 % / 59 % |
| P3·P4 | Zaslavsky 상한 / 첫 층 상한과 일치 | 1,326 ≤ 2,081 / 1,881 > 529 |
| P5 | 영역은 깊이 무관하게 볼록 (784차원 포함) | **위반 0** |
| P6 | 2층 유닛의 꺾임은 첫 층 직선 위에서만 | **100.00 %** (대조군 대비 15배 분리) |

**방법 검증 (4조각 앵커, 손으로 검산됨)**

| | 예측 | 결과 |
|---|---|---|
| P7 | 해밍 거리는 유닛 중요도를 못 본다 | **PASS** — 정답과 오답에 똑같이 1.000, 고르지 못함 |
| P8 | 절편 가중치 λ가 앵커를 깨뜨린다 | **부분 PASS** — 교차점 λ=√12 는 손계산과 일치하나, **깨지는 건 greedy뿐이고 전수탐색은 안 깨진다** |
| P9 | greedy(average linkage)가 정답을 복원한다 | **PASS** — 유클리드·logit 통과, 해밍·cosine 실패 |
| P10 | 클러스터링이 랜덤 배정을 이긴다 | **PASS** — K=2에서 **1.67배** |

> P8이 절반 틀린 것이 오히려 수확이었습니다. **λ의 위험은 greedy의 위험**이고,
> 이것이 "우리 ε-path는 상계다"라는 서술의 첫 구체적 실례입니다 (STEP 9).

## 4. 다음에 할 일

[`docs/open_questions.md`](docs/open_questions.md) — 현재 위치, 미결정 D1~D4,
겪은 실패 L1~L4, 영감 I1~I4, 열린 질문 Q1~Q4.

[`docs/prior_work.md`](docs/prior_work.md) — **선행연구가 D1~D4에 실제로 뭐라고 답했는지.**
근거 등급([원문]/[코드]/[요약])을 표기해서, 어디까지 믿어도 되는지 함께 적었다.

---

## 나머지 파일은 무엇인가

| | 역할 |
|---|---|
| `docs/concepts.md` | 용어 사전 / 정의 모음. **walkthrough를 읽다 막힐 때 찾아보는 용도** |
| `docs/prior_work.md` | 선행연구가 D1~D4·Q1~Q2에 준 답. 근거 등급 표기 |
| `docs/local_session_manual.md` | 원격 세션에서 못 읽은 논문을 로컬에서 마저 확인하는 절차 |
| `src/mlpinterp/` | 재사용 모듈. 핵심은 `models.py`의 `effective_affine` (A_r 계산)과 `regions.py` |
| `scripts/train.py` | 6개 모델 학습 (2D×2 + MNIST, 각각 1층/2층) |
| `scripts/extract_regions.py` | 활성 패턴 추출 → `artifacts/regions/` |
| `scripts/viz_polytopes.py` | P1~P4 검증 + 폴리토프 그림 |
| `scripts/check_depth.py` | P5~P6 검증 + 영점 집합 그림 |
| `CLAUDE.md` | 작업 규칙. 다음 세션에서 자동으로 읽힘 |

**전부 다시 만들려면:**

```bash
uv run python scripts/train.py && uv run python scripts/extract_regions.py && uv run python scripts/viz_polytopes.py && uv run python scripts/check_depth.py && uv run python scripts/walkthrough.py
```

---

## 목표

MLP는 CNN·Transformer를 포함한 모든 현대 신경망에 들어 있으면서 가장 해석하기 어려운
컴포넌트다. 이 레포는 MLP 하나를 제대로 이해하는 **프로토콜**을 익히고 재현하는 것이 목적이다.
새로운 발견이 아니라 개념 이해와 선행연구 재현이 목적이며, 알려진 안정적인 세팅을 우선한다.

## 수학적 문제 설정

    interpretation P = argmin Ω(P)   s.t.   err(⟦P⟧, f_θ) ≤ ε

- `f_θ` : 학습된 신경망
- `P`   : 그 신경망에 대한 "설명"
- `⟦P⟧` : 설명 P를 실제로 실행했을 때의 함수 (denotation)
- `Ω(P)`: 설명의 복잡도
- `ε`   : 우리가 감수하기로 한 오차

단순한 입출력 일치가 아니라 **내부 상태 대응(simulation relation)** 이 있어야 유효한 설명으로 친다.
`interpretability`란 이 최소 `Ω`가 작다는 것.

따라서 이 프로젝트의 실질적 산출물은 "MLP를 이해했다"가 아니라
**ε ↦ Ω(ε) 곡선을 뽑아내는 재현 가능한 절차**다. 곡선이 나쁘게 나오는 것도 결과다.

## 단계별로 Ω와 ε이 무엇인가

| Stage | 설명 P | ⟦P⟧ (실행하면) | Ω(P) | err |
|---|---|---|---|---|
| 1 | 클러스터 K개 + 대표 affine map `(A_k, b_k)` | `x ↦ A_k(x)·x + b_k(x)` | 클러스터 수 K | 원 모델 대비 정확도 / logit 차이 |
| 2 | 파라미터 컴포넌트 `{P_c}` (Σ P_c = θ) + causal importance | 활성 컴포넌트만 합친 네트워크 | 컴포넌트 수 × 각 rank | faithfulness loss |
| 3 | 푸리에 회로 (주파수 몇 개 + 삼각 항등식) | 손으로 짠 알고리즘 | 주파수 개수 | 재현 정확도 |
| 4 | Stage 1 ↔ Stage 2 대응표 | — | — | 대응이 성립하는가 |
| 5 | 손으로 짠 분류기 | 신경망 없이 실행 | 사람이 읽을 수 있는 규칙 수 | 원본 정확도의 몇 % |

Stage 3이 **"Ω가 실제로 작아지는 사례"의 기준선**이고, Stage 1은 그것과 비교당하는 대상이다.
이 대조가 없으면 Stage 1의 곡선이 좋은지 나쁜지 말할 수 없다.

## 세팅 (Stage 0에서 확정)

Stage 1은 두 데이터 세팅 × 두 깊이로 진행한다.

| 세팅 | 입력 | 아키텍처 | 역할 |
|---|---|---|---|
| A. 2D 합성 (two-moons / spiral) | 2차원 | `2 → 64 → C` | 폴리토프 분할을 **평면에 직접 그려서** 검산 |
| B. MNIST | 784차원 | `784 → h → 10` | 실제성. `A_r`의 각 행이 28×28 이미지로 시각화됨 |

**일반성 검증 축 4개** — 데이터셋 하나만 바꾸는 것으로는 일반성을 보였다고 하지 않는다.

| 축 | 대조 | Stage |
|---|---|---|
| 데이터 | 2D 합성 ↔ MNIST | 1 |
| 깊이 | hidden 1층 ↔ 2층 (경계가 직선 ↔ 꺾임) | 1 |
| 태스크 구조 | 지각적 ↔ 대수적 (modular addition) | 3 |
| 목적함수 | supervised CE ↔ contrastive / self-distillation | 5(b) |

깊이 축이 특히 중요하다. hidden 1층에서는 영역 경계가 전역 초평면(직선)이지만,
2층이 되면 첫 층 경계가 둘째 층 게이트에 의해 조각조각 접혀 **꺾인다**.
프로토콜이 양쪽 모두에서 돌아간다면 "직선 경계라서 됐던 것"이 아님을 보인 셈이고,
비용은 거의 0이다.

## 환경

- 주력: RTX 4060 (Linux, Ubuntu 26.04). 개발과 반복.
- 보조: MacBook M4 Pro. 메모리 큰 작업.
- Python 3.12 (uv 관리). 시스템 Python 3.14는 건드리지 않는다.
- torch/torchvision은 기본 PyPI 인덱스에서 설치 → linux는 CUDA 휠, macOS는 MPS 휠이 자동 선택.
- 코드는 device-agnostic. 체크포인트는 `map_location`으로 저장/로드.
- MPS는 float64 미지원, `torch.compile` 불안정 → 방어적으로 작성.
- wandb는 offline (`WANDB_MODE=offline`).

### 셋업 (두 머신 공통)

```bash
uv sync --extra dev
```

`uv`가 없으면 먼저: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 맥(M4 Pro)에서 이어서 작업할 때

체크포인트는 레포에 들어 있으므로 **재학습이 필요 없다.**
영역 파일(`artifacts/regions/`, 20M)만 용량 문제로 제외했으니 한 번 재생성한다:

```bash
uv run python scripts/extract_regions.py
```

MPS에서 문제가 생기면 코드를 고치지 말고 환경변수로 CPU로 내린다:

```bash
MLPINTERP_DEVICE=cpu uv run python scripts/extract_regions.py
```

### 개념을 따라가는 순서

1. `docs/walkthrough_stage1.md` — 한 걸음씩 쌓아 올리는 설명 (여기부터)
2. `python scripts/walkthrough.py` — 위 문서의 모든 숫자를 직접 출력
3. `docs/concepts.md` — 정의와 용어 사전
4. `docs/stage1_log.md` — 실행 로그, 예측 P1~P6과 결과

## 디렉토리 구조

```
docs/          개념 정리 문서 (concepts.md 등)
src/mlpinterp/ 재사용 모듈
scripts/       단계별 실행 스크립트
artifacts/     중간 산출물 — 뒷단계 재실행 시 앞단계를 반복하지 않기 위함
  checkpoints/   학습된 모델
  regions/       활성 패턴, A_r, 클러스터 할당
  figures/       그림
  logs/          wandb offline run 등
external/      Stage 2에서 클론할 외부 레포 (param-decomp)
```

**중간 산출물은 반드시 디스크에 저장한다.** 뒷단계를 다시 돌릴 때 앞단계를 반복하지 않는 것이
이 프로젝트의 명시적 요구사항이다.

## 진행 원칙

1. 각 단계 시작 전에 무엇을·왜·어떤 결과를 예상하는지 먼저 설명하고 확인을 받는다.
2. 코드는 한 번에 한 파일씩. 긴 파일은 함수 단위로 끊어서 설명하며 작성한다.
3. 새로운 개념은 **코드를 쓰기 전에** 먼저 설명한다.
4. 각 단계 끝에서 "이 단계에서 알게 된 것"을 한 문단으로 정리한다.
5. 결과가 예상과 다르면 넘어가지 않고 왜 다른지 파고든다.
6. 속도보다 이해가 우선.

## 선행연구

각 문헌이 우리 미결정 사항에 어떤 답을 주는지는 [`docs/prior_work.md`](docs/prior_work.md) 참조.

| 문헌 | 우리에게 주는 것 |
|---|---|
| **Sudjianto et al. 2020** — Unwrapping the Black Box of Deep ReLU Networks ([Aletheia](https://github.com/SelfExplainML/Aletheia)) | 영역 병합의 유일한 완성된 구현. `Merger`(agglomerative+kNN 연결성+refit), **`Pruner`(topk core+최근접 → D3의 답)**, `flatten`(→ Q1) |
| Black et al. 2022 — polytope lens | 활성 / **부호벡터 `r` 위의 해밍 거리**로 HDBSCAN (D1의 네 번째 척도). ⚠️ Frobenius는 각주 8의 제안일 뿐 쓰이지 않았다 |
| Hanin & Rolnick 2019 — Deep ReLU Nets Have Surprisingly Few Activation Patterns | **증분 정확 계수법**(꼭짓점 부호 검사 → Q4), 영역 수가 학습으로 거의 안 변한다는 결과(→ Q3 대조군), 볼록성 정리(Lemma 7 = 우리 P5) |
| Srivastava et al. 2015 | 활성 코드(**submask**)의 t-SNE **시각화** + kNN·검색. ⚠️ 클러스터링 알고리즘은 쓰지 않는다. 학습 전/후 대조(Figure 3)가 우리 대조군의 선례 |
| Chu et al. 2018 — OpenBox | PLNN의 국소 선형 분류기. **분해까지만 — 병합·클러스터링·ε 없음** (우리와 여기서 갈라진다) |
| Elhage et al. 2022 — Toy Model of Superposition | Stage 2 배경 |
| Nanda et al. 2023 — modular addition grokking | Stage 3 대조군 |
| **[goodfire-ai/param-decomp](https://github.com/goodfire-ai/param-decomp)** — APD → SPD → VPD | Stage 2 본체. 그리고 **ε 척도(KL/MSE), MDL 클러스터링, 확률적 병합(γ=0.2), "앵커 필수" 규범**의 출처 |

**2024~2026 — 이 계열은 멈추지 않았다** (2026-08-19 검색으로 확인. 자세히는 `prior_work.md` §9)

| 문헌 | 우리에게 주는 것 |
|---|---|
| **From Directions to Regions** (`arXiv:2602.02464`, 2026-02) | ★★ **영역 기반 분해가 SAE를 크게 이긴다** (해석가능 비율 0.96 vs 0.29). 단 이기는 형태는 **soft 영역 + 국소 부분공간 + 여러 컴포넌트 동시 활성** — 우리 `k=1` 조견표가 아니다 |
| **AffineLens** (`arXiv:2605.06218`, 2026-05) | 정확 영역 열거 도구 (BN·pooling·residual·conv 지원). **Q4의 도구가 이미 나와 있다** |
| **Expressivity Saturation** (`arXiv:2606.21687`, 2026-06) | 선분 프로브 + 정확 열거. 과제가 어려울수록 **실현 영역이 줄어든다** — 우리 I2와 겹침 |
| **Region Seeding** (`arXiv:2605.06300`, 2026-05) | 정규화 ↔ 영역 수를 정면으로. **Q3의 "정규화가 단순화한다"와 방향이 반대** |
| **Re3** (Machine Learning, Springer, 2026-01) | 영역별 피처 귀속. OpenBox 계보의 현재형 — 여전히 **병합·ε 없음** |
