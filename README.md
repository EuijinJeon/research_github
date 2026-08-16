# 학습된 MLP의 웨이트를 이해 가능한 형태로 번역하기

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

- Srivastava et al. 2015 — 활성 코드 클러스터링
- Black et al. 2022 — polytope lens
- Chu et al. 2018 — PLNN의 국소 선형 분류기
- Elhage et al. 2022 — Toy Model of Superposition
- Nanda et al. 2023 — modular addition grokking / 푸리에 회로
- goodfire-ai/param-decomp — SPD (Stochastic Parameter Decomposition)
