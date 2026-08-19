# 로컬 세션 인계 매뉴얼 — 선행연구 조사 이어가기

> **이 문서 하나만 들고 로컬에서 세션을 시작할 수 있도록 쓴다.**
> 원격(GitHub) 세션에서 네트워크가 막혀 못 끝낸 조사를 로컬 RTX 4060 머신에서 이어가는 절차.
>
> 조사 결과 본문: `docs/prior_work.md`
> 작성: 2026-08-19 (GitHub 원격 세션)

---

## 0. 왜 이 문서가 있나 — 원격 세션의 네트워크 제약

GitHub 원격 세션은 egress 프록시 뒤에 있고, **논문 호스트가 거의 다 막혀 있다.**
직접 확인한 결과다 (추측 아님):

| 도메인 | 결과 |
|---|---|
| `arxiv.org`, `export.arxiv.org` | ❌ CONNECT tunnel failed, 403 |
| `www.semanticscholar.org`, `api.semanticscholar.org` | ❌ 차단 |
| `proceedings.neurips.cc`, `proceedings.mlr.press` | ❌ 차단 |
| `openreview.net`, `huggingface.co`, `alphaxiv.org` | ❌ 차단 |
| `www.alignmentforum.org`, `deepai.org` | ❌ 차단 |
| **`github.com`, `raw.githubusercontent.com`** | ✅ 열림 |
| **`pypi.org`, `files.pythonhosted.org`** | ✅ 열림 |

→ 그래서 **GitHub에 코드나 논문이 올라와 있는 자료만 원문 확인이 됐다.**
   (`goodfire-ai/param-decomp`는 논문을 마크다운으로 동봉하고 있어서 전문을 읽을 수 있었고,
   Aletheia는 PyPI 휠에서 심볼을 뽑아 확인했다.)

→ **로컬에서는 이 제약이 없다. arxiv PDF를 그냥 열면 된다.**

---

## 1. 로컬에서 처음 5분

```bash
cd ~/research_github            # 각자 경로
git fetch origin claude/research-status-check-vpg4bl
git checkout claude/research-status-check-vpg4bl
git pull origin claude/research-status-check-vpg4bl

# 이 문서와 조사 결과 확인
less docs/prior_work.md
```

환경은 그대로다 (`uv`, Python 3.12, `.venv`). 이 단계에서 새로 설치할 것은 없다.

---

## 2. 받아야 할 자료 — 우선순위 순

`.gitignore`에 `external/*/`가 이미 있으므로 **`external/` 아래에 두면 추적되지 않는다.**
논문 PDF도 같은 원칙으로 `external/papers/`에 둔다.

```bash
mkdir -p external/papers
```

### 우선순위 A — Stage 1(지금 하는 것)에 직접 영향

| # | 자료 | 받는 법 | 왜 |
|---|---|---|---|
| A1 | **Sudjianto et al. 2020, Unwrapping the Black Box of Deep ReLU Networks** | `https://arxiv.org/pdf/2011.04041` | D1의 절편 가중치, D2의 병합 기준. **코드는 봤지만 논문 서술을 못 봄** |
| A2 | **Black et al. 2022, Interpreting NNs through the Polytope Lens** | `https://arxiv.org/abs/2211.12312` (또는 alignmentforum 전문) | Frobenius를 정의만 했는지 실제로 썼는지 |
| A3 | **Hanin & Rolnick 2019, Deep ReLU Networks Have Surprisingly Few Activation Patterns** | `https://arxiv.org/abs/1906.00904` | 우리 선분 테스트(P5, I1)의 표준 프로토콜 |

### 우선순위 B — README 선행연구 목록의 나머지

| # | 자료 | 받는 법 |
|---|---|---|
| B1 | Srivastava et al. 2015, Understanding Locally Competitive Networks | `https://arxiv.org/abs/1410.1165` |
| B2 | Chu et al. 2018, Exact and Consistent Interpretation for PLNN: A Closed Form Solution (OpenBox, KDD'18) | `https://arxiv.org/abs/1802.06259` |

### 우선순위 C — 이미 원문 확보됨, 로컬에도 두면 편한 것

```bash
# 이 세션에서 실제로 읽은 것들. 로컬에 그대로 복제하면 재확인이 쉽다.
git clone --depth 1 https://github.com/goodfire-ai/param-decomp.git external/param-decomp
git clone --depth 1 https://github.com/SelfExplainML/Aletheia.git   external/Aletheia
```

**`external/param-decomp` 안에서 볼 곳 (줄번호는 2026-08-19 시점 HEAD 기준):**

| 파일 | 줄 | 내용 |
|---|---|---|
| `docs/handbook.md` | 82 | "재구성은 타깃 자신의 출력 척도로 채점" (D4의 근거) |
| `docs/handbook.md` | 124 | 서브컴포넌트 클러스터링 요약 (D2) |
| `docs/handbook.md` | 61 | **앵커는 필수** (Q2의 답) |
| `docs/handbook.md` | 93, 95 | Pareto front 공정성 조건, dense/chance 종점 |
| `papers/Interpreting_Language_Model_Parameters.md` | **1758–1832** | **MDL 클러스터링 부록 전문** — 비용함수, 병합비용 ΔL, 확률적 선택, 정지규칙, α 고르는 법 |
| `papers/Stochastic_Parameter_Decomposition/spd_paper.md` | 110 | "$D$는 출력공간 divergence, LM은 KL, 아니면 MSE" |
| `papers/Attribution_based_Parameter_Decomposition/apd_paper.md` | 25–40 | faithfulness / minimality / simplicity 세 성질 |
| `papers/Attribution_based_Parameter_Decomposition/apd_paper.md` | 74–80, 474–480 | top-k 어트리뷰션 (I4와 연결) |

**Aletheia 구현체를 다시 확인하고 싶을 때** (레포에는 예제만 있고 소스는 PyPI 휠에 있다.
컴파일된 Cython이라 `.py`가 없으므로 `strings`로 본다):

```bash
cd external && python -m pip download aletheia-dnn --no-deps -d ale_whl
unzip -o ale_whl/aletheia_dnn-1.3.5-cp37-none-manylinux_2_17_x86_64.whl -d ale_pkg
strings -n 3 ale_pkg/aletheia/merge.cpython-37m-x86_64-linux-gnu.so | less   # Merger
strings -n 3 ale_pkg/aletheia/prune.cpython-37m-x86_64-linux-gnu.so | less   # Pruner ★
strings -n 3 ale_pkg/aletheia/flatten.cpython-37m-x86_64-linux-gnu.so | less # flatten (Q1)
```

---

## 3. 각 자료에서 뽑을 것 — 체크리스트

**읽고 나서 답을 `docs/prior_work.md`의 해당 절에 적고, 근거 등급을 [요약] → [원문]으로 올린다.**
답이 우리 잠정안과 다르면 `docs/open_questions.md`도 함께 고친다.

### A1. Aletheia 논문 (2011.04041)

- [ ] 병합할 LLM 쌍을 고르는 **거리 정의가 논문에 수식으로 나오는가?** 계수만인가 절편 포함인가
- [ ] 절편(intercept)에 **가중치를 곱하는가**. 곱한다면 값은 어떻게 정했나 → **D1의 λ에 직결**
- [ ] 클러스터링 전에 **표준화(standardize)를 하는가**
- [ ] `n_neighbors`(연결성 그래프) 기본값의 근거. 예제는 `LLM 수 × 0.01`
- [ ] 병합 후 **refit**을 하는 이유를 뭐라고 설명하는가. refit 없는 버전과 비교한 표가 있는가
- [ ] 실험한 데이터셋의 **영역당 평균 샘플 수** — 우리 R/N과 비교하기 위해
- [ ] `Pruner`(topk core + 최근접)와 `Merger`(agglomerative)의 **성능 격차 표가 있는가** → **D3에 직결**
- [ ] `flatten`(병합 결과를 1층 망으로)의 성능이 원본 대비 얼마나 떨어지는가 → **Q1에 직결**

### A2. polytope lens (2211.12312)

- [ ] Frobenius norm 유사도를 **실제 실험에서 썼는가, 정의만 하고 활성 위에서 클러스터링했는가**
      → 후자라면 **"Frobenius를 실제로 쓴 선행연구는 없다"** 가 되고, D1의 서술을 고쳐야 한다
- [ ] HDBSCAN을 무엇 위에 돌렸나 (활성 / spline code / affine map)
- [ ] UMAP 같은 차원축소를 먼저 하는가
- [ ] "monosemantic polytope"을 **어떻게 검증**했나 (사람 라벨? 자동?)
- [ ] 폴리토프 개수 규모를 어떻게 다뤘나 (서브샘플? 층별?)

### A3. Hanin & Rolnick (1906.00904)

- [ ] 직선/평면을 **몇 개, 어떻게** 뽑나 (데이터 두 점 잇기? 랜덤 방향? 격자?)
- [ ] 교차 횟수를 어떻게 세나 (스텝 수, 수치 허용오차)
- [ ] 보고 단위가 무엇인가 (단위 길이당 영역 수? 총 개수?)
- [ ] **학습 전후로 영역 밀도가 어떻게 변하나** → Q3(정규화가 ε-path를 낙관적으로 만드는가)의 대조군
- [ ] Q4(격자 실측 10 vs 공식 11 등, 격차가 h에 따라 커짐)에 대한 설명이 있는가
      — 학습이 유닛을 정렬시켜 퇴화 배치를 만든다는 관찰이 이 논문에 있는지

### B1. Srivastava 2015

- [ ] README가 "활성 코드 클러스터링"이라 적어뒀는데, **실제로 클러스터링인가 검색용 해싱인가**
- [ ] 활성 패턴 간 거리를 뭘로 쟀나 (Hamming?)
- [ ] "데이터 포인트에서 영역을 샘플링" — `concepts.md` §1.4가 이 논문을 인용하는데, **맞는 인용인지 확인**

### B2. Chu et al. 2018

- [ ] README 목록에 있는데 이번에 아무것도 확인 못 함. **읽고 한 줄 요약을 남기거나, 목록에서 뺀다**

---

## 4. 조사가 끝나면 확정할 것 (결정 템플릿)

`docs/open_questions.md`의 [미결정 사항]을 아래 형식으로 바꿔 적는다.
**각 항목에 "선행연구 근거"와 "우리가 다르게 가는 부분"을 반드시 함께 적는다.**

```
### D1. 거리 척도  →  확정: logit 기준 / Frobenius·cosine 대리
근거: SPD(출력공간 divergence), polytope lens(Frobenius)
우리 추가분: row-wise cosine (선행연구 근거 없음), 절편 가중치 λ 민감도
```

확정해야 할 목록 (`docs/prior_work.md` §7 표가 제안):

- [ ] **D1** — logit 기준 유지. `[A_r | λ·b_r]`의 **λ를 무엇으로 둘지** (1.0 고정? 민감도 측정?)
- [ ] **D2** — agglomerative(average) + kNN 연결성 + **확률적 선택 γ=0.2**.
      **refit을 쓸지 말지** (2D는 가능, MNIST는 영역당 1점이라 불가 → 세팅별로 다르게 갈지)
- [ ] **D3** — `Pruner` 방식(topk core + 최근접)으로 **서브샘플 없이** 6만 개 전부 처리.
      2D에서 Merger 대비 격차를 먼저 측정
- [ ] **D4** — **KL 하나를 기준으로.** 정확도는 보조 표시, 선택 기준 아님
- [ ] **곡선** — K=R, K=1 양 끝점 + **랜덤 초기화 모델 대조군** 포함
- [ ] **보고** — 손잡이(α 또는 K, γ, linkage, λ) 없이 "K개로 압축된다"고 쓰지 않기

---

## 5. 그다음 실행 순서

선행연구가 요구하는 순서다 (`prior_work.md` §6.2 — 앵커는 필수).
**순서를 바꾸지 말 것.** 앵커를 건너뛰면 "구조가 없다"와 "코드가 틀렸다"를 구분할 수 없다.

```
1. 앵커     4조각 장난감(walkthrough STEP 6)에 클러스터링 코드를 돌려
            전수탐색 최적 {0,2}|{1,3}을 복원하는지 확인          ← 통과 못 하면 여기서 멈춤
2. D1 측정  2D 실모델에서 세 척도(logit/Frobenius/cosine)가
            같은 답을 주는지. λ 민감도도 여기서
3. D2 측정  2D에서 (a) greedy vs 확률적 선택(γ=0.2) 곡선 격차
                    (b) Merger vs Pruner 격차
                    (c) 소형 모델(h=6~8, 조각 20~30개)에서 전수탐색 vs greedy
4. ε-path   2D 4개 모델에 대해 K=R → K=1 전체 곡선 (KL 기준) + 대조군
5. MNIST    Pruner 방식으로 6만 개 → K개. 평균 대표값(refit 불가)
```

각 단계 전에 **무엇을·왜·어떤 결과를 예상하는지 먼저 적고 확인받는다** (CLAUDE.md 규칙 1).
예측을 문서에 먼저 적고 실행한다 (검증 규칙). 대조군을 함께 둔다.

---

## 6. 로컬 세션 시작용 프롬프트 (복붙)

```
docs/open_questions.md 와 docs/prior_work.md 를 읽어줘.
선행연구 조사에서 원문을 못 읽어 [요약] 등급으로 남은 항목이 있다.
docs/local_session_manual.md §2 우선순위 A의 논문 3개를 받아서
§3 체크리스트의 질문에 답하고, 그 결과로 prior_work.md 의 근거 등급을 올려줘.
답이 우리 잠정안과 다르면 open_questions.md 도 함께 고치고,
무엇이 어떻게 바뀌었는지 요약해서 알려줘. 코드는 아직 쓰지 마.
```

조사가 끝난 뒤 코드 단계로 넘어갈 때:

```
prior_work.md §7 표와 local_session_manual.md §4 를 근거로 D1~D4를 확정하자.
확정되면 §5 실행 순서의 1단계(4조각 앵커)부터 시작하는데,
CLAUDE.md 규칙대로 무엇을·왜·어떤 결과를 예상하는지 먼저 설명하고 확인받아줘.
```
