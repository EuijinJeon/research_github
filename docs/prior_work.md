# 선행연구가 이미 답한 것 · 아직 답하지 않은 것

> `open_questions.md`의 미결정 사항 D1~D4와 열린 질문 Q1~Q2에 대해,
> 선행연구가 실제로 어떤 선택을 했는지 조사한 결과.
> **이 프로젝트의 목적이 선행연구 재현이므로, 여기 적힌 선택을 기본값으로 삼는다.**
> 우리가 선행연구와 다르게 가는 지점은 그렇게 명시한다.
>
> 조사일: 2026-08-19 (GitHub 원격 세션) →
> **2026-08-19 로컬 세션에서 논문 5편 원문 확인 완료. 등급이 전부 [원문]으로 올랐고,
> 그 과정에서 아래 §2·§4의 서술 두 곳이 원문과 어긋난 것이 드러나 고쳤다.**
> PDF는 `external/papers/`(gitignore됨). 인계 절차는 `docs/local_session_manual.md`

---

## 0. 근거 등급 — 어디까지 확인했는지

1차 조사 환경(GitHub 원격 세션)에서 **arxiv.org를 포함한 대부분의 논문 호스트가 차단**돼 있어
자료마다 확인 깊이가 달랐다. **2026-08-19 로컬 세션에서 미확인분을 전부 받아 읽었다.**
아래 표기를 문서 전체에서 사용한다.

| 표기 | 뜻 |
|---|---|
| **[원문]** | 논문 전문을 읽고 인용함. 신뢰해도 된다. |
| **[코드]** | 논문은 못 읽었지만 **공식 구현체에서 직접 확인**함. API·기본값·알고리즘 이름 수준은 확실. |
| **[요약]** | 검색 요약만 봄. **방향은 맞지만 세부 숫자는 믿지 말 것.** 로컬에서 원문 확인 필요. |

| 자료 | 등급 | 확인 경로 |
|---|---|---|
| APD (Braun et al. 2025) | **[원문]** | `goodfire-ai/param-decomp` 레포에 `papers/`로 동봉 |
| SPD (Bushnaq et al. 2025) | **[원문]** | 〃 |
| VPD (Goodfire, 2026-04) | **[원문]** | 〃 (`Interpreting_Language_Model_Parameters.md`, 2,326줄) |
| param-decomp 핸드북/스킬 | **[원문]** | 〃 `docs/handbook.md`, `docs/skill.md` |
| Aletheia (Sudjianto et al. 2020) | **[원문]** ↑ | `arXiv:2011.04041` §5.1, Algorithm 1–2, Table 1·3 (+ PyPI `aletheia-dnn==1.3.5` 휠) |
| polytope lens (Black et al. 2022) | **[원문]** ↑ | `arXiv:2211.12312` 각주 8, §"monosemantic polytopes" |
| Hanin & Rolnick 2019 | **[원문]** ↑ | `arXiv:1906.00904` §4, Appendix A, Lemma 4·7 |
| Srivastava et al. 2015 | **[원문]** ↑ | `arXiv:1410.1165` §3, Figure 3 |
| Chu et al. 2018 (OpenBox) | **[원문]** ↑ | `arXiv:1802.06259` 초록·§1 |

> **↑ = 2026-08-19 로컬 세션에서 원문으로 승급.** 승급 결과 뒤집힌 내용은 각 절에 ⚠️로 표시했다.

---

## 1. 계열이 두 개이고, 답이 다르다

| | **영역 병합 계열** | **파라미터 분해 계열** |
|---|---|---|
| 대표 | Aletheia (2020), polytope lens (2022) | APD (2025) → SPD (2025) → VPD (2026) |
| 병합 대상 | **입력공간** 영역의 affine map `[A_r\|b_r]` | **파라미터공간** 컴포넌트 |
| 거리 | 계수벡터 유클리드 / Frobenius | **거리 없음** — 출력 divergence로 직접 |
| 알고리즘 | agglomerative(+연결성), HDBSCAN | 학습 + 사후 MDL 계층 병합 |
| ε | AUC / MSE | KL(분류·LM) / MSE(회귀) |
| 상태 | ⚠️ **정정: 멈추지 않았다.** 2024~2026에 세 갈래로 활발 (§9) | 2026년까지 활발, **증거 기준이 문서화됨** |

Stage 1은 첫째 계열, Stage 2는 둘째 계열이다.
**그런데 D1~D4에 대한 더 성숙한 규범은 둘째 계열에 있다.** 첫째 계열은 "이렇게 했다"만
있고, 둘째 계열은 "무엇을 증거로 인정하고 무엇을 인정하지 않는지"까지 적어 놨다.
→ **Stage 1의 방법론도 둘째 계열의 규범을 따르는 것이 낫다.**

> ⚠️ **2026-08-19 2차 정정:** 위 표의 "2020년경 멈춤" 은 **틀렸다.**
> 1차 조사가 차단된 네트워크에서 얻은 인상을 그대로 적었고, 로컬 세션은 이미 알던 논문
> 5편만 받아서 확인할 기회가 없었다. **문헌 검색을 한 번도 안 한 채 "멈췄다"고 단정했다.**
> 실제로는 2024~2026 에 세 갈래로 활발하다. **§9 를 볼 것 — 방향 판단이 바뀐다.**

---

## 2. D1 — 거리 척도

### 선행연구는 셋을 비교하지 않았다. 각자 하나를 골랐다.

| 연구 | 무엇을 썼나 | 등급 |
|---|---|---|
| polytope lens | ⚠️ **각주 8 한 줄이 전부다.** 본문이 아니라 각주에서 *"This could be quantified, **for instance**, as the Frobenius norm of the difference matrix between the implied weight matrices"* 라고 지나가듯 제안하고 **끝.** 실험에서는 쓰지 않는다 | [원문] |
| Aletheia | *"The similarity between two LLMs is defined as the **Euclidean distance between their local linear coefficients and intercepts**, subject to the connectivity constraint."* (§5.1) — ⚠️ **절편 가중치도 표준화도 논문에 없다.** 계수와 절편을 그냥 이어붙인다 | [원문] |
| SPD / VPD | 파라미터 공간 거리를 **아예 쓰지 않음**. 출력 공간 divergence로 직접 채점 | [원문] |
| row-wise cosine | **어디서도 표준이 아님** | — |

### SPD 원문 인용 [원문]

> $\mathcal{L}_{\text{stochastic-recon}}=\frac{1}{S}\sum^S_{s=1} D\left(f(x|W'(x,r^{(s)})), f(x|W)\right)$
> Here, $D$ is some appropriate divergence measure **in the space of model outputs**,
> such as KL-divergence for language models, or MSE loss.

### 우리에게 주는 답

1. **`open_questions.md` D1의 방침("logit이 기준, 나머지는 대리")은 SPD 계열 규범과 일치한다.**
   그대로 간다.
2. ⚠️ **정정 — "Frobenius는 근거가 있다"는 과장이었다.**
   1차 조사는 [요약]만 보고 "polytope lens가 제안한 대리 지표"라고 적었다. 원문을 열어 보니
   `Frobenius`는 **논문 전체에서 딱 한 번**, 그것도 **각주 8**에 *"예를 들어 이렇게 잴 수도 있다"*
   로만 나온다. 실제 실험은 affine map을 아예 건드리지 않고 **활성(activations)** 또는
   **spline code(활성 패턴 이진화)** 위에서 HDBSCAN을 돌린다 (§3.3).
   → **affine map `[A_r|b_r]` 위에서 거리를 재고 클러스터링한 선행연구는 Aletheia 하나뿐이고,
     그 하나는 Frobenius가 아니라 유클리드다.** (증강행렬을 벡터로 펴면 둘은 같은 값이지만,
     "Frobenius를 쓴 선행연구가 있다"고 쓰면 안 된다 — 없다.)
3. **row-wise cosine은 선행연구 근거가 없는 우리만의 축이다.**
   → 재현이 아니라 우리가 추가하는 부분. 그렇게 명시할 것.
4. ⚠️ **정정 — 절편 가중치 λ는 Aletheia "논문"에 없다.** 1차 조사가 코드에서 발견한
   `all_bias_weight`를 근거로 "Aletheia는 이걸 튜닝 대상으로 뒀다"고 적었는데, **논문 §5.1은
   가중치 없이 계수와 절편의 유클리드 거리라고만 쓴다.** 표준화도 언급이 없다.
   → 따라서 λ는 **구현체에만 있는 미문서화 손잡이**이고, 우리가 쓰면 **우리 추가분**이다.
     그래도 쓸 이유는 남아 있다 — `A`의 성분이 `b`보다 크면 절편 차이가 거리에서 사라지고,
     그건 L2(교훈 L2)에서 피하려던 바로 그 문제다.
   → **`[A_r | λ·b_r]`의 λ 민감도를 재되, "선행연구 재현"이 아니라 "우리 추가분"으로 적는다.**
     λ=1이 Aletheia 논문의 선택이므로 **λ=1이 재현 기준선**이다.

---

## 3. D2 — 클러스터링 알고리즘, 그리고 greedy 상계 문제의 실제 해법 ★

### 3.1 Aletheia: agglomerative + 연결성 제약 + refit [원문]

논문 **Algorithm 1**이 전부다. 다섯 줄이고, 순서가 중요하다:

```
Require: 데이터 {x_i, y_i}, 풀어낸 LLM들, K(클러스터 수), T(이웃 수), τ(작은 클러스터 기준)
 1: 각 LLM의 계수·절편 (w̃_P, b̃_P) 와 영역 중심 μ_P 를 모은다
 2: μ_P 기준 T-최근접이웃으로 연결성 행렬을 만든다        ← 입력공간 위치 기준
 3: (w̃_P, b̃_P) 를 연결성 제약 하에 K개로 계층 병합한다     ← 계수공간 기준
 4: 큰 클러스터(샘플 수 > τ)와 작은 클러스터를 가른다
 5: 작은 클러스터를 가장 가까운 큰 클러스터에 흡수시킨다     ← 병합 "후" 처리
 6: 클러스터마다 (정규화된) GLM 을 다시 적합한다
```

원문 기본값: **linkage = average, metric = euclidean**, `sklearn`의 `AgglomerativeClustering`
+ `kneighbors_graph`. **K는 grid search `{1..10, 15, 20}`**, 채점은 AUC.
**T는 `|P_train|`의 1%로 시작해, 모든 LLM이 하나로 연결될 때까지 키운다.** τ 예시값은 30.

네 가지가 우리에게 직접 쓸모가 있고, **그중 둘은 1차 조사가 잘못 읽었던 것이다:**

1. **average linkage + euclidean** — D2에서 유력하다고 적어둔 것과 같다. 검증됨.
2. ⚠️ **kNN 연결성은 "계수 공간"이 아니라 "입력 공간" 위에서 잡는다.**
   이웃 판정에 쓰는 것은 `[A_r|b_r]`가 아니라 **영역 중심** `μ_P = (1/|R_P|)·Σ_{x_i∈R_P} x_i`,
   즉 그 영역에 떨어진 데이터 점들의 평균 위치다 (식 19).
   → 의미가 다르다. 이 제약은 *"계수가 비슷해도 **입력공간에서 멀면** 안 합친다"* 를 강제한다.
     Aletheia가 말하는 **locality(입력공간 근접) + homogeneity(계수 유사)** 의 이중 조건이 이것이다.
   → 우리에게 부수효과: 밀집 거리행렬 O(R²) 대신 희소 그래프라 **D3의 우회로**이기도 하다.
   → **주의:** MNIST에서 μ_P는 영역당 점이 하나뿐이라 **그 점 자체**가 된다.
     즉 연결성 제약이 "입력 이미지끼리의 784차원 최근접이웃"으로 바뀐다. 2D와 의미가 달라진다.
3. ⚠️ **τ(=30)는 사전 필터가 아니라 사후 처리다.** 1차 조사는 코드의 `min_samples`를 보고
   "샘플이 적은 **영역**은 애초에 병합 후보에서 뺀다"고 적었는데, 논문의 순서는 반대다 —
   **먼저 전부 병합하고(3단계), 그 뒤에 작은 "클러스터"를 큰 클러스터에 흡수시킨다(4~5단계).**
   버려지는 영역은 없다. (§4.2에서 이 정정이 결론을 뒤집는다.)
4. **`refit_model` ← 제일 중요하다.**
   병합 후 affine map을 **평균내지 않고, 그 클러스터에 속한 데이터로 GLM을 다시 적합**한다
   (저차원은 단순 GLM, 고차원은 `ℓ1`/`ℓ2` 정규화 GLM).
   → 우리가 상정한 "클러스터 대표 = 평균 `[A|b]`"보다 ε이 **항상 낮게** 나온다.
   → **우리 ε-path와 Aletheia 숫자는 직접 비교 불가.** refit을 넣거나, 안 넣는다면 왜 다른지 적어야 한다.

**우리와 다른 점 하나 더:** Aletheia의 K 선택은 MDL도 ΔL=0 교차점도 아닌 **AUC grid search**다.
즉 **ε(성능)만 보고 K를 고른다** — Ω 쪽 비용을 명시적으로 세지 않는다.
VPD(§3.2)가 그 자리에 MDL을 넣은 것이고, 우리 ε-path는 **고르지 않고 곡선 전체를 보여주는** 쪽이다.

### 3.2 VPD: greedy 국소최소를 벗어나는 구체적 방법 [원문] ★★

`open_questions.md` D2에서 "우리 ε-path는 진짜 최적보다 위에 있는 상계"라고 적어둔 문제에
대해, **선행연구도 최적을 못 찾고, 대신 두 가지로 대응한다.**

#### (a) K를 고르는 대신 MDL을 최소화한다

부분집합 $\{\theta_1,\dots,\theta_k\}$에 대해:

$$\mathcal{L}_{\text{MDL}} = \sum_{i=1}^{k} s_i\left(\log_2(k) + \alpha \cdot r(\theta_i)\right)$$

- $s_i$ = 컴포넌트 $i$가 causally important였던 횟수 (데이터셋 전체 합)
- $r(\theta_i)$ = 그 컴포넌트의 rank
- 첫 항 $\log_2(k)$ = **사전(dictionary) 인덱스 비용** — 컴포넌트가 많을수록 비쌈
- 둘째 항 $\alpha \cdot r$ = **개별 컴포넌트 복잡도 비용** — 뭉칠수록 비쌈
- $\alpha$ = 그 둘의 교환비를 정하는 **해상도 손잡이**

**우리 Ω = K는 이 식에서 둘째 항을 버리고 첫 항만 남긴 특수 케이스다.**
즉 우리는 "클러스터 개수"만 세고 "클러스터 하나가 얼마나 복잡한지"는 안 세고 있다.

#### (b) 정지 규칙: 격자 탐색이 아니라 ΔL = 0 교차점

> We run the hierarchical clustering algorithm **until all subcomponents have been merged into a
> single component.** Then, we find the iteration at which the marginal change in description length
> from merging **$\Delta\mathcal{L}$ crossed $\Delta\mathcal{L}=0$**, and use the clusters at that
> iteration as our components.

→ K를 1부터 20까지 훑는 게 아니라, **K=R에서 K=1까지 한 번 끝까지 병합하면서
전체 곡선을 얻고, 비용이 증가로 바뀌는 지점을 고른다.**
우리 ε-path와 계산 구조가 정확히 같다 — 한 번의 계층 병합으로 곡선 전체가 나온다.

#### (c) greedy 대신 확률적 선택 ★

> Naively, one might greedily select the pair $\arg\min \Delta\mathcal{L}$ ... **but this risks
> getting stuck in local minima.** ... we rank all candidate merge pairs by their cost in ascending
> order and assign each pair a probability that decays exponentially in its rank:
> $P \propto \exp(-\gamma J)$, $J = 0,1,\dots,\binom{k}{2}-1$
>
> 역CDF 샘플링: $J = \left\lfloor \dfrac{-\log(1-u(1-e^{-\gamma N}))}{\gamma} \right\rfloor$, $u\sim U(0,1)$
>
> In our experiments, we use **$\gamma = 0.2$**.

**손으로 따라가 볼 것 — $\gamma=0.2$가 실제로 뭘 하나:**

가중치 $e^{-0.2J}$의 합은 $N$이 클 때 $\frac{1}{1-e^{-0.2}} = \frac{1}{0.1813} = 5.517$ 로 수렴한다.
따라서 각 순위가 뽑힐 확률은

| 순위 $J$ | 0 | 1 | 2 | 3 | 4 | 상위 5개 합 |
|---|---|---|---|---|---|---|
| $e^{-0.2J}$ | 1.000 | 0.819 | 0.670 | 0.549 | 0.449 | — |
| 확률 | **18.1 %** | 14.8 % | 12.1 % | 9.9 % | 8.1 % | **63.2 %** |

즉 **최선의 병합을 고를 확률이 18%뿐이다.** 나머지 82%는 일부러 차선을 고른다.
논문이 "대략 가장 싼 다섯 후보에 의미 있는 확률을 유지한다"고 한 게 이 숫자다.
$\gamma\to\infty$면 greedy, $\gamma\to 0$이면 균등추출이다.

→ **구현은 10줄이면 된다. 우리 ε-path의 상계를 낮추는 가장 값싼 개선.**
   여러 번 돌려서 가장 좋은 곡선을 취하면 되고, **greedy 곡선과 나란히 그리면
   "알고리즘 탓"과 "모델 탓"이 얼마나 분리되는지가 그대로 보인다.**

#### (d) 이진화와 OR [원문]

그룹의 중요도는 멤버들의 **OR**로 정의하고, 개별 중요도는 **$\tau = 0.01$** 로 이진화한다.
논문의 각주가 왜 OR인지 설명한다 — *"n개의 파라미터 벡터가 어떤 조합으로도 ablate 가능하면,
그 합도 ablate 가능함이 보장된다"* — 즉 **보수적인 쪽으로 틀리게 만든 선택**이다.

#### (e) 보고 규범 [원문]

> Clustering is **post-hoc and its resolution is a choice** — do not report a component count
> as a fact about the model **without reporting the knob**.

→ "이 모델은 K개로 압축된다"는 문장은 손잡이($\alpha$, $\gamma$, linkage) 없이는 쓰면 안 된다.
   `open_questions.md` D2의 걱정("압축이 안 되는 것"과 "우리가 못 찾은 것"을 구분해야 한다)에
   대한 선행연구의 답이 이것이다: **구분하려 애쓰는 대신, 손잡이를 함께 보고한다.**

### 3.3 polytope lens: HDBSCAN — 무엇 위에 돌렸는지가 핵심 [원문]

밀도 기반이라 **K를 정하지 않는다.** 그리고 알고리즘 무관성은 **각주 9에 명시**돼 있다:

> While we use HDBSCAN in this work, **the specific algorithm isn't important.** Any clustering
> algorithm that groups together any sufficiently nearby activations or codes should yield
> monosemantic clusters.

**무엇 위에 돌렸나 — 두 가지이고, 둘 다 affine map이 아니다:**

| 대상 | 거리 | 비고 |
|---|---|---|
| **활성 그 자체** (InceptionV1 채널 차원, GPT2-small MLP 층) | 기본 | 주 실험 |
| **spline code** (활성 패턴을 이진화한 것 = 우리 `r`) | **Hamming** | Figure 12 |

두 번째가 우리 것과 가장 가깝지만, 그들이 묶는 것은 **부호벡터 `r`끼리의 해밍 거리**이지
`[A_r|b_r]`가 아니다. 논문 자신이 이걸 "mildly surprising"이라고 부르며 설명한다 —
*"부호벡터 하나가 이미 선형 제약들의 집합을 정의하고, 그 제약들이 활성이 놓일 수 있는 영역을
좁은 범위로 가둔다. 그래서 이진화 후에도 크기 정보가 상당히 남아 있다."*

**차원축소:** PCA/NMF 방향과 비교하는 실험이 따로 있지만, **클러스터링 전에 UMAP 같은 축소를
먼저 걸지는 않는다.**

**단의성(monosemanticity) 검증은 정성적이다** — 클러스터별 데이터셋 예시를 사람이 읽는 방식이고
(Appendix E), 자동 지표는 없다. "majority of clusters found are monosemantic"이 보고 수준이다.

→ **우리에게 주는 답:** polytope lens는 **영역 병합 계열이 아니라 활성 클러스터링 계열**이다.
   `open_questions.md`가 이 논문을 "Frobenius의 출처"로 인용해 왔지만, 실제로 이 논문에서
   가져올 것은 **(a) 알고리즘 선택은 중요하지 않다는 명시적 입장**과
   **(b) 부호벡터 `r` 위의 해밍 거리라는 네 번째 척도**다.
   (b)는 우리 D1의 세 척도에 없던 것이고, `[A_r|b_r]`를 만들 필요조차 없어 **가장 싸다.**
   → **대리 지표 후보에 추가할 가치가 있다.** 2D에서 셋과 함께 재보면 된다.

---

## 4. D3 — MNIST 규모 문제 ★

### 4.1 Aletheia의 두 번째 알고리즘이 답이다 [코드] — **논문에는 없다**

> ⚠️ **출처 주의:** `Pruner`는 **PyPI 구현체에만 있고 논문에는 한 번도 안 나온다.**
> (`arXiv:2011.04041` 전문에서 `prune`은 무관한 문맥의 서론 한 곳뿐.)
> 아래 내용의 등급은 여전히 **[코드]**다. 논문 쪽의 대응물은 §4.2의 Algorithm 1 4~5단계다.

Aletheia에는 `Merger` 말고 **`Pruner`** 가 따로 있다. docstring 전문:

> Initializes the PrunerClassifier object. It selects the **topk largest LLMs as core LLMs**,
> and **all the other LLMs are merged to their nearest core LLMs.**

즉:

```
1. 영역을 샘플 수 기준으로 정렬해 상위 K개를 core로 뽑는다
2. 나머지 영역은 각자 가장 가까운 core에 배정한다
```

**비용이 O(R·K)다. 쌍거리 행렬이 필요 없다.**
`open_questions.md` D3의 "60000² = 36억 → 계산 불가"가 여기서 사라진다.
60,000 × K=100 = 600만 번 거리계산이면 끝이다.

→ **D3의 답: 랜덤 서브샘플 M을 정하는 문제가 아니라, 알고리즘을 바꾸는 문제였다.**
   `Pruner` 방식이면 6만 개 영역을 **하나도 버리지 않고** 전부 배정할 수 있고,
   "표본 M개를 K개로 줄였다"가 아니라 **"영역 6만 개를 K개로 줄였다"** 를 그대로 말할 수 있다.

단, 대가가 있다. `Pruner`는 **core 선택이 병합 품질과 무관**하다(그냥 큰 것부터).
따라서 `Merger`보다 ε이 높게 나올 것이다.
→ **2D에서 `Merger`(agglomerative)와 `Pruner`(core+최근접)를 둘 다 돌려 격차를 재두면,
   MNIST에서 `Pruner`만 쓸 때 얼마를 손해 보는지 보정할 수 있다.**
   이건 D2의 "전수탐색 vs greedy 격차 측정"과 정확히 같은 구조의 실험이다.

### 4.2 ⚠️ **정정 — "τ 필터가 MNIST를 원천 무효화한다"는 틀렸다. 그리고 선행연구에 이미 R/N≈1 사례가 있다** ★★

1차 조사는 이렇게 적었다: *"`min_samples=30`은 우리 MNIST의 모든 영역을 삭제한다.
refit할 데이터가 영역당 1점뿐이므로 refit 자체가 정의되지 않는다."*
**원문을 읽으니 둘 다 틀렸다.** 이유가 각각 다르다.

#### (1) τ는 "영역"이 아니라 "클러스터"에 걸린다 — 순서가 반대였다

> After merging, it is possible that **some clusters** only contain a small number of samples. ...
> we use a **post-hoc processing step** to merge these small clusters to their nearby large clusters.

Algorithm 1의 순서는 **병합(3) → 큰/작은 클러스터 구분(4) → 작은 것을 큰 것에 흡수(5)** 다.
τ가 걸리는 시점에는 이미 병합이 끝나 클러스터가 커져 있다.
**영역 단계에서 버려지는 것은 하나도 없다.** 따라서 R/N=1.000이어도 τ는 아무것도 삭제하지 않는다.

#### (2) refit도 "클러스터" 단위다 — K=100이면 MNIST에서 클러스터당 600점

Algorithm 1의 6단계는 *"클러스터마다 GLM을 다시 적합"* 이다. 영역마다가 아니다.
6만 개 영역을 K=100으로 묶으면 클러스터당 평균 600개 샘플이 있다. **refit은 정의된다.**
(정말 불가능해지는 것은 K가 R에 가까울 때뿐이고, 그건 ε≈0 구간이라 refit이 필요 없는 구간이다.)

#### (3) 결정적 증거 — Aletheia의 FicoHeloc이 이미 R/N ≈ 0.94다 ★

Table 1을 우리 `stage1_log.md` §2 형식으로 옮기면 (학습 80% 기준):

| Aletheia 데이터셋 | 차원 | 학습 N | LLM 수 R | **R/N** | 영역당 점 | 최종 클러스터 수 |
|---|---|---|---|---|---|---|
| ChirpWave | 1 | 1,600 | 64 | 0.040 | 25.0 | 10.3 |
| CoCircles | 2 | 1,600 | 530 | 0.331 | 3.0 | 9.4 |
| BostonHouse | 13 | 405 | 155 | 0.383 | 2.6 | 4.0 |
| **FicoHeloc** | **35** | **7,897** | **7,423** | **0.940** | **1.06** | **1.6** |
| *(우리 moons/spiral)* | *2* | *3,200* | *454~602* | *0.14~0.19* | *5~9* | *—* |
| *(우리 MNIST)* | *784* | *60,000* | *59,978* | ***1.000*** | *1.00* | *—* |

**FicoHeloc은 R/N = 0.94, 영역당 1.06점으로 사실상 우리 MNIST와 같은 체제다.**
그런데도 방법이 돌아갔고, 오히려 **테스트 AUC가 원본보다 올라갔다** (0.8002 → 0.8050, Table 3).
최종 클러스터 수는 평균 **1.55개** — 논문의 해석은
*"이 데이터셋에는 딥 ReLU 망이 과하고, 로지스틱 회귀 하나면 패턴을 다 잡는다"* 이다.

→ **`open_questions.md` I3의 "R/N=1.000은 MNIST를 특수하게 만든다"는 관찰 자체는 여전히 맞다.**
   (영역 클러스터링 = 데이터 클러스터링이라는 함의는 그대로다.)
   **틀린 것은 "그래서 선행연구 방법이 전이되지 않는다"는 추론이었다.**
   R/N≈1은 선행연구가 이미 겪었고, 통과했다.

#### 그래서 무엇이 남는가 — 진짜 차이는 차원이지 R/N이 아니다

| | FicoHeloc | 우리 MNIST |
|---|---|---|
| R/N | 0.94 | 1.000 |
| 입력 차원 d | 35 | **784** |
| 증강행렬 `[A_r\|b_r]` 크기 | 2 × 36 | **10 × 785 = 7,850** |
| 연결성용 μ_P | 35차원 점 | **784차원 점 (= 이미지 자체)** |

→ **전이되지 않는 것은 τ나 refit이 아니라 "차원"이다.** 거리 하나 계산하는 비용이
   36개 성분 vs 7,850개 성분으로 **218배** 차이나고, 이게 D3의 실제 병목이다.
   `Pruner`(O(R·K))가 필요한 이유도 R/N이 아니라 여기에 있다.

→ **방침:** MNIST에서도 **refit을 시도할 수 있다.** 2D와 MNIST에서 방법을 다르게 갈 필요가
   (1차 조사가 생각한 이유로는) 없다. 다만 refit은 ε을 낮추는 별개의 손잡이이므로,
   **refit 있음/없음 두 곡선을 나란히 그리는 것**이 정직하다 — 우리 Ω=K 정의에는
   "대표 affine map을 어떻게 정하는가"가 안 들어 있기 때문이다.

### 4.3 Hanin & Rolnick: ⚠️ 우리 선분 테스트는 그들 프로토콜과 **다르다** [원문]

1차 조사는 *"영역 세기를 1D 직선과 2D 평면으로 잘라 교차 횟수로 센다"* 고만 적고
우리 P5가 "표준 기법의 재현"이라고 결론지었다. **원문의 실제 프로토콜은 이렇다:**

#### (1) 무엇으로 자르나 — 주력은 2D 평면이다

> for each run the number of regions is averaged over **5 different 2D cross-sections**, where for
> each cross-section we count the number of regions in the **(infinite) plane passing through the
> origin and two random training examples.** ... averaged over **10 independent training runs**

- **평면 = 원점 + 무작위 학습 예시 2개**를 지나는 무한 평면 (원점을 안 지나는 변형은 예시 3개)
- 단면 5개 × 학습 5회~40회 반복. **직선(1D)은 이론(정리 5) 쪽 서술에서 쓰이고, MNIST 실험은 평면이다.**

#### (2) 어떻게 세나 — **샘플링이 아니라 정확 계수다** ★★

> we add neurons of the network **one by one** from the first to last hidden layer, observing how
> each neuron cuts existing regions. Determining whether a region is cut by a neuron involves
> identifying whether the corresponding linear function on that region has zeros within the region.
> This can be solved easily by identifying whether **all vertices of the region have the same sign**
> (region is not cut) or **some two of the vertices have different signs** (region is cut).

즉 **영역 목록과 각 영역 위의 선형함수를 들고 다니면서, 뉴런을 하나씩 추가할 때마다
꼭짓점 부호를 검사해 쪼갠다.** 격자도 난수도 쓰지 않는다.

→ **이게 `open_questions.md` Q4의 원인 (b)(표본이 얇은 영역을 놓침)를 구조적으로 없앤 방법이다.**
   우리가 STEP 0-3에서 찾아낸 해법(부호벡터 2^h개를 LP로 실현가능성 판정)과 **답은 같지만
   비용이 다르다** — 그들은 **이미 존재하는 영역만** 검사하므로 O(영역수 × 뉴런수)이고,
   우리 LP 전수판정은 O(2^h)라 h=8이 한계다.
   → **Q4를 h=16·32·64에서 직접 확인하려면 그들 방식으로 바꿔야 한다.** (§Q4 항목 참조)

#### (3) 보고 단위 — 개수가 아니라 밀도

정리 5가 상계를 거는 대상은 **"입력공간 단위 부피당 평균 영역 수"**, 1차원에서는
**"단위 길이당 영역 수"** 다. 총 개수가 아니다.
→ 우리 P5의 "선분 하나가 평균 22.4개 영역을 지난다"는 **선분 길이를 함께 적어야** 그들 단위와 비교된다.

#### (4) 정정 결론

우리 선분 테스트가 재현한 것은 **영역 세기 프로토콜이 아니라 볼록성**이다.
그리고 볼록성은 이 논문의 **Lemma 7 (Activation Regions are Convex)** 로 이미 증명돼 있다
— 임의의 조각별 선형 활성함수에 대해 일반화된 형태로.
→ **P5는 "정리의 수치 확인"이고, I1의 "63개 영역 통과"는 그들 밀도 측정의 1D 변형이다.**
   "표준 기법을 재현했다"보다 이렇게 적는 것이 정확하다.

### 4.4 [새로 확인] Hanin & Rolnick이 Q3(정규화·학습이 구조를 바꾸는가)에 주는 대조군 [원문]

이 논문의 핵심 주장 자체가 **`open_questions.md` Q3의 대조군**이다:

> the number of activation patterns ... **is independent of the depth**, is tight both **at
> initialization and during training** ... We empirically verify that this behavior holds

- MNIST 학습 중 영역 수는 **거의 변하지 않는다** (처음엔 살짝 줄었다가 회복).
- **무작위 라벨 암기 과제에서만 영역 수가 증가**한다 (Figure 5, depth 3 width 32,
  정리 5 예측 4,608에서 출발) — 그리고 암기량이 클수록 더 많이 증가한다.
- 학습률·초기 가중치 스케일을 바꿔도 영역 수는 **초기값의 작은 상수배를 못 넘는다** (Figure 6).
- 초기화 기본값: Adam, lr `1e-3`, batch 128, 가중치 `N(0, 2/fan-in)`, **편향 `N(0, 1e-6)`**.

→ **Q3에 주는 답:** "정규화가 폴리토프 구조를 인위적으로 단순하게 만들어 ε-path를 좋아 보이게
   한다"는 우리 우려는, **영역 "개수"에 관한 한 선행연구가 이미 반증에 가깝게 답했다** —
   학습 자체가 영역 수를 거의 안 바꾸는데 weight decay가 크게 바꾸리라 기대하기 어렵다.
   **다만 ε-path가 재는 것은 개수가 아니라 "영역들이 서로 얼마나 비슷한가"이고, 그건 이 논문이
   재지 않았다.** → Q3은 여전히 열려 있되, **질문을 "개수"가 아니라 "유사도"로 좁혀 적어야 한다.**

→ **덤으로 얻는 대조군 설계:** §6.3이 요구하는 "랜덤 초기화 모델 대조군"에 더해,
   **무작위 라벨로 암기시킨 모델**을 하나 더 두면 좋다. 영역 수가 실제로 늘어나는
   유일한 조건이므로 **ε-path가 위로 밀려야 한다** — 곡선이 뭔가를 재고 있다면.

### 4.3 Hanin & Rolnick: 우리 선분 테스트는 표준 기법이었다 [요약]

영역 세기를 **1D 직선과 2D 평면으로 잘라 교차 횟수로** 센다고 알려져 있다.
→ `open_questions.md` I1(선분 볼록성 테스트, MNIST 평균 63개 영역 통과)은
**우리가 발명한 게 아니라 표준 기법을 재현한 것이다.** 그렇게 적는 게 정직하다.
(정확한 프로토콜 — 직선을 어떻게 뽑는지, 몇 개인지 — 은 로컬에서 원문 확인 필요.)

---

## 5. D4 — ε을 무엇으로 재는가

여기는 답이 제일 선명하다. **[원문]**

> **Reconstruction is scored in the target's own output metric** — KL per position for a language
> model, MSE for TMS or ResidMLP, whatever a new domain's output space calls for.
> **Report which metric a reconstruction number is in; values are not comparable across metrics**,
> and a new domain earns its recon metric before it earns a Pareto front.

- SPD 논문도 같다: "$D$는 출력 공간의 적절한 divergence, LM은 KL, 아니면 MSE."
- Aletheia는 **AUC(분류) / MSE(회귀)**, 그리고 추가로 영역별 **local AUC vs global AUC**를
  나란히 보고한다 [코드] — "이 영역 안에서는 잘 맞는데 전역적으로는 아닌" 경우를 잡는
  진단이고, 우리에게 없는 지표다.

### 우리에게 주는 답

`open_questions.md` D4의 잠정 결론("둘 이상을 같이 그리는 것이 안전할 듯")은
**선행연구 규범과 어긋난다.** 규범은 *기준 척도 하나를 도메인에서 정하고 명시*하는 것이다.

→ **우리 타깃은 분류기이므로 KL이 기준 자리다.**
   정확도는 사람이 읽는 보조 표시로만 쓰고 **선택 기준으로는 쓰지 않는다.**
   (정확도가 계단형이라 거칠다는 우려는 D4에 이미 적혀 있고, 선행연구가 그걸 피한 방식이 이것이다.)

→ 덤: **영역별 local/global 성능을 나란히 보는 진단**을 추가할 것. 공짜로 나온다.

---

## 6. 우리 열린 질문에 닿는 것

### 6.1 Q1 (라우팅 비용) ← Aletheia의 `flatten` [원문] ★ — **단, 공짜가 아니다**

논문 §5.3 **Algorithm 2**:

```
Require: 데이터 {x_i, y_i}, 병합된 LLM 클러스터들
 1: 각 클러스터의 국소 선형 계수·절편을 모은다
 2: 그것을 1층 망의 은닉층 가중치·편향으로 그대로 꽂는다   ← 은닉 뉴런 수 = 클러스터 수 K
 3: 순전파해서 출력층을 GLM으로 추정한다
 4: SGD 계열로 fine-tune 한다                              ← ⚠️ 순수 컴파일이 아니다
```

→ Q1("`k(x)`를 알려면 원본 네트워크 첫 층을 돌려야 하므로 우리 설명은 자립적이지 않다")에 대한
   선행연구의 답이 맞다: **병합 결과를 다시 네트워크로 컴파일하면 라우팅이 오라클 호출이 아니라
   그 네트워크의 일부가 된다.** 그런데 원문을 읽으니 대가가 둘 있다.

**대가 (1) — fine-tune이 들어간다.** 4단계가 없으면 성능이 얼마나 떨어지는지 논문은 보고하지 않는다.
즉 `flatten`은 "라우팅을 공짜로 컴파일했다"가 아니라 **"컴파일 + 재학습"** 이다.
우리 Ω 회계로는 재학습된 가중치도 설명의 일부이므로, **비용을 청구해야 한다.**

**대가 (2) — 국소 배타성을 잃는다.** 논문 자신의 서술:

> In the original deep ReLU network, each LLM corresponds to a linear model that is **only active
> locally**. But as the network is flattened, each of the original local regions is now affected by
> **an ensemble of multiple linear models.**

→ 즉 flatten 후에는 *"입력 x는 조견표의 k번째 줄 하나를 쓴다"* 가 더 이상 참이 아니다.
   **조견표라는 설명 형태 자체가 바뀐다.** Ω=K의 의미가 달라지므로 그냥 이어붙일 수 없다.

**대가의 크기 — Table 3 (100회 몬테카를로 평균):**

| 데이터 | 지표 | 원본 ReLU-Net | Merge-Net | FL-Net | SLFN(랜덤초기화 동일크기) | 클러스터 수 |
|---|---|---|---|---|---|---|
| ChirpWave | MSE ↓ | 0.0486 | 0.0721 | 0.0839 | 0.3351 | 10.3 |
| CoCircles | AUC ↑ | 0.9174 | 0.8720 | 0.8845 | 0.8280 | 9.4 |
| BostonHouse | MSE ↓ | 0.0074 | 0.0079 | 0.0087 | 0.0132 | 4.0 |
| FicoHeloc | AUC ↑ | 0.8002 | **0.8050** | 0.7962 | 0.7545 | 1.6 |

읽는 법 세 가지:

1. **병합 비용은 작다.** MSE가 1.5배 나빠지는 정도(ChirpWave)이고, FicoHeloc은 **오히려 개선**된다.
2. **flatten 추가 비용도 작다.** Merge-Net → FL-Net 격차가 병합 격차보다 작다.
3. ★ **SLFN 열이 이 표의 대조군이다.** FL-Net과 **크기가 완전히 같고 초기화만 다른** 망이다.
   FL-Net이 SLFN을 크게 이긴다(0.0839 vs 0.3351) → **"클러스터의 affine map이 좋은 초기화다"**
   가 논문의 결론이고, 이게 §6.3이 요구하는 "dense/chance 종점" 규범의 실제 사례다.
   → **우리도 같은 형태의 대조군을 둘 수 있다:** K개 클러스터로 만든 조견표 vs
     **같은 K를 무작위로 배정한 조견표.** 후자를 못 이기면 클러스터링이 아무것도 안 한 것이다.
     이건 §6.3의 "랜덤 초기화 모델 대조군"보다 **더 엄한 기준**이고, 비용은 거의 없다.

### 6.2 Q2 (두 가지 실패 구분) ← 앵커가 **필수 절차**다 [원문] ★★

> In a new domain, you should **first decompose a target whose mechanisms are already known and
> verify that the method recovers them. This anchor is mandatory:** under-training and an
> ill-matched reconstruction objective can both produce **clean-looking nulls**, so an unanchored
> target cannot distinguish between 'real absence of structure' and 'badly configured method'.

Q2에서 "미리 설계해둘 것"이라고 적어둔 sanity check가 바로 이것이고,
**우리는 이미 앵커를 갖고 있다** — `scripts/walkthrough.py` STEP 6의 4조각 장난감은
전수탐색 최적을 알고, 정답 분할(`{0,2} | {1,3}`)도 안다.

→ **클러스터링 코드를 실모델에 돌리기 전에, 4조각 장난감에서 전수탐색 답을 복원하는지
   먼저 확인한다.** 이게 선행연구가 요구하는 순서다.

### 6.3 ε-path의 정식 이름과 공정성 조건 [원문]

이 계열에서는 **"reconstruction / minimality Pareto front"** 라고 부른다. 조건이 명문화돼 있다:

> A fair comparison ... holds the target, data, mask family, reconstruction metric, and adversarial
> budget fixed; it compares active-unit count at matched reconstruction (or reconstruction at
> matched count), reports each method's unit granularity, and **includes the dense and chance
> endpoints.** Merely beating the architectural-unit count is not evidence of useful minimality:
> even an untrained hundred-step smoke can often clear a threshold that weak.
> **If a bad or untrained baseline passes the stated bar, strengthen the bar** before interpreting.

→ 우리 ε-path에 **양 끝점이 반드시 들어가야 한다**: K = R(병합 없음, ε=0)과 K = 1(전부 하나).
   그리고 **대조군**: 학습 안 된 랜덤 초기화 모델의 ε-path. 그게 우리 곡선과 구분 안 되면
   곡선은 아무것도 말하지 않는 것이다.

### 6.4 I4(ε 최소화만으로 중요도가 나옴)의 선행연구 대응 [원문]

APD의 세 성질 — **faithfulness**(컴포넌트 합 = 원본 파라미터), **minimality**(입력당 최소
개수로 행동 재현), **simplicity**(컴포넌트당 rank·layer 최소) — 중 minimality가 그것이다.
APD는 top-$k$ 하드코딩, SPD는 이걸 **학습되는 causal importance 함수**로 바꿨다
(ablatability로 정의: $g=0$이면 완전히 ablate 가능, $g=1$이면 불가).

→ I4에서 관찰한 "중요도를 정의한 적 없는데 알아서 나왔다"는 현상이
   **이 계열 전체의 출발점**이다. Stage 4의 다리는 여기서 놓는다.

---

## 7. 그래서 D1~D4를 어떻게 확정하는가 (제안)

> **2026-08-19 로컬 원문 확인 후 갱신됨.** 바뀐 칸에 ⚠️ 표시.

| | 선행연구 답 | `open_questions.md` 잠정안 | 판정 · 제안 |
|---|---|---|---|
| **D1** | 출력 divergence 기준. ⚠️ **affine map 위에서 거리를 잰 선행연구는 Aletheia 하나뿐이고 유클리드다. Frobenius를 실제로 쓴 연구는 없다** | 동일 | ✅ 방침 유지. ⚠️ **λ와 Frobenius 둘 다 "우리 추가분"으로 강등.** λ=1이 재현 기준선 |
| **D1+** | ⚠️ **[새로 발견] 네 번째 척도: 부호벡터 `r` 위의 해밍 거리** (polytope lens) | 없음 | ➕ **가장 싸다** (`[A_r\|b_r]` 생성 불필요). 2D 비교에 추가 |
| **D2** | agglomerative+average+**kNN 연결성**, 병합 후 **refit**, **확률적 선택(γ=0.2)** | agglomerative+average | ⚠️ 연결성·refit·확률선택 셋 다 누락 → 추가. **연결성은 계수공간이 아니라 입력공간 μ_P 기준** |
| **D3** | **`Pruner`: topk core + 최근접 배정, O(R·K)** ([코드] 등급 유지) | 랜덤 서브샘플 M | ✅ **M을 정할 필요가 없어짐.** 2D에서 Merger vs Pruner 격차 측정 |
| **D3+** | ⚠️ **R/N≈1은 장애물이 아니다** — FicoHeloc이 0.94로 이미 통과. **진짜 병목은 차원(35 vs 784)** | "MNIST는 refit 불가" | ❌ **틀렸음.** refit은 클러스터 단위라 MNIST에서도 정의된다 → **refit 있음/없음 두 곡선** |
| **D4** | 도메인 출력 척도 **하나**, 명시 | 둘 이상 병기 | ❌ 규범과 어긋남 → **KL 하나를 기준으로**, 정확도는 보조 표시 |
| **상계** | hierarchical + **확률적 재시작**, 손잡이 함께 보고 | greedy 상계임을 명시만 | ⚠️ 확률적 선택 구현 (10줄) |
| **검증** | **알려진 메커니즘 앵커 필수** | Q2에 "설계해둘 것" | ⚠️ **4조각 앵커를 먼저 통과시킬 것** |
| **곡선** | Pareto front, **dense/chance 종점 포함**. ⚠️ **Aletheia Table 3의 SLFN 열이 실제 사례** | 종점 언급 없음 | ⚠️ K=R, K=1 + **랜덤 배정 조견표 대조군**(더 엄함) + 랜덤초기화 모델 |
| **Q3** | ⚠️ **학습은 영역 "개수"를 거의 안 바꾼다** (무작위 라벨 암기만 예외) | "정규화가 구조를 단순화할까" | ⚠️ **질문을 "개수"→"유사도"로 좁힐 것.** 암기 모델을 대조군에 추가 |
| **Q4** | ⚠️ **증분 정확 계수법**(꼭짓점 부호 검사, O(영역수×뉴런수)) | LP 전수판정 O(2^h), h≤8 한계 | ➕ **h=16·32·64로 확장 가능** → (a)상자절단 vs (c)비일반위치를 실모델에서 직접 가름 |

**1차 조사의 결론이었던 *"D3만 열려 있다고 생각했는데 D2·D4가 더 문제였다"* 는 유지된다.
로컬 원문 확인이 더한 것은 세 가지다:**

1. **우리가 "재현"이라고 부르던 것 중 일부는 재현이 아니었다** — Frobenius도, 절편 가중치 λ도
   선행연구에 없다. 그렇게 적으면 근거를 과장하는 것이 된다. (D1)
2. **MNIST가 특수하다는 믿음이 과했다** — R/N≈1 사례가 이미 선행연구에 있고 통과했다.
   진짜 어려움은 R/N이 아니라 차원이다. (D3)
3. **Q4와 Q3에 쓸 도구·대조군이 공짜로 딸려 왔다** — 증분 정확 계수법과 암기 모델 대조군.

---

## 8. 로컬 세션에서 확인한 것 — 미확인 항목 전부 종결 (2026-08-19)

`local_session_manual.md` §3 체크리스트에 대한 답이다. **5개 항목 모두 종결됐고,
그중 3개가 1차 조사의 서술을 뒤집었다.**

| 미확인 항목 | 결과 | 어디에 |
|---|---|---|
| Aletheia 논문의 병합 기준 | ⚠️ **뒤집힘** — 논문에는 절편 가중치도 표준화도 없다. λ는 코드에만 있는 미문서화 손잡이 | §2, §3.1 |
| polytope lens의 클러스터링 대상 | ⚠️ **뒤집힘** — Frobenius는 **각주 8의 "예를 들어"** 한 줄. 실험은 활성/spline code 위 HDBSCAN(+Hamming) | §2, §3.3 |
| Hanin & Rolnick의 프로토콜 | ⚠️ **뒤집힘** — 샘플링이 아니라 **정확 계수**(꼭짓점 부호 검사). 주력은 1D가 아니라 **2D 평면** | §4.3 |
| Srivastava 2015의 클러스터링 | 종결 — **클러스터링 알고리즘이 아니라 t-SNE 시각화 + kNN/검색** | §8.1 |
| Chu et al. 2018 (OpenBox) | 종결 — **분해까지만. 병합·클러스터링·ε 없음** | §8.2 |

### 8.1 Srivastava et al. 2015 — "클러스터링"이 아니었다 [원문]

README와 `concepts.md` §1.4가 이 논문을 "활성 코드 클러스터링"의 출처로 인용해 왔다.
**원문에서 실제로 한 것은 이렇다:**

- 활성 패턴을 **submask** `s_i ∈ {0,1}^u` 라 부른다 (유닛이 0이면 0, 아니면 1). **우리 `r`과 같다.**
- 이 submask들을 **t-SNE로 2D에 찍는다.** 클러스터링 알고리즘은 돌리지 않는다.
  "클러스터"는 그림에서 사람이 보는 것이다.
- 실용적으로는 submask를 **이진 코드로 써서 kNN 분류와 검색(retrieval)** 에 쓴다 (DiffHash와 비교).
- **마지막 은닉층 submask만 쓴다.** 이유: 깊은 층 submask가 더 유용하고 길이가 짧아서.
  (1000 유닛 → 비트열 1000자리)

→ **정정:** "활성 코드 클러스터링" 대신 **"활성 코드 시각화 + 검색"** 이라고 적어야 한다.
   `concepts.md` §1.4가 "데이터 포인트에서 영역을 샘플링"의 근거로 이 논문을 든 것은 **맞다** —
   그들도 정확히 그렇게 한다(테스트셋 1만 개 → submask 1만 개).

→ ★ **뜻밖의 수확 — 이 논문이 이미 "랜덤 초기화 대조군"을 갖고 있다 (Figure 3):**

  > (a) shows the submasks from an **untrained** network layer which **lacks any discernable
  > structure.** (b) shows submasks from a **trained** network layer, showing **clearly demarcated
  > clusters** relevant to the supervised learning task.

  학습 전에는 구조가 없고 학습 후에는 MNIST 10개 클래스에 대응하는 군집이 생긴다.
  **§6.3이 요구하는 대조군을 2015년에 이미 이 형태로 했다.** 우리 ε-path 대조군의 선례로 인용 가능.
  덤: 틀린 예측을 한 입력의 submask는 **틀린 클래스 군집 안에 놓인다** — 활성 패턴이 예측을
  설명한다는 직접 증거이고, 우리 클러스터 라벨 해석의 선례다.

### 8.2 Chu et al. 2018 (OpenBox) — 우리와 같은 분해, 그러나 병합이 없다 [원문]

> we propose an elegant closed form solution named OpenBox to compute exact and consistent
> interpretations for the family of Piecewise Linear Neural Networks (PLNN). The major idea is to
> **first transform a PLNN into a mathematically equivalent set of linear classifiers**, then
> **interpret each linear classifier by the features that dominate its prediction.**

- 앞부분(PLNN → 등가 선형분류기 집합, 볼록 폴리토프)은 **우리 Stage 1의 affine 항등식과 같다.**
  독립적인 재현 근거가 하나 더 생긴 셈이다.
- 뒷부분은 **각 폴리토프 안에서 어떤 피처가 예측을 지배하는지** 보는 것(피처 귀속)이다.
- **병합·클러스터링·압축이 없다.** `merge`/`cluster`/`polytope 수` 관련 서술이 전문에 없다.
  ε 개념도 없다 — 해석이 "정확(exact)"한 것이 그들의 주장이므로 근사 오차 자체가 없다.

→ **판정: D1~D4에 기여하는 바가 없다.** README 선행연구 목록에는 남기되
   **"분해까지만 — 병합 없음"** 한 줄을 붙여 위치를 분명히 한다.
   우리와의 관계는 *"같은 분해에서 출발해 우리는 압축으로, 그들은 피처 귀속으로 갈라진다"* 이다.

### 8.3 남은 미확인

없다. 다만 **`Pruner`는 여전히 [코드] 등급**이다 (§4.1 — 논문에 존재하지 않는 구현체 전용 기능).
논문 근거가 필요하면 Algorithm 1의 4~5단계(작은 클러스터 → 가까운 큰 클러스터 흡수)를 인용해야 하고,
그건 `Pruner`와 **적용 시점이 다르다**(병합 후 vs 병합 대신).

---

## 9. ⚠️ 이 계열은 멈추지 않았다 — 2024~2026 문헌 (2026-08-19 검색)

**1차 조사의 "2020년경 멈춤" 은 검증된 적 없는 인상이었다.** 실제로 검색해 보니 세 갈래로
활발하고, **그중 둘은 우리 열린 질문에 직접 답하며, 하나는 우리 방향 판단을 바꾼다.**

> **근거 등급:** PDF 5편을 `external/papers/` 에 받아뒀다
> (`2602.02464`, `2605.06218`, `2605.06300`, `2606.21687`, `2412.18283`).
> **§9.3 의 MFA 숫자·인용은 전문에서 직접 확인했다 [원문].**
> 나머지(§9.1·9.2·9.4)는 **초록 수준 [요약]** 이다 — 결과에 쓰려면 전문 확인이 먼저다.

### 9.1 갈래 A — 영역 기하·계수: 우리 도구가 이미 표준화돼 있다

| 자료 | 무엇을 하나 | 우리와의 관계 |
|---|---|---|
| **AffineLens** (`2605.06218`, 2026-05) | 뉴런 초평면 배치를 층별로 열거해 **"증명 가능하게 비어 있지 않은 maximal CPA 영역 + 내부 대표점"** 을 돌려주는 도구. BN·pooling·residual·conv 지원 | **Q4 의 도구가 이미 나와 있다.** 우리 `_count_regions_exact()`(LP, O(2^h))도, Hanin & Rolnick 의 증분 부호검사도 이걸로 대체 가능 |
| **Expressivity Saturation** (`2606.21687`, 2026-06) | **선분 프로브**로 1D 조각 수 상계를 유도하고, 2D 이상에서 **실현된 영역을 정확 열거**. 과제 복잡도를 올리면 **실현 영역이 오히려 줄어든다**("expressivity saturation") | **우리 P5·I1 이 그들 프로토콜과 같은 계열.** 그리고 "영역 수 ↔ 과제 난이도" 는 우리 **I2**(spiral 깊이 효과)와 정면으로 겹친다 |
| **Region Seeding** (`2605.06300`, 2026-05) | 뉴런 스위칭 표면을 데이터 근처로 끌어당기는 **정규화**를 걸면 국소 영역 수가 **엄격히 증가**한다는 충분조건 + ImageNet 실험 | ★ **우리 Q3 이 부분적으로 답해졌다** (아래 9.4) |
| 그 외 | 국소 복잡도(`2412.18283`), 영역 세기의 계산복잡도(`2505.16716`), 이산 기하(`2606.07728`) | 이론 쪽 배경 |

→ **함의:** `open_questions.md` I1 에 "선분 볼록성 테스트는 차원 무관 도구다 ★" 라고 적어둔 것은
이미 §4.3 에서 "H&R 재현" 으로 강등했는데, **2026 년 기준으로는 그보다 더 낮다** —
전용 도구(AffineLens)와 정확 열거 프로토콜이 공개돼 있다.
우리 기여로 내세울 것이 아니라 **가져다 쓸 것**이다.

### 9.2 갈래 B — 영역별 해석: OpenBox 의 후계

**Re3 (ReLU Region Reasoning)** — `Machine Learning` (Springer), 2026-01.
영역별로 뉴런 활성과 피처 기여를 읽어 예측을 **오차 없이 재현**하고, 활성 희소성과 지배 유닛을
찾아 모델 단순화 대상을 짚는다. 표형·이미지 데이터 모두 실험.

→ Chu et al. 2018(OpenBox)과 **같은 형태**다 — 분해 후 영역별 피처 귀속, 병합 없음.
   즉 §8.2 에서 "OpenBox 는 분해까지만" 이라 적은 갈래가 2026년에도 이어지고 있다.
   **여전히 ε 도 압축도 없다.** 우리 Ω-ε 축은 이 갈래에 없다.

### 9.3 갈래 C — ★★ 방향 판단을 바꾸는 것: 영역 기반 분해가 SAE 를 이겼다

**"From Directions to Regions: Decomposing Activations in Language Models via Local Geometry"**
(`2602.02464`, 2026-02). **MFA(Mixture of Factor Analyzers)** 로 활성 공간을 가우시안 영역으로
쪼개고, 영역마다 국소 저차원 부분공간을 붙인다.

| | SAE (dictionary learning) | **MFA (영역 기반)** |
|---|---|---|
| 해석가능 비율 | 0.29 ± 0.2 | **0.96 ± 0.2** |
| 스티어링 (coherence·개념정합) | 기준 | **약 2배** |
| 재구성 방식 | 전역 사전 피처를 많이 누적 | **① 영역에 앵커 → ② 국소 변동으로 정제** |

원문 인용 (`2602.02464` L448–450, L217–219) [원문]:

> Across all settings, **MFA achieves an average IF of 0.96 ± 0.2** ... compared to **0.29 ± 0.2**

> These responsibilities assign each activation to the component whose local subspace best
> explains it, **allowing us to express the activation as a mixture of the components.**

즉 배정이 **하드 라벨이 아니라 혼합(mixture)** 이다. 그리고 컴포넌트마다 **저랭크 부분공간**이
붙어 있어, 같은 영역 안에서도 변동을 몇 개 방향으로 더 쪼갠다.
→ 우리 조견표 대비 **표현력 축이 둘 더 있다: (1) 소프트 배정 (2) 영역 내 국소 부분공간.**

**확인된 사실 하나 더:** 이 논문에는 `polytope` 도 `spline` 도 **한 번도 나오지 않는다.**
polytope lens 계보를 전혀 참조하지 않고 같은 결론에 도달했다.
→ **방향의 독립적 재발견이다.** 우리에게는 좋은 신호(방향이 옳다)이자
  나쁜 신호(우리 계보의 어휘가 그 성과에 기여하지 못했다)다.

**세 가지가 중요하다:**

1. **"영역으로 분해한다" 는 방향은 2026년에 검증됐다.** 그것도 SAE 를 크게 이기는 형태로.
   → 우리 Stage 1 의 문제의식 자체는 살아 있고, 오히려 지금이 그 방향의 전성기다.
2. **그런데 이기는 형태가 우리 형태가 아니다.**
   - MFA 는 **soft** 영역(가우시안 responsibility) — 우리는 **hard** 폴리토프
   - MFA 는 **여러 컴포넌트가 동시에 활성** (`k>1`) — 우리는 **입력당 조견표 한 줄** (`k=1`)
   - MFA 는 **영역 + 국소 부분공간 2단** — 우리는 **영역당 affine map 하나**
3. **polytope lens 가 2022년에 제안한 합성이 실현됐다 — 다른 재료로, 인용 없이.**
   MFA 는 Black et al. 2022 를 인용하지 않는 것으로 보인다.
   → **"이 방향은 비어 있다" 는 더 이상 참이 아니다.** 채워졌고, 우리가 서 있는 구석만 비었다.

### 9.4 Q3 이 부분적으로 답해졌다 — 그리고 우리 가설과 방향이 반대다

`open_questions.md` Q3 의 우려: *"정규화를 걸면 폴리토프 구조가 인위적으로 단순해져
ε-path 가 좋아 보일 수 있다."*

**Region Seeding (`2605.06300`)** 이 정규화 ↔ 영역 수를 정면으로 다룬다. 그런데:

- 그들이 보인 것은 **정규화가 영역을 늘릴 수 있다**는 것이다(스위칭 표면을 데이터 근처로 끌면
  국소 영역 수가 엄격히 증가). 우리가 걱정한 "단순해진다" 의 반대 방향이다.
- 그리고 **"표준 학습은 구조적으로 가능한 것보다 적은 영역만 실현한다"** 는 관찰은
  Hanin & Rolnick(§4.4)과 Expressivity Saturation 과 **셋 다 일치**한다.

→ **Q3 을 다시 좁혀 적어야 한다.** "정규화가 구조를 단순화하는가" 는 이미 반증에 가깝다.
   남는 질문은 §4.4 에서 좁힌 그대로 — **"영역 수가 아니라 영역 간 유사도가 어떻게 변하는가"** 다.
   그리고 **그건 세 논문 중 누구도 재지 않았다.** (그들은 전부 개수·밀도만 잰다.)
   → 여기가 우리 ε-path 가 실제로 새로 재는 지점이다. **작지만 진짜로 비어 있다.**

### 9.5 그래서 우리 위치를 정직하게 적으면

| 우리가 하는 것 | 2026년 기준 위치 |
|---|---|
| 영역 추출·정확 계수 | **도구가 이미 있다** (AffineLens). 재현 학습으로는 유효, 기여로는 아님 |
| 선분 볼록성·밀도 측정 | **표준 프로토콜** (H&R → Expressivity Saturation). 재현 |
| 영역별 affine map 해석 | **활발** (Re3), 그러나 병합·ε 없음 |
| **영역 병합 + ε-path (Ω=K)** | **비어 있다.** 단, 아래 단서 붙음 |
| 영역 기반 분해 일반 | **전성기.** 그런데 `k>1` soft 형태가 이긴다 (MFA) |

**단서:** "비어 있다" 가 "가치 있다" 를 뜻하지 않는다.
polytope lens 자신이 `k=1` 클러스터링을 *"decomposable description 을 찾는 데 명백히 최적이 아니다"*
라고 적었고(§10), MFA 가 `k>1` 로 실제 성과를 냈다.
**우리가 서 있는 자리는 설계공간에서 가장 약한 구석일 가능성이 높다.**

→ 그런데 이건 **이 프로젝트의 목표(개념 이해 + 선행연구 재현)에는 문제가 되지 않는다.**
   가장 단순한 구석이 배우기에는 가장 좋은 자리다.
   **문제가 되는 것은 결과를 "이 방향이 유망하다" 로 프레이밍할 때뿐이다.**
   → **Stage 1 을 `k=1` 기준선으로 명시**하고, Stage 4 를 "`k=1` → `k>1` 격차 측정" 으로
     바꾸면 이 위험이 사라지고, 비교 축도 실재하게 된다 (MFA 라는 상대가 실제로 있으므로).

---

## 10. ⚠️ 우리가 놓친 문단 — polytope lens 는 `Ω=K` 를 스스로 부정했다

§3.3 을 쓰면서 이 문단을 빠뜨렸다. **논문에서 방향 판단에 가장 중요한 대목이다.**

> Clustering activations can be thought of as finding a k-sparse set of features in the activations
> **where k = 1** (when k is the number of active elements). In other words, finding N clusters is
> equivalent to finding an overcomplete basis with N basis directions, **only one of which can be
> active at any one time. This clearly isn't optimal for finding decomposable descriptions of neural
> networks**; ideally we'd let more features be active at a time i.e. we'd like to let k > 1, but
> with clustering k = 1.

**우리 조견표는 입력 하나당 정확히 한 줄이다. 그게 `k=1` 이다.**
그리고 SPD/VPD 가 하는 일은 정확히 `k>1` — 입력마다 여러 컴포넌트가 동시에 causally important 하다.

→ **§1 에서 "두 계열" 이라 부른 것은 대등한 두 접근이 아니라 같은 축의 두 지점일 수 있다.**
   그리고 우리는 열등한 쪽 끝에 서 있다.

### 그런데 같은 문단에 반론도 있다 — 이게 우리 편이다

> But clustering isn't completely senseless — If every combination of sparse overcomplete basis
> vectors interacts with nonlinearities in a different way, then every combination behaves like a
> different feature. ... **Overcomplete basis features will be one component of that structure, but
> they don't account for scale; polytopes do.** A path toward understanding superposition in neural
> networks might be an approach that describes it in terms of an overcomplete basis **and** in terms
> of polytopes. A potential future research direction might therefore be to **find overcomplete
> bases in spline codes rather than simply clustering them.**

두 가지를 준다:

1. **폴리토프는 스케일을 설명하고 overcomplete basis 는 못 한다.** 우리 `[A_r | b_r]` 는
   방향뿐 아니라 크기와 절편을 다 담는다 — STEP 8 에서 cosine 이 실패한 이유가 정확히
   "스케일과 절편을 버려서" 였다. **그 실험이 이 주장의 축소판 재현이다.**
2. **제안된 길이 우리 Stage 4 다.** 다만 §9.3 대로 **이미 다른 재료(가우시안 MFA)로 채워졌다.**
   우리가 할 수 있는 차별점이 남아 있다면 그건 **hard 폴리토프의 정확성**이지 참신함이 아니다.

### 이걸 놓친 이유도 기록해둔다

`local_session_manual.md` §3 체크리스트가 **D1~D4 를 풀려고** 쓰였다 —
"Frobenius 를 썼나", "HDBSCAN 을 뭐에 돌렸나", "UMAP 을 쓰나".
**논문 자신의 결론과 한계를 묻는 항목이 하나도 없었다.**

→ **교훈: 전술적 질문만 담은 체크리스트는 전략적 내용을 걸러낸다.**
   다음 문헌 조사 체크리스트에는 반드시 넣을 것:
   *"저자들이 스스로 밝힌 한계는 무엇인가"* / *"저자들이 다음에 하라고 한 것은 무엇인가"*.

---

## 11. Q2 를 다시 적어야 한다 — 간극은 "공간" 이 아니라 "k" 다

`open_questions.md` Q2 는 이렇게 적어뒀다:
*"SPD 컴포넌트는 파라미터공간, polytope 클러스터는 입력공간. 단위가 달라 직접 비교가 안 된다."*

**§10 을 반영하면 진짜 간극은 공간이 아니다:**

| | Stage 1 클러스터 | Stage 2 컴포넌트 |
|---|---|---|
| 입력 하나에 대응하는 것 | **기호 하나** (라벨 `k(x)`) | **부분집합** (활성 컴포넌트 집합) |
| `k` | **1** | **>1** |

→ 이 둘을 상호정보량으로 재면 **대응이 실제로 있어도 구조적으로 낮게 나온다.**
   `k=1` 라벨이 담을 수 있는 정보량 자체가 `log₂K` 로 묶여 있기 때문이다.
   → **Q2 가 배제하려던 실패 모드 (b)"재는 방법이 틀렸다" 에, 예상과 다른 경로로 빠진다.**

→ **Q2 의 sanity check 설계에 반드시 넣을 것:**
   toy 상황을 만들 때 **`k>1` 인 정답을 `k=1` 라벨로 재면 MI 가 얼마나 떨어지는지**를 먼저 잰다.
   그 값이 "대응 없음" 의 기준선(chance)이다. 이걸 안 재면 무엇과 비교할지 알 수 없다.
