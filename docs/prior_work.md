# 선행연구가 이미 답한 것 · 아직 답하지 않은 것

> `open_questions.md`의 미결정 사항 D1~D4와 열린 질문 Q1~Q2에 대해,
> 선행연구가 실제로 어떤 선택을 했는지 조사한 결과.
> **이 프로젝트의 목적이 선행연구 재현이므로, 여기 적힌 선택을 기본값으로 삼는다.**
> 우리가 선행연구와 다르게 가는 지점은 그렇게 명시한다.
>
> 조사일: 2026-08-19 (GitHub 원격 세션)
> 못 채운 칸을 로컬에서 이어가는 절차: `docs/local_session_manual.md`

---

## 0. 근거 등급 — 어디까지 확인했는지

조사 환경(GitHub 원격 세션)에서 **arxiv.org를 포함한 대부분의 논문 호스트가 차단**돼 있었다.
그래서 자료마다 확인 깊이가 다르다. 아래 표기를 문서 전체에서 사용한다.

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
| Aletheia (Sudjianto et al. 2020) | **[코드]** | PyPI `aletheia-dnn==1.3.5` 휠에서 심볼·docstring 추출 |
| polytope lens (Black et al. 2022) | **[요약]** | arxiv/alignmentforum 차단 |
| Hanin & Rolnick 2019 | **[요약]** | 〃 |
| Srivastava et al. 2015 | **[요약]** | 〃 |
| Chu et al. 2018 (OpenBox) | **미확인** | 〃 |

---

## 1. 계열이 두 개이고, 답이 다르다

| | **영역 병합 계열** | **파라미터 분해 계열** |
|---|---|---|
| 대표 | Aletheia (2020), polytope lens (2022) | APD (2025) → SPD (2025) → VPD (2026) |
| 병합 대상 | **입력공간** 영역의 affine map `[A_r\|b_r]` | **파라미터공간** 컴포넌트 |
| 거리 | 계수벡터 유클리드 / Frobenius | **거리 없음** — 출력 divergence로 직접 |
| 알고리즘 | agglomerative(+연결성), HDBSCAN | 학습 + 사후 MDL 계층 병합 |
| ε | AUC / MSE | KL(분류·LM) / MSE(회귀) |
| 상태 | 2020년경 멈춤 | 2026년까지 활발, **증거 기준이 문서화됨** |

Stage 1은 첫째 계열, Stage 2는 둘째 계열이다.
**그런데 D1~D4에 대한 더 성숙한 규범은 둘째 계열에 있다.** 첫째 계열은 "이렇게 했다"만
있고, 둘째 계열은 "무엇을 증거로 인정하고 무엇을 인정하지 않는지"까지 적어 놨다.
→ **Stage 1의 방법론도 둘째 계열의 규범을 따르는 것이 낫다.**

---

## 2. D1 — 거리 척도

### 선행연구는 셋을 비교하지 않았다. 각자 하나를 골랐다.

| 연구 | 무엇을 썼나 | 등급 |
|---|---|---|
| polytope lens | "각 폴리토프가 구현하는 affine 변환의 implied weight matrix 차이의 **Frobenius norm**"으로 폴리토프 유사도를 정의. 단 실제 클러스터링은 affine map이 아니라 **활성/spline code** 위에서 돌림 | [요약] |
| Aletheia | `euclidean_distances`, 피처는 `all_coefs` + `all_intercepts`. 내부에 `all_bias_weight`가 따로 있음 → **절편 항에 가중치를 걸어** 계수와 스케일을 맞춤 | [코드] |
| SPD / VPD | 파라미터 공간 거리를 **아예 쓰지 않음**. 출력 공간 divergence로 직접 채점 | [원문] |
| row-wise cosine | **어디서도 표준이 아님** | — |

### SPD 원문 인용 [원문]

> $\mathcal{L}_{\text{stochastic-recon}}=\frac{1}{S}\sum^S_{s=1} D\left(f(x|W'(x,r^{(s)})), f(x|W)\right)$
> Here, $D$ is some appropriate divergence measure **in the space of model outputs**,
> such as KL-divergence for language models, or MSE loss.

### 우리에게 주는 답

1. **`open_questions.md` D1의 방침("logit이 기준, 나머지는 대리")은 SPD 계열 규범과 일치한다.**
   그대로 간다.
2. Frobenius는 polytope lens가 제안한 대리 지표라 **근거가 있다.**
3. **row-wise cosine은 선행연구 근거가 없는 우리만의 축이다.**
   → 재현이 아니라 우리가 추가하는 부분. 그렇게 명시할 것.
4. **[새로 배운 것] 절편에 가중치를 걸 수 있다.** 우리는 `[A_r | b_r]`에서 절편에 암묵적으로
   가중치 1을 주고 있다. Aletheia는 이걸 **튜닝 대상**으로 뒀다. `A`의 성분이 `b`보다 크면
   절편 차이가 거리에서 사라지는데, 그건 L2에서 피하려던 바로 그 문제다.
   → **가중치 `λ`를 붙인 `[A_r | λ·b_r]`로 두고, λ에 대한 민감도를 재는 것이 정직하다.**

---

## 3. D2 — 클러스터링 알고리즘, 그리고 greedy 상계 문제의 실제 해법 ★

### 3.1 Aletheia: agglomerative + 연결성 제약 + refit [코드]

`MergerClassifier`의 실제 시그니처:

```
AgglomerativeClustering + kneighbors_graph(connectivity)
  linkage      : average
  metric       : euclidean
  n_neighbors  : "두 LLM은 연결돼 있을 때만 병합될 수 있다"
  min_samples  : 병합 영역의 최소 샘플 수 (예제값 30)
  n_clusters   : GridSearchCV 1~20, scoring = AUC
  refit_model  : LogisticRegression()
```

셋 다 우리에게 직접 쓸모가 있다:

1. **average linkage + euclidean** — D2에서 유력하다고 적어둔 것과 같다. 검증됨.
2. **kNN 연결성 제약** — 밀집 거리행렬 O(R²) 대신 희소 그래프. D3의 우회로.
   부수효과로 "멀리 떨어진 영역은 계수가 비슷해도 안 합쳐진다"는 제약이 생긴다.
3. **`refit_model` ← 제일 중요하다.**
   병합 후 affine map을 **평균내지 않고, 그 클러스터에 속한 데이터로 로지스틱 회귀를 다시 적합**한다.
   → 우리가 상정한 "클러스터 대표 = 평균 `[A|b]`"보다 ε이 **항상 낮게** 나온다.
   → **우리 ε-path와 Aletheia 숫자는 직접 비교 불가.** refit을 넣거나, 안 넣는다면 왜 다른지 적어야 한다.

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

### 3.3 polytope lens: HDBSCAN [요약]

밀도 기반이라 **K를 정하지 않는다.** 그리고 "알고리즘 자체는 중요하지 않다 — 충분히 가까운
활성/코드를 묶는 것이면 무엇이든 된다"는 입장이라고 알려져 있다. (원문 확인 필요)

---

## 4. D3 — MNIST 규모 문제 ★

### 4.1 Aletheia의 두 번째 알고리즘이 답이다 [코드]

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

### 4.2 `min_samples` 필터는 우리에게 전이되지 않는다 [코드] ⚠️

Aletheia는 서브샘플링 대신 **`min_samples`(예제값 30)와 `topk`로 거른다.**
샘플이 적은 영역은 애초에 병합 후보에서 뺀다. 근거는 명확하다 — 점 하나짜리 영역에서는
국소 선형모델을 신뢰성 있게 **재적합할 수 없으니까**(그들은 refit을 하므로 이게 필수다).

**그런데 우리 MNIST는 R/N = 1.000이라 모든 영역의 샘플 수가 정확히 1이다.**
`min_samples = 30`은 **모든 영역을 삭제한다.** Aletheia가 다룬 건 영역당 점이 여럿인
저차원·표형 데이터였다.

→ 이건 실패가 아니라 **관측 결과다.** stage1_log의 R/N = 1.000이 갖는 또 하나의 함의:
   **선행연구의 표준 완화책(샘플 수 필터 + 클러스터별 refit)이 MNIST에서는 원천적으로 무효다.**
   refit할 데이터가 영역당 1점뿐이므로 refit 자체가 정의되지 않는다.
   → **MNIST에서는 평균 `[A|b]` 대표값을 쓸 수밖에 없고, 2D에서는 refit을 쓸 수 있다.**
   두 세팅에서 방법이 달라지는 것을 숨기지 말고 명시할 것.

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

### 6.1 Q1 (라우팅 비용) ← Aletheia의 `flatten` [코드] ★

`FlattenClassifier` docstring:

> This method return a **flattened single hidden layer network**, where the first hidden weights
> are **initialized using the local linear models of merged regions.**

즉 병합된 영역들을 **실제 1층 신경망으로 되돌린다.**
→ `open_questions.md` Q1("`k(x)`를 알려면 원본 네트워크 첫 층을 돌려야 하므로 우리 설명은
   자립적이지 않다")에 대한 선행연구의 답이다: **병합 결과를 다시 네트워크로 컴파일하면
   라우팅이 오라클 호출이 아니라 그 네트워크의 일부가 된다.**
   Stage 5(c)로 미뤄둔 정산을 Stage 1에서 부분적으로 할 수 있다는 뜻.

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

| | 선행연구 답 | `open_questions.md` 잠정안 | 판정 · 제안 |
|---|---|---|---|
| **D1** | 출력 divergence 기준, Frobenius는 대리 | 동일 | ✅ 유지. **단 `[A_r \| λ·b_r]`의 λ 민감도 추가.** cosine은 "우리 추가분"으로 명시 |
| **D2** | agglomerative+average+**kNN 연결성**, 병합 후 **refit**, **확률적 선택(γ=0.2)** | agglomerative+average | ⚠️ 연결성·refit·확률선택 **셋 다 누락** → 추가 |
| **D3** | **`Pruner`: topk core + 최근접 배정, O(R·K)** | 랜덤 서브샘플 M | ✅ **M을 정할 필요가 없어짐.** 2D에서 Merger vs Pruner 격차 측정 |
| **D4** | 도메인 출력 척도 **하나**, 명시 | 둘 이상 병기 | ❌ 규범과 어긋남 → **KL 하나를 기준으로**, 정확도는 보조 표시 |
| **상계** | hierarchical + **확률적 재시작**, 손잡이 함께 보고 | greedy 상계임을 명시만 | ⚠️ 확률적 선택 구현 (10줄) |
| **검증** | **알려진 메커니즘 앵커 필수** | Q2에 "설계해둘 것" | ⚠️ **4조각 앵커를 먼저 통과시킬 것** |
| **곡선** | Pareto front, **dense/chance 종점 포함** | 종점 언급 없음 | ⚠️ K=R, K=1, 랜덤초기화 대조군 추가 |

**결론: D3만 열려 있다고 생각했는데, 실제로는 D3이 제일 깨끗하게 풀렸고
D2와 D4에서 우리가 놓치고 있던 것이 더 많았다.**

---

## 8. 아직 확인 못 한 것 — 로컬로 넘김

아래는 이 세션에서 **원문을 못 읽어서 요약 수준에 머문 것들**이다.
`docs/local_session_manual.md`에 자료별 URL과 "무엇을 뽑을지" 질문 목록이 있다.

| 미확인 항목 | 왜 중요한가 |
|---|---|
| Aletheia **논문**의 병합 기준 서술 | 코드에서 `euclidean` + `all_bias_weight`까지는 봤지만, **절편 가중치를 실제로 어떻게 정했는지**가 D1의 λ에 직결 |
| polytope lens의 클러스터링 대상 | Frobenius를 **정의만** 했는지 **실제로 썼는지**. 실제로는 활성 위에서 HDBSCAN을 돌린 것으로 보이는데, 그렇다면 "Frobenius를 쓴 선행연구"는 사실상 없다 |
| Hanin & Rolnick의 샘플링 프로토콜 | 직선을 몇 개, 어떻게 뽑는지. 우리 P5(선분 200개 × 4000스텝)의 근거가 됨 |
| Srivastava 2015의 클러스터링 | README가 "활성 코드 클러스터링"이라 적어뒀는데 **실제로 클러스터링을 했는지, 검색용 해싱을 한 건지** 불명 |
| Chu et al. 2018 (OpenBox) | README에 있는데 이번 조사에서 아무것도 확인 못 함 |
