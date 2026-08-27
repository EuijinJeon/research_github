# Stage 1' 자료 — 공식 구현체와 아직 안 받은 논문

> 방향 전환(`open_questions.md`, 2026-08-19) 직후 원격 세션에서 조사한 것.
> **논문 3편(`2305.03452` → `2406.03947` → `2410.08417`)은 이미 받아뒀으므로 여기 없다.**
> 여기 있는 건 ① 논문의 **공식 코드** ② 아직 안 받은 관련 논문.

---

## 1. 공식 구현체가 있다 — `tdooms/bilinear-decomposition` ★

`2410.08417`의 저자(Thomas Dooms)가 올린 공식 레포다.
**Stage 1' 계획의 네 항목이 거의 그대로 들어 있다.**

```bash
git clone --depth 1 https://github.com/tdooms/bilinear-decomposition.git external/bilinear-decomposition
```

```
src/shared/components.py   Bilinear 층 (20줄)
src/image/model.py         이미지 모델 + decompose() (174줄)
src/image/datasets.py      MNIST, FMNIST 래퍼 (GPU 상주)
src/language/              언어모델 쪽
src/sae/                   SAE 비교군
tutorials/0_introduction.ipynb   개념
tutorials/1_image.ipynb          ← 우리 Stage 1' 과 같은 자리
tutorials/2_language.ipynb
exercises/0_decomposition.ipynb  ← 연습문제 형태
```

`uv` + `python>=3.12` 를 쓴다. **우리 환경과 같다.**

### 1.1 Bilinear 층은 20줄이다

```python
class Bilinear(nn.Linear):
    """A bilinear layer with optional gate and noise"""
    def __init__(self, d_in, d_out, bias=False, gate=None):
        super().__init__(d_in, 2 * d_out, bias=bias)
        self.gate = {"relu": nn.ReLU(), "silu": nn.SiLU(),
                     "gelu": nn.GELU(), None: nn.Identity()}[gate]

    def forward(self, x):
        left, right = super().forward(x).chunk(2, dim=-1)
        return self.gate(left) * right
```

**읽는 법:** 출력 폭을 `2*d_out` 으로 잡아 한 번에 곱한 뒤 반으로 쪼개
좌·우로 쓴다. `gate=None` 이면 `Identity` 라 순수 bilinear —
`g(x) = (W_l x) ⊙ (W_r x)`. `gate="relu"` 면 통상 GLU 가 되므로
**같은 클래스로 대조군까지 만들 수 있다.**

### 1.2 3차 텐서 고유분해는 8줄이다

```python
def decompose(self):
    l, r = self.w_lr[0].unbind()                                    # 좌/우 가중치
    b = einsum(self.w_u, l, r, "cls out, out in1, out in2 -> cls in1 in2")
    b = 0.5 * (b + b.mT)                                            # 대칭화
    vals, vecs = torch.linalg.eigh(b)                               # 고유분해
    vecs = einsum(vecs, self.w_e, "cls emb comp, emb inp -> cls comp inp")
    return vals, vecs
```

클래스마다 `d_in × d_in` 대칭행렬 `B_cls` 하나가 나오고,
그 고유벡터가 **입력공간으로 되돌려져** 이미지로 볼 수 있게 된다.

#### 대칭화가 왜 정당한가 — 손으로 확인

`x^T B x` 는 **B 의 대칭부분만** 본다. 반대칭부분은 이차형식에 0으로 기여한다.

    B = [1 4]      x = (1, 1)
        [0 1]

    x^T B x = 1·1 + 1·4 + 1·0 + 1·1 = 6

    sym(B) = ½(B + Bᵀ) = [1 2]      x^T sym(B) x = 1 + 2 + 2 + 1 = 6   ✅ 같다
                         [2 1]

    skew(B) = ½(B − Bᵀ) = [ 0  2]   x^T skew(B) x = 0 + 2 − 2 + 0 = 0  ✅ 사라진다
                          [−2  0]

즉 `0.5*(b + b.mT)` 는 **정보를 버리는 근사가 아니라 항등식**이다.
그리고 `torch.linalg.eigh` 는 대칭행렬만 받으므로, 대칭화는 **필수 단계**이기도 하다.

### 1.3 우리 프로젝트와 부딪히는 두 가지 ⚠️

**(a) `bias=False` 가 전부 기본값이다.**

```python
self.embed  = Linear(d_input, d_hidden, bias=False)
self.blocks = [Bilinear(d_hidden, d_hidden, bias=bias) for _ in range(n_layer)]   # config.bias = False
self.head   = Linear(d_hidden, d_output, bias=False)
```

절편이 없어야 `f(x) = x^T B x` 라는 **순수 이차형식**이 되기 때문이다.
절편이 있으면 일차항과 상수항이 붙어 3차 텐서 하나로 안 닫힌다
(닫으려면 `[x; 1]` 로 증강해야 한다).

→ **`open_questions.md` L2 의 교훈("절편을 빼먹으면 안 된다")이 여기서 뒤집혀 나타난다.**
   ReLU 영역 분해에서는 절편을 반드시 챙겨야 했는데, bilinear 분해는 **절편을 없애서**
   깔끔함을 산다. 무엇을 포기하고 무엇을 얻는지가 이 대비에 다 들어 있다.

**(b) `wd = 0.5` 다.** 기본 config 가 weight decay 0.5 를 쓴다 (lr=1e-3, epochs=100,
batch=2048, d_hidden=256).

→ **우리 Q3 이 정면으로 걸린다.** 우리는 "정규화가 구조를 인위적으로 단순하게 만들어
   결과를 낙관적으로 보이게 할까 봐" `weight_decay=0` 으로 뒀다.
   그런데 이 계열의 기준 설정은 **wd=0.5** 다.
   → 재현하려면 그들 값을 써야 하고, Q3 를 보려면 **wd 를 축으로 두 번 돌려야 한다.**
   어느 쪽이든 **의식적으로 고르고 기록할 것.** 기본값을 무심코 따라가면 Q3 가 사라진다.

### 1.4 앵커는 그대로 쓸 수 있다

`open_questions.md` 가 "새 방향의 앵커 = `x^T Q x` 항등식 검산"이라고 정해뒀다.
`decompose()` 가 있으니 검산은 한 줄이다 — 임의의 `x` 에 대해

    model(x)[cls]  ==  x_emb^T B_cls x_emb        (bias=False 이므로 정확히 같아야 함)

Stage 1 의 affine 항등식 검산(6개 모델 전부 PASS, 상대오차 ~1e-6)과 **같은 자리**의 검사다.

---

## 2. 아직 안 받은 논문

`bash scripts/fetch_papers.sh` 로 한 번에 받는다. `external/papers/` 에 들어간다.

### 2.1 bilinear 의 뿌리 — GLU 계보 (우선순위 A)

`2410.08417` 은 "bilinear = **element-wise 비선형이 없는 GLU**" 라고 정의한다.
그 GLU 가 어디서 왔는지가 지금 레포에 없다.

| arXiv | 논문 | 왜 |
|---|---|---|
| `1612.08083` | Dauphin et al. 2017, Language Modeling with Gated Convolutional Networks | **GLU 의 출처.** `σ(W₁x) ⊙ (W₂x)` 가 처음 나온 곳 |
| `2002.05202` | Shazeer 2020, GLU Variants Improve Transformer | **"게이트를 선형으로 둬도(=bilinear) 성능이 난다"** 를 실험으로 보인 곳. bilinear 가 성능을 안 잃는다는 주장의 근거 |
| `2003.03828` | Chrysos et al. 2020, Π-nets: Deep Polynomial Neural Networks (CVPR) | 출력이 입력의 **고차 다항식**인 망을 고차 텐서로 구현. bilinear 는 그 2차 특수 케이스 |

### 2.2 아직 안 받은 Stage 1 고전 (우선순위 B)

방향은 바뀌었지만 워크스루에 이미 들어간 내용의 원전이다.

| arXiv / 출처 | 논문 | 왜 |
|---|---|---|
| `1402.1869` | Montúfar et al. 2014, On the Number of Linear Regions | **Zaslavsky 상한을 딥러닝에 들여온 원전.** `concepts.md` §1.4 가 쓰는 공식. Zaslavsky 원본(AMS Memoirs 154, 유료)을 대신한다 |
| `2302.12828` | SplineCam (Humayun et al. 2023) | 2D 슬라이스 영역 분할을 **샘플링 없이 정확히** 계산. Q4 의 또 하나의 도구 (AffineLens `2605.06218` 와 비교용) |
| `1711.02114` | Serra et al. 2018, Bounding and Counting Linear Regions | MILP 정확 열거 |
| PMLR v80 | Balestriero & Baraniuk 2018, A Spline Theory of Deep Networks | polytope lens 의 **'spline code'** 용어 출처. 우리가 STEP 8 에서 해밍 거리를 잰 그 대상 |

### 2.3 Stage 2·3 (우선순위 C)

| arXiv | 논문 |
|---|---|
| `2501.14926` | APD — param-decomp 레포에 마크다운 전문 있음. PDF 는 그림 때문에 |
| `2506.20790` | SPD — 〃 |
| `2301.05217` | Nanda et al. 2023, Progress Measures for Grokking (Stage 3 대조군) |

---

## 3. 라이선스가 실제로 필요한 것은 하나뿐

지금까지 조사한 논문 중 **유료 장벽 뒤에 있는 것은 하나다.**

> **Zaslavsky (1975)**, *Facing up to Arrangements: Face-Count Formulas for
> Partitions of Space by Hyperplanes.* Memoirs of the AMS, no. 154.
> <https://bookstore.ams.org/memo-1-154/>

**다만 받을 필요는 없다.** 우리가 쓰는 건 상한 공식 하나뿐이고,
`1402.1869`(Montúfar 2014)가 그 공식을 딥러닝 맥락에서 다시 서술한다.

나머지는 전부 arXiv 또는 PMLR 무료다. 원격 세션에서 못 읽었던 것은
접근 권한 문제가 아니라 **컨테이너의 egress 차단**이었다
(`local_session_manual.md` §0 의 실측표).
