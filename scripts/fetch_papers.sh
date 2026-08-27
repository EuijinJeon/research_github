#!/usr/bin/env bash
# 아직 안 받은 선행연구 논문 내려받기 → external/papers/
#
# 이미 받은 것 (2026-08-19): 2011.04041 2211.12312 1906.00904 1410.1165 1802.06259
#                            2305.03452 2406.03947 2410.08417
# 여기 있는 건 그 뒤로 추가된 목록이다. 배경은 docs/bilinear_resources.md §2.
#
#   bash scripts/fetch_papers.sh        # 전부
#   bash scripts/fetch_papers.sh A      # A 만 (bilinear 의 뿌리 = GLU 계보)
#
# 이미 있는 파일은 건너뛴다. external/*/ 는 .gitignore 라 추적되지 않는다.
# 유료 장벽 뒤에 있는 것은 Zaslavsky 1975 하나뿐이고, 받을 필요가 없다 (맨 아래 참고).

set -uo pipefail
cd "$(dirname "$0")/.."
OUT="external/papers"
mkdir -p "$OUT"
WANT="${1:-ALL}"

# 우선순위 | 소스 | 식별자 | 저장이름 | 왜 필요한가
PAPERS=$(cat <<'EOF'
A|arxiv|1612.08083|Dauphin2017_Gated_Convolutional_Networks|GLU 의 출처. bilinear 는 게이트를 항등함수로 둔 GLU 다
A|arxiv|2002.05202|Shazeer2020_GLU_Variants|게이트가 선형이어도(=bilinear) 성능이 난다는 실험 근거
A|arxiv|2003.03828|Chrysos2020_Pi_nets_Polynomial_Networks|고차 다항식 망을 고차 텐서로. bilinear 는 그 2차 특수케이스
B|arxiv|1402.1869|Montufar2014_Number_of_Linear_Regions|Zaslavsky 상한의 딥러닝 원전. concepts.md 1.4 의 공식
B|arxiv|2302.12828|Humayun2023_SplineCam|영역 분할을 샘플링 없이 정확 계산. Q4 도구 (AffineLens 와 비교)
B|arxiv|1711.02114|Serra2018_Bounding_Counting_Linear_Regions|MILP 정확 열거
B|mlr|v80/balestriero18b|Balestriero2018_Spline_Theory|'spline code' 용어 출처. STEP 8 해밍 거리의 대상
C|arxiv|2501.14926|Braun2025_APD|Stage 2. param-decomp 레포에 마크다운 전문도 있음
C|arxiv|2506.20790|Bushnaq2025_SPD|Stage 2
C|arxiv|2301.05217|Nanda2023_Grokking_Progress_Measures|Stage 3 대조군
EOF
)

ok=0; skip=0; fail=0; failed=""
while IFS='|' read -r prio src id name why; do
  [ -z "${prio:-}" ] && continue
  [ "$WANT" != "ALL" ] && [ "$prio" != "$WANT" ] && continue

  dest="$OUT/${prio}_${name}.pdf"
  if [ -s "$dest" ]; then echo "  [건너뜀] $name"; skip=$((skip+1)); continue; fi

  case "$src" in
    arxiv) url="https://arxiv.org/pdf/${id}" ;;
    mlr)   url="https://proceedings.mlr.press/${id}/$(basename "$id").pdf" ;;
  esac

  printf "  [받는중] %-48s " "$name"
  if curl -fsSL --retry 3 --retry-delay 2 --max-time 120 -o "$dest.part" "$url" \
     && [ -s "$dest.part" ] && head -c 4 "$dest.part" | grep -q '%PDF'; then
    mv "$dest.part" "$dest"
    # 다른 논문들과 같은 방식으로 텍스트도 뽑아둔다 (없으면 조용히 건너뜀)
    command -v pdftotext >/dev/null && pdftotext -layout "$dest" 2>/dev/null
    echo "OK ($(du -h "$dest" | cut -f1))"; ok=$((ok+1))
  else
    rm -f "$dest.part"; echo "실패"
    fail=$((fail+1)); failed="$failed\n    $name  →  $url"
  fi
done <<< "$PAPERS"

echo
echo "받음 $ok / 건너뜀 $skip / 실패 $fail   →  $OUT/"
[ "$fail" -gt 0 ] && echo -e "실패 목록 (브라우저로 직접):$failed"

cat <<'NOTE'

---
논문 말고 받을 것 — 2410.08417 의 공식 코드 (docs/bilinear_resources.md §1)

  git clone --depth 1 https://github.com/tdooms/bilinear-decomposition.git \
      external/bilinear-decomposition

  Bilinear 층 20줄, decompose() 8줄, MNIST/FMNIST 래퍼, 튜토리얼 3편 + 연습 1편.
  uv + python>=3.12 로 우리 환경과 같다.

arXiv 에 없는 것

  Zaslavsky (1975), Facing up to Arrangements. Memoirs of the AMS no. 154.
    https://bookstore.ams.org/memo-1-154/     ← 유일한 유료 항목
  → 우리는 상한 공식만 쓰므로 필수가 아니다. 위 B 목록의 Montufar 2014 가 대신한다.

  Elhage et al. (2022), Toy Models of Superposition
    https://transformer-circuits.pub/2022/toy_model/index.html
NOTE
