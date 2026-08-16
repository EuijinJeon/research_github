"""마크다운 문서를 HTML로 변환하면서 기호에 hover 툴팁을 자동으로 붙인다.

문제: 문서를 고치다 보면 기호(K, ε, A_r ...)가 정의보다 먼저 나오게 된다.
      실제로 walkthrough_stage1.md 에서 K가 214줄에서 쓰이고 358줄에서 정의됐다.

해결: 정의를 docs/symbols.md 한 곳에만 두고, 문서에서 백틱으로 감싼 기호에
      마우스를 올리면 정의가 뜨게 만든다. 문서는 마크다운 하나만 관리하면 된다.

왜 백틱만 잡는가: 본문 아무 데나 있는 'K'를 잡으면 오탐이 쏟아진다.
      `K` 처럼 백틱으로 감싼 것만 잡으면 오탐이 0이다.
      (부수 효과: 기호를 백틱으로 쓰는 습관이 강제된다. 좋은 습관이다.)

사용법:
    python scripts/build_docs.py            # 문서별 HTML (artifacts/docs_html/)
    python scripts/build_docs.py --single   # 전부 한 페이지로 + 그림 내장
    python scripts/build_docs.py --check    # 정의 없는 기호만 보고하고 끝
"""

from __future__ import annotations

import argparse
import base64
import html
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mlpinterp.symbols import SYMBOLS_MD, load_symbols  # noqa: E402
from mlpinterp.utils import ARTIFACTS  # noqa: E402

DOCS = REPO / "docs"
FIGS = ARTIFACTS / "figures"
OUT = ARTIFACTS / "docs_html"

PAGES = [
    ("walkthrough_stage1.md", "걸어서 따라가기", "유닛 1개에서 ε-path까지 한 걸음씩"),
    ("symbols.md", "기호표", "모든 기호의 정의 원본"),
    ("concepts.md", "개념 사전", "정의와 용어. 통독용이 아니라 찾아보는 용도"),
    ("stage1_log.md", "실행 로그", "예측 P1~P6과 실측 결과"),
    ("open_questions.md", "현재 위치와 열린 질문", "다음에 할 일, 미결정 사항, 교훈"),
]

# 단일 페이지에 내장할 그림과 설명
FIGURES = [
    ("walkthrough_buildup.png", "ReLU 유닛을 하나씩 켜면 평면이 쪼개진다",
     "유닛 1→64개. 조각 수는 2^h 가 아니라 훨씬 느리게 늘어난다 (64개 → 1,378조각, 2^64 = 1.8×10¹⁹)."),
    ("polytopes_moons.png", "moons: 학습된 모델의 실제 폴리토프 분할",
     "왼쪽부터 영역 지도 / 첫 층 직선 겹침 / 결정 경계. 위 hidden=[64], 아래 hidden=[32,32]."),
    ("polytopes_spiral.png", "spiral: 같은 분석",
     "나선은 경계가 여러 번 감겨 더 많은 조각을 요구한다. 1층 95.5% vs 2층 100.0%."),
    ("zero_sets_moons.png", "moons: 유닛 하나의 영점 집합",
     "회색이 첫 층 직선. 첫 층 유닛은 전역 직선, 둘째 층 유닛은 회색 선 위에서만 꺾인다."),
    ("zero_sets_spiral.png", "spiral: 같은 분석", "꺾임점의 100%가 첫 층 직선 위에 있다 (대조군 대비 15배 분리)."),
]


# ---------------------------------------------------------------- 툴팁 주입

CODE_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)
PRE_RE = re.compile(r"<pre>.*?</pre>", re.DOTALL)

# 기호 앞뒤가 이 문자면 매칭하지 않는다 ('python' 안의 h 같은 것을 막는다).
WORDCHAR = r"A-Za-z0-9_"


def _symbol_regex(syms: dict[str, str]) -> re.Pattern:
    """기호 전체를 하나의 교대 정규식으로. 긴 것부터 매칭한다.

    긴 것 우선이 중요하다: 'A_r'을 'A'보다 먼저 시도해야
    'A_r'이 'A' + '_r' 로 쪼개지지 않는다.
    """
    keys = sorted(syms, key=len, reverse=True)
    return re.compile(f"(?<![{WORDCHAR}])({'|'.join(re.escape(k) for k in keys)})(?![{WORDCHAR}])")


def inject_tooltips(body: str, syms: dict[str, str]) -> tuple[str, set[str], set[str]]:
    """인라인 코드 스팬 '안에서' 기호를 찾아 그 부분만 툴팁 span으로 감싼다.

    `K = 60,000` 이나 `2^h` 처럼 복합 표현 안에 기호가 섞인 경우가 대부분이라
    코드 스팬 전체가 기호와 정확히 일치하기를 기대할 수 없다.
    <pre> 블록은 통째로 건너뛴다 — 거기까지 툴팁을 달면 시끄럽다.
    """
    hit: set[str] = set()
    miss: set[str] = set()
    sym_re = _symbol_regex(syms)

    def wrap(m: re.Match) -> str:
        inner = m.group(1)
        found = False

        def one(sm: re.Match) -> str:
            nonlocal found
            key = html.unescape(sm.group(1))
            if key not in syms:
                return sm.group(0)
            found = True
            hit.add(key)
            d = html.escape(syms[key], quote=True)
            return f'<button type="button" class="sym" data-def="{d}">{sm.group(1)}</button>'

        new_inner = sym_re.sub(one, inner)
        if not found:
            text = html.unescape(inner)
            if len(text) <= 14 and "\n" not in text and not text.startswith(("python ", "uv ")):
                miss.add(text)
        return f"<code>{new_inner}</code>"

    parts, last = [], 0
    for pm in PRE_RE.finditer(body):
        parts.append(CODE_RE.sub(wrap, body[last : pm.start()]))
        parts.append(pm.group(0))
        last = pm.end()
    parts.append(CODE_RE.sub(wrap, body[last:]))
    return "".join(parts), hit, miss


# ---------------------------------------------------------------- 구조 가공

H_RE = re.compile(r"<h([123])>(.*?)</h\1>", re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


def add_ids(body: str, slug: str) -> tuple[str, list[tuple[int, str, str]]]:
    """h1~h3에 고유 id를 붙이고 목차 항목을 수집한다.

    id는 슬러그+번호로 만든다. 한글 제목을 슬러그화하면 충돌과 인코딩 문제가
    생기므로, 안정적인 번호를 쓰는 편이 낫다.
    """
    toc: list[tuple[int, str, str]] = []
    n = 0

    def repl(m: re.Match) -> str:
        nonlocal n
        n += 1
        level, inner = int(m.group(1)), m.group(2)
        hid = f"{slug}-{n}"
        label = html.unescape(TAG_RE.sub("", inner)).strip()
        toc.append((level, hid, label))
        return f'<h{level} id="{hid}">{inner}</h{level}>'

    return H_RE.sub(repl, body), toc


def wrap_blocks(body: str) -> str:
    """표와 코드 블록을 가로 스크롤 컨테이너로 감싼다 (본문이 옆으로 안 밀리게)."""
    body = body.replace("<table>", '<div class="scroller"><table>')
    body = body.replace("</table>", "</table></div>")
    return body


# ---------------------------------------------------------------- 디자인 토큰

CSS = """
/* 색: 주제가 '평면을 직선으로 자르기'라 제도(drafting) 어휘를 쓴다.
   중립색은 슬레이트로 살짝 기울이고, 강조는 딥 티일 하나.
   앰버는 장식이 아니라 기능색 — '이 기호는 만질 수 있다'는 신호. */
:root{
  --bg:#fbfcfd; --surface:#eef2f5; --raise:#ffffff;
  --ink:#13171c; --muted:#59626d; --line:#dbe1e7; --line-strong:#c3ccd4;
  --accent:#0f6b62; --accent-soft:#d8ecea;
  --sym-bg:#fdf1d2; --sym-edge:#c08d14; --sym-ink:#5b4408;
  --tip-bg:#171d24; --tip-ink:#eef2f5;
  --shadow:0 1px 2px rgba(19,23,28,.05), 0 8px 28px rgba(19,23,28,.09);
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#101418; --surface:#191f26; --raise:#1d242c;
    --ink:#e3e8ec; --muted:#929ca7; --line:#262e37; --line-strong:#39434e;
    --accent:#5ed0c2; --accent-soft:#14312e;
    --sym-bg:#372f18; --sym-edge:#b18b28; --sym-ink:#f2e2b8;
    --tip-bg:#eef2f5; --tip-ink:#13171c;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 30px rgba(0,0,0,.5);
  }
}
:root[data-theme=light]{
  --bg:#fbfcfd; --surface:#eef2f5; --raise:#ffffff;
  --ink:#13171c; --muted:#59626d; --line:#dbe1e7; --line-strong:#c3ccd4;
  --accent:#0f6b62; --accent-soft:#d8ecea;
  --sym-bg:#fdf1d2; --sym-edge:#c08d14; --sym-ink:#5b4408;
  --tip-bg:#171d24; --tip-ink:#eef2f5;
  --shadow:0 1px 2px rgba(19,23,28,.05), 0 8px 28px rgba(19,23,28,.09);
}
:root[data-theme=dark]{
  --bg:#101418; --surface:#191f26; --raise:#1d242c;
  --ink:#e3e8ec; --muted:#929ca7; --line:#262e37; --line-strong:#39434e;
  --accent:#5ed0c2; --accent-soft:#14312e;
  --sym-bg:#372f18; --sym-edge:#b18b28; --sym-ink:#f2e2b8;
  --tip-bg:#eef2f5; --tip-ink:#13171c;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 30px rgba(0,0,0,.5);
}

/* 타입: CSP가 폰트 CDN을 막고 한글 웹폰트는 인라인하기엔 수 MB라 불가능하다.
   그래서 서체 대비 대신 크기/굵기/자간 + 산세리프↔모노 대비로 위계를 만든다. */
:root{
  --sans:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Pretendard",
         "Noto Sans KR","Malgun Gothic","Segoe UI",sans-serif;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,
         "D2Coding","Noto Sans Mono CJK KR",monospace;
}

*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:16.5px;line-height:1.78;-webkit-font-smoothing:antialiased}

/* 레이아웃: 긴 기술 노트라 좌측 고정 목차 + 본문 72ch */
.shell{display:grid;grid-template-columns:1fr;gap:0;max-width:1320px;margin:0 auto}
@media (min-width:1080px){
  .shell{grid-template-columns:264px minmax(0,1fr);gap:48px;padding:0 32px}
}

/* --- 좌측 레일 --- */
.rail{position:sticky;top:0;align-self:start;max-height:100vh;overflow-y:auto;
  padding:28px 0 56px;border-bottom:1px solid var(--line)}
@media (min-width:1080px){.rail{border-bottom:0;padding:40px 0 56px}}
.rail-head{font-family:var(--mono);font-size:11px;letter-spacing:.15em;
  text-transform:uppercase;color:var(--muted);padding:0 20px 12px;
  border-bottom:1px solid var(--line);margin-bottom:14px}
@media (min-width:1080px){.rail-head{padding-left:0;padding-right:0}}
.rail nav{display:flex;flex-direction:column;gap:1px;padding:0 12px}
@media (min-width:1080px){.rail nav{padding:0}}
.rail a{color:var(--muted);text-decoration:none;font-size:13.5px;line-height:1.5;
  padding:5px 10px;border-radius:6px;border-left:2px solid transparent}
.rail a:hover{color:var(--ink);background:var(--surface)}
.rail a.lv1{color:var(--ink);font-weight:650;margin-top:16px;font-size:14px}
.rail a.lv1:first-child{margin-top:0}
.rail a.lv3{padding-left:24px;font-size:12.5px}
.rail a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

/* --- 본문 --- */
main{padding:32px 20px 120px;min-width:0}
@media (min-width:1080px){main{padding:56px 0 160px}}
.col{max-width:72ch}

.masthead{margin-bottom:56px;padding-bottom:28px;border-bottom:2px solid var(--ink)}
.eyebrow{font-family:var(--mono);font-size:11.5px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--accent);margin:0 0 14px}
.masthead h1{font-size:clamp(1.9rem,4.4vw,2.7rem);line-height:1.18;margin:0 0 14px;
  letter-spacing:-.022em;font-weight:750;text-wrap:balance}
.dek{color:var(--muted);font-size:1.02rem;margin:0;max-width:60ch}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:22px}
.chip{font-family:var(--mono);font-size:11.5px;color:var(--muted);
  border:1px solid var(--line-strong);border-radius:999px;padding:3px 11px;
  font-variant-numeric:tabular-nums}

.doc{margin-bottom:96px;scroll-margin-top:20px}
.doc + .doc{padding-top:72px;border-top:1px solid var(--line)}

h1,h2,h3{letter-spacing:-.018em;text-wrap:balance;scroll-margin-top:20px}
.doc h1{font-size:1.72rem;font-weight:730;margin:0 0 8px;line-height:1.25}
h2{font-size:1.24rem;font-weight:700;margin:2.6em 0 .7em;padding-bottom:.35em;
  border-bottom:1px solid var(--line)}
h3{font-size:1.04rem;font-weight:680;margin:2.1em 0 .5em;color:var(--ink)}
p{margin:0 0 1.05em}
ul,ol{padding-left:1.35em;margin:0 0 1.05em}
li{margin-bottom:.35em}
a{color:var(--accent);text-underline-offset:3px}
hr{border:0;border-top:1px solid var(--line);margin:2.8em 0}
strong{font-weight:680}

blockquote{margin:1.5em 0;padding:.95em 1.25em;background:var(--surface);
  border-left:3px solid var(--accent);border-radius:0 8px 8px 0}
blockquote > :last-child{margin-bottom:0}

code{font-family:var(--mono);font-size:.855em;background:var(--surface);
  padding:.14em .4em;border-radius:5px;border:1px solid var(--line)}
pre{background:var(--surface);border:1px solid var(--line);border-radius:10px;
  padding:1.1em 1.25em;overflow-x:auto;margin:1.5em 0}
pre code{background:none;border:0;padding:0;font-size:.83em;line-height:1.62}

.scroller{overflow-x:auto;margin:1.6em 0;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:.93rem;
  font-variant-numeric:tabular-nums}
th,td{padding:.62em .85em;text-align:left;border-bottom:1px solid var(--line);
  vertical-align:top}
thead th{background:var(--surface);font-weight:660;font-size:.86rem;
  letter-spacing:.01em;white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
.scroller table code{white-space:nowrap}

/* --- 기호 툴팁: 앰버는 '만질 수 있다'는 기능 신호 --- */
.sym{position:relative;display:inline;font:inherit;font-family:var(--mono);
  font-size:.98em;background:var(--sym-bg);color:var(--sym-ink);
  border:0;border-bottom:2px dotted var(--sym-edge);border-radius:4px;
  padding:.06em .3em;margin:0;cursor:help}
.sym:hover,.sym:focus-visible{background:var(--sym-edge);color:var(--bg)}
.sym:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.sym::after{content:attr(data-def);position:absolute;left:50%;bottom:calc(100% + 11px);
  transform:translateX(-50%) translateY(3px);
  background:var(--tip-bg);color:var(--tip-ink);padding:.7em .9em;border-radius:9px;
  font-family:var(--sans);font-size:13px;line-height:1.58;font-weight:420;
  letter-spacing:0;text-align:left;white-space:normal;width:max-content;
  max-width:min(360px,74vw);opacity:0;visibility:hidden;z-index:60;
  box-shadow:var(--shadow);pointer-events:none;
  transition:opacity .13s ease,transform .13s ease}
.sym::before{content:"";position:absolute;left:50%;bottom:calc(100% + 5px);
  transform:translateX(-50%);border:6px solid transparent;
  border-top-color:var(--tip-bg);opacity:0;visibility:hidden;z-index:61;
  pointer-events:none;transition:opacity .13s ease}
.sym:hover::after,.sym:focus-visible::after{opacity:1;visibility:visible;
  transform:translateX(-50%) translateY(0)}
.sym:hover::before,.sym:focus-visible::before{opacity:1;visibility:visible}
@media (prefers-reduced-motion:reduce){
  .sym::after,.sym::before{transition:none}
  .sym:hover::after,.sym:focus-visible::after{transform:translateX(-50%)}
}

/* --- 그림 --- */
figure{margin:2.4em 0;padding:0}
figure img{width:100%;height:auto;display:block;border:1px solid var(--line);
  border-radius:10px;background:var(--raise)}
figcaption{margin-top:.8em;font-size:.87rem;color:var(--muted);line-height:1.6}
figcaption b{color:var(--ink);font-weight:650;display:block;margin-bottom:.15em;
  font-size:.94rem}

.note{background:var(--accent-soft);border:1px solid var(--line);border-radius:10px;
  padding:.9em 1.15em;margin:0 0 2.4em;font-size:.92rem;color:var(--ink)}
.note b{font-weight:680}
"""


def masthead(n_syms: int, n_figs: int) -> str:
    return f"""<header class="masthead">
<p class="eyebrow">Stage 0–1 · 연구 노트</p>
<h1>학습된 MLP의 웨이트를 이해 가능한 형태로 번역하기</h1>
<p class="dek">ReLU MLP를 입력 공간의 다면체 분할로 다시 쓰고, 그 조각들을
얼마나 합칠 수 있는지 재는 프로토콜. 목적은 새 발견이 아니라 개념 이해와 선행연구 재현.</p>
<div class="meta">
<span class="chip">모델 6개 · 검산 전부 PASS</span>
<span class="chip">예측 P1–P6 전부 적중</span>
<span class="chip">기호 {n_syms}개</span>
<span class="chip">그림 {n_figs}장</span>
</div></header>
<p class="note"><b>노란 기호에 마우스를 올리면 정의가 뜹니다.</b>
정의 원본은 <code>docs/symbols.md</code> 한 곳에 있고, 이 페이지의 툴팁과
터미널 출력의 범례가 모두 그 파일 하나를 읽습니다.</p>"""


def figures_section(slug: str = "fig") -> tuple[str, list[tuple[int, str, str]]]:
    """그림을 data URI로 내장한다. 외부 요청 없이 완전히 자립하는 페이지가 된다."""
    toc = [(1, f"{slug}-0", "그림")]
    out = [f'<section class="doc" id="{slug}-0"><h1>그림</h1>']
    for i, (name, title, cap) in enumerate(FIGURES, start=1):
        p = FIGS / name
        if not p.exists():
            continue
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        hid = f"{slug}-{i}"
        toc.append((2, hid, title))
        out.append(
            f'<h2 id="{hid}">{html.escape(title)}</h2>'
            f'<figure><img src="data:image/png;base64,{b64}" '
            f'alt="{html.escape(title)}" loading="lazy">'
            f"<figcaption><b>{html.escape(name)}</b>{html.escape(cap)}</figcaption></figure>"
        )
    out.append("</section>")
    return "".join(out), toc


def render_toc(items: list[tuple[int, str, str]]) -> str:
    links = "".join(
        f'<a class="lv{lv}" href="#{hid}">{html.escape(label)}</a>' for lv, hid, label in items
    )
    return f'<aside class="rail"><p class="rail-head">목차</p><nav>{links}</nav></aside>'


# ---------------------------------------------------------------- 빌드


def build(single: bool, check: bool) -> None:
    from markdown_it import MarkdownIt

    syms = load_symbols(str(SYMBOLS_MD))
    print(f"기호 {len(syms)}개 로드 (docs/symbols.md)\n")

    # gfm-like는 표를 지원한다. linkify는 별도 패키지가 필요하고 우리 문서엔 불필요.
    md = MarkdownIt("gfm-like", {"linkify": False})

    all_hit: set[str] = set()
    all_miss: dict[str, list[str]] = {}
    sections: list[str] = []
    toc: list[tuple[int, str, str]] = []

    for fname, title, _desc in PAGES:
        src = DOCS / fname
        if not src.exists():
            print(f"  [skip] {fname} 없음")
            continue
        slug = src.stem
        body = md.render(src.read_text(encoding="utf-8"))
        body, hit, miss = inject_tooltips(body, syms)
        body, sec_toc = add_ids(body, slug)
        body = wrap_blocks(body)

        all_hit |= hit
        for m in miss:
            all_miss.setdefault(m, []).append(fname)
        # 목차는 h1/h2만 (h3까지 넣으면 레일이 너무 길어진다)
        toc += [t for t in sec_toc if t[0] <= 2]
        anchor = sec_toc[0][1] if sec_toc else slug
        sections.append(f'<section class="doc" id="{anchor}">{body}</section>')
        print(f"  {fname:26s} 기호 {len(hit):3d}개 적용")

    fig_html, fig_toc = figures_section()
    sections.append(fig_html)
    toc += fig_toc

    unused = sorted(set(syms) - all_hit)
    if unused:
        print(f"\n[정보] 정의는 있으나 아직 문서에서 안 쓰인 기호 {len(unused)}개: {', '.join(unused)}")
    if all_miss:
        print(f"\n[확인 필요] 백틱으로 쓰였지만 symbols.md에 정의가 없는 것 {len(all_miss)}개:")
        for k in sorted(all_miss)[:20]:
            print(f"       `{k}`  ({', '.join(sorted(set(all_miss[k])))})")
        print("       기호라면 docs/symbols.md 에 추가하세요. 코드 조각이면 무시해도 됩니다.")

    if check:
        return

    OUT.mkdir(parents=True, exist_ok=True)
    inner = (
        f'<div class="shell">{render_toc(toc)}'
        f'<main><div class="col">{masthead(len(syms), len(FIGURES))}'
        f'{"".join(sections)}</div></main></div>'
    )
    title = "학습된 MLP를 이해 가능한 형태로 번역하기 — Stage 0–1 연구 노트"

    # 1) 로컬에서 바로 열 수 있는 완전한 HTML
    full = (
        f"<!doctype html><html lang=ko><head><meta charset=utf-8>"
        f'<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
        f"<body>{inner}</body></html>"
    )
    (OUT / "all.html").write_text(full, encoding="utf-8")

    # 2) Artifact용 조각 — 게시할 때 doctype/head/body가 자동으로 씌워진다
    frag = f"<title>{html.escape(title)}</title><style>{CSS}</style>{inner}"
    (OUT / "artifact.html").write_text(frag, encoding="utf-8")

    mb = len(frag.encode("utf-8")) / 1e6
    print(f"\n생성됨 -> {OUT}")
    print(f"  all.html       {len(full)/1e6:5.2f} MB   로컬에서 열기용")
    print(f"  artifact.html  {mb:5.2f} MB   게시용 (한도 16 MB)")
    print(f"  목차 {len(toc)}항목, 그림 {len(FIGURES)}장 내장")
    print(f"\n열기:  xdg-open {OUT / 'all.html'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true", help="(기본 동작. 호환용 플래그)")
    ap.add_argument("--check", action="store_true", help="정의 없는 기호만 보고")
    args = ap.parse_args()
    build(single=True, check=args.check)


if __name__ == "__main__":
    main()
