"""마크다운 문서를 HTML로 변환하면서 기호에 hover 툴팁을 자동으로 붙인다.

문제: 문서를 고치다 보면 기호(K, ε, A_r ...)가 정의보다 먼저 나오게 된다.
      실제로 walkthrough_stage1.md 에서 K가 214줄에서 쓰이고 358줄에서 정의됐다.

해결: 정의를 docs/symbols.md 한 곳에만 두고, 문서에서 백틱으로 감싼 기호에
      마우스를 올리면 정의가 뜨게 만든다. 문서는 마크다운 하나만 관리하면 된다.

왜 백틱만 잡는가: 본문 아무 데나 있는 'K'를 잡으면 오탐이 쏟아진다.
      `K` 처럼 백틱으로 감싼 것만 잡으면 오탐이 0이다.
      (부수 효과: 기호를 백틱으로 쓰는 습관이 강제된다. 좋은 습관이다.)

사용법:
    python scripts/build_docs.py            # artifacts/docs_html/ 에 생성
    python scripts/build_docs.py --check    # 정의 없는 기호만 보고하고 끝
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mlpinterp.symbols import SYMBOLS_MD, load_symbols  # noqa: E402
from mlpinterp.utils import ARTIFACTS  # noqa: E402

DOCS = REPO / "docs"
OUT = ARTIFACTS / "docs_html"

# 이 순서로 목차에 나온다
PAGES = [
    ("walkthrough_stage1.md", "걸어서 따라가기", "유닛 1개에서 ε-path까지 한 걸음씩"),
    ("symbols.md", "기호표", "모든 기호의 정의 원본"),
    ("concepts.md", "개념 사전", "정의와 용어. 통독용이 아니라 찾아보는 용도"),
    ("stage1_log.md", "실행 로그", "예측 P1~P6과 실측 결과"),
    ("open_questions.md", "현재 위치와 열린 질문", "다음에 할 일, 미결정 사항, 교훈"),
]


# ---------------------------------------------------------------- 기호 파싱


# 파서는 src/mlpinterp/symbols.py 에 있다. 터미널 범례(walkthrough.py)와
# HTML 툴팁(이 파일)이 같은 원본을 읽어야 정의가 어긋나지 않는다.


# ---------------------------------------------------------------- 툴팁 주입


CODE_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)
PRE_RE = re.compile(r"<pre>.*?</pre>", re.DOTALL)

# 기호 앞뒤가 이 문자면 매칭하지 않는다 (단어 중간을 잡는 것을 막는다).
# 예: 'python' 안의 h, 'xdg-open' 안의 x 를 걸러낸다.
WORDCHAR = r"A-Za-z0-9_"


def _symbol_regex(syms: dict[str, str]) -> re.Pattern:
    """기호 전체를 하나의 교대(alternation) 정규식으로. 긴 것부터 매칭한다.

    긴 것 우선이 중요하다: 'A_r'을 'A'보다 먼저 시도해야
    'A_r'이 'A' + '_r'로 쪼개지지 않는다.
    """
    keys = sorted(syms, key=len, reverse=True)
    body = "|".join(re.escape(k) for k in keys)
    return re.compile(f"(?<![{WORDCHAR}])({body})(?![{WORDCHAR}])")


def inject_tooltips(html_body: str, syms: dict[str, str]) -> tuple[str, set[str], set[str]]:
    """인라인 코드 스팬 '안에서' 기호를 찾아 그 부분만 툴팁 span으로 감싼다.

    `K = 60,000` 이나 `2^h` 처럼 복합 표현 안에 기호가 섞여 있는 경우가 대부분이라
    코드 스팬 전체가 기호와 정확히 일치하기를 기대할 수 없다.

    <pre> 블록(펜스 코드)은 통째로 건너뛴다 — 거기까지 툴팁을 달면 시끄럽다.

    Returns: (변환된 html, 매칭된 기호들, 아무 기호도 못 찾은 짧은 코드스팬들)
    """
    hit: set[str] = set()
    miss: set[str] = set()
    sym_re = _symbol_regex(syms)

    def wrap_symbols(m: re.Match) -> str:
        inner = m.group(1)
        found = False

        def one(sm: re.Match) -> str:
            nonlocal found
            key = html.unescape(sm.group(1))
            if key not in syms:
                return sm.group(0)
            found = True
            hit.add(key)
            definition = html.escape(syms[key], quote=True)
            return f'<span class="sym" data-def="{definition}">{sm.group(1)}</span>'

        new_inner = sym_re.sub(one, inner)
        if not found:
            text = html.unescape(inner)
            # 짧은데 아무 기호도 없다 -> 정의를 빠뜨렸을 가능성. 보고용.
            if len(text) <= 14 and "\n" not in text and not text.startswith(("python ", "uv ")):
                miss.add(text)
        return f"<code>{new_inner}</code>"

    # <pre> 블록은 건드리지 않는다
    parts = []
    last = 0
    for pm in PRE_RE.finditer(html_body):
        parts.append(CODE_RE.sub(wrap_symbols, html_body[last : pm.start()]))
        parts.append(pm.group(0))
        last = pm.end()
    parts.append(CODE_RE.sub(wrap_symbols, html_body[last:]))
    return "".join(parts), hit, miss


# ---------------------------------------------------------------- 페이지


CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e2e2e2;--code-bg:#f4f4f5;
--sym-bg:#fff3cd;--sym-line:#e0a800;--tip-bg:#1f2430;--tip-fg:#f2f4f8;--accent:#0b62d0}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e6e8eb;--mut:#9aa1ab;
--line:#2a2f38;--code-bg:#1e222a;--sym-bg:#3a3520;--sym-line:#b8901f;
--tip-bg:#f2f4f8;--tip-fg:#14161a;--accent:#63a4ff}}
:root[data-theme=dark]{--bg:#14161a;--fg:#e6e8eb;--mut:#9aa1ab;--line:#2a2f38;
--code-bg:#1e222a;--sym-bg:#3a3520;--sym-line:#b8901f;--tip-bg:#f2f4f8;
--tip-fg:#14161a;--accent:#63a4ff}
:root[data-theme=light]{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e2e2e2;
--code-bg:#f4f4f5;--sym-bg:#fff3cd;--sym-line:#e0a800;--tip-bg:#1f2430;
--tip-fg:#f2f4f8;--accent:#0b62d0}

body{background:var(--bg);color:var(--fg);margin:0;
font-family:-apple-system,BlinkMacSystemFont,"Noto Sans KR","Segoe UI",sans-serif;
line-height:1.75;font-size:16px}
.wrap{max-width:900px;margin:0 auto;padding:2rem 1.25rem 6rem}
h1,h2,h3{line-height:1.3;margin-top:2.2em}
h1{font-size:1.9rem;border-bottom:2px solid var(--line);padding-bottom:.4em}
h2{font-size:1.4rem;border-bottom:1px solid var(--line);padding-bottom:.3em}
h3{font-size:1.15rem}
a{color:var(--accent)}
hr{border:0;border-top:1px solid var(--line);margin:2.5em 0}
blockquote{border-left:4px solid var(--sym-line);margin:1.2em 0;padding:.4em 1.1em;
background:var(--code-bg);border-radius:0 6px 6px 0}
code{background:var(--code-bg);padding:.15em .38em;border-radius:4px;
font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.88em}
pre{background:var(--code-bg);padding:1em;border-radius:8px;overflow-x:auto}
pre code{background:none;padding:0;font-size:.85em;line-height:1.5}
.tablewrap{overflow-x:auto;margin:1.2em 0}
table{border-collapse:collapse;width:100%;font-size:.93em}
th,td{border:1px solid var(--line);padding:.5em .7em;text-align:left}
th{background:var(--code-bg);font-weight:600}

/* ---- 기호 툴팁 ---- */
.sym{position:relative;background:var(--sym-bg);border-bottom:2px dotted var(--sym-line);
padding:.15em .38em;border-radius:4px;cursor:help;
font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.88em}
.sym::after{content:attr(data-def);position:absolute;left:50%;bottom:calc(100% + 10px);
transform:translateX(-50%);background:var(--tip-bg);color:var(--tip-fg);
padding:.6em .85em;border-radius:7px;font-size:13px;line-height:1.5;font-family:
-apple-system,BlinkMacSystemFont,"Noto Sans KR",sans-serif;white-space:normal;
width:max-content;max-width:min(380px,80vw);text-align:left;
opacity:0;visibility:hidden;transition:opacity .12s;z-index:50;
box-shadow:0 6px 24px rgba(0,0,0,.28);pointer-events:none}
.sym::before{content:"";position:absolute;left:50%;bottom:calc(100% + 4px);
transform:translateX(-50%);border:6px solid transparent;border-top-color:var(--tip-bg);
opacity:0;visibility:hidden;transition:opacity .12s;z-index:51;pointer-events:none}
.sym:hover::after,.sym:hover::before,.sym:focus::after,.sym:focus::before{
opacity:1;visibility:visible}

nav.top{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);
padding:.7em 0;margin-bottom:1.5em;font-size:.88rem;z-index:100}
nav.top a{margin-right:1.1em;text-decoration:none}
nav.top a.here{font-weight:700;text-decoration:underline}
.hint{color:var(--mut);font-size:.85rem;margin:.4em 0 0}
"""

HINT = (
    '<p class="hint">노란 배경의 기호에 <b>마우스를 올리면 정의가 뜹니다.</b> '
    '정의 원본은 <a href="symbols.html">기호표</a>에 있습니다.</p>'
)


def nav(current: str) -> str:
    links = []
    for fname, title, _ in PAGES:
        stem = Path(fname).stem
        cls = ' class="here"' if fname == current else ""
        links.append(f'<a href="{stem}.html"{cls}>{title}</a>')
    return f'<nav class="top">{"".join(links)}</nav>'


def wrap_tables(body: str) -> str:
    """표를 가로 스크롤 컨테이너로 감싼다 (좁은 화면에서 페이지가 안 밀리도록)."""
    return body.replace("<table>", '<div class="tablewrap"><table>').replace(
        "</table>", "</table></div>"
    )


def build_page(md_path: Path, title: str, syms: dict, md) -> tuple[str, set, set]:
    body = md.render(md_path.read_text(encoding="utf-8"))
    body, hit, miss = inject_tooltips(body, syms)
    body = wrap_tables(body)
    page = (
        f"<!doctype html><html lang=ko><head><meta charset=utf-8>"
        f'<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
        f'<div class="wrap">{nav(md_path.name)}{HINT}{body}</div></body></html>'
    )
    return page, hit, miss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="정의 없는 기호만 보고")
    args = ap.parse_args()

    from markdown_it import MarkdownIt

    syms = load_symbols(str(SYMBOLS_MD))
    print(f"기호 {len(syms)}개 로드 (docs/symbols.md)\n")

    # gfm-like는 표를 지원한다. linkify는 별도 패키지가 필요하고
    # 우리 문서엔 자동 링크가 필요 없으므로 끈다.
    md = MarkdownIt("gfm-like", {"linkify": False})
    OUT.mkdir(parents=True, exist_ok=True)

    all_hit: set[str] = set()
    all_miss: dict[str, list[str]] = {}

    for fname, title, _desc in PAGES:
        src = DOCS / fname
        if not src.exists():
            print(f"  [skip] {fname} 없음")
            continue
        page, hit, miss = build_page(src, title, syms, md)
        all_hit |= hit
        for m in miss:
            all_miss.setdefault(m, []).append(fname)
        if not args.check:
            (OUT / f"{src.stem}.html").write_text(page, encoding="utf-8")
        print(f"  {fname:26s} 기호 {len(hit):3d}개 적용")

    unused = sorted(set(syms) - all_hit)
    if unused:
        print(f"\n[정보] 정의돼 있지만 아직 문서에서 안 쓰인 기호 {len(unused)}개:")
        print(f"       {', '.join(unused)}")

    if all_miss:
        print(f"\n[확인 필요] 백틱으로 쓰였지만 symbols.md에 정의가 없는 것 {len(all_miss)}개:")
        for k in sorted(all_miss)[:30]:
            print(f"       `{k}`  ({', '.join(sorted(set(all_miss[k])))})")
        print("       기호라면 docs/symbols.md 에 추가하세요. 코드 조각이면 무시해도 됩니다.")

    if not args.check:
        print(f"\n생성됨 -> {OUT}")
        print(f"열기:   xdg-open {OUT / 'walkthrough_stage1.html'}")


if __name__ == "__main__":
    main()
