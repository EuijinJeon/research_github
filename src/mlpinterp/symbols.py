"""기호 정의를 docs/symbols.md 한 곳에서 읽어온다.

정의 원본이 하나여야 한다는 것이 요점이다. 같은 정의가
문서 / HTML 툴팁 / 터미널 출력 세 군데에 흩어지면 반드시 어긋난다.
그래서 셋 다 이 모듈을 거쳐 docs/symbols.md 를 읽는다.

    docs/symbols.md  (원본)
        ├─ scripts/build_docs.py  -> HTML hover 툴팁
        ├─ scripts/walkthrough.py -> 터미널 표 위의 범례
        └─ 사람이 직접 읽기
"""

from __future__ import annotations

import re
import textwrap
from functools import lru_cache
from pathlib import Path

from .utils import REPO_ROOT

SYMBOLS_MD = REPO_ROOT / "docs" / "symbols.md"

# | `기호` | 정의 |  형태의 표 행만 취한다. 헤더와 구분선은 자동으로 걸러진다.
_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*(.+?)\s*\|\s*$")


@lru_cache(maxsize=1)
def load_symbols(path: str | None = None) -> dict[str, str]:
    """symbols.md 의 표에서 기호 -> 정의 를 뽑는다."""
    p = Path(path) if path else SYMBOLS_MD
    syms: dict[str, str] = {}
    if not p.exists():
        return syms
    for line in p.read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line.strip())
        if not m:
            continue
        key = m.group(1).replace(r"\|", "|").strip()
        val = m.group(2).replace(r"\|", "|").strip()
        syms[key] = val
    return syms


def _strip_md(s: str) -> str:
    """터미널 출력용으로 마크다운 강조 표시를 벗긴다."""
    return s.replace("**", "").replace("`", "")


def legend(*keys: str, indent: str = "  ", width: int = 74) -> None:
    """표를 찍기 전에 그 표에 나오는 기호의 정의를 먼저 출력한다.

    문서를 안 보고 터미널만 보는 경우에도 기호가 미정의로 남지 않게 하는 장치.
    정의가 symbols.md 에 없으면 조용히 건너뛴다 (출력이 깨지지 않도록).
    """
    syms = load_symbols()
    rows = [(k, syms[k]) for k in keys if k in syms]
    if not rows:
        return

    pad = max(len(k) for k, _ in rows)
    print(f"{indent}[기호]")
    for k, v in rows:
        body = textwrap.wrap(_strip_md(v), width=width - pad - len(indent) - 6)
        first = body[0] if body else ""
        print(f"{indent}  {k:<{pad}}  {first}")
        for cont in body[1:]:
            print(f"{indent}  {'':<{pad}}  {cont}")
    print()
