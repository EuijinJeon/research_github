"""학습된 모델들에서 활성 패턴/영역을 뽑아 artifacts/regions/에 저장한다.

이게 Stage 1에서 가장 비싼 계산이고, 뒤따르는 클러스터링·ε-path는 전부
이 산출물 위에서 돈다. 한 번 뽑아두면 다시 뽑지 않는다 (요구사항 6).

사용법:
    python scripts/extract_regions.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlpinterp.data import get_dataset  # noqa: E402
from mlpinterp.models import max_regions_one_hidden  # noqa: E402
from mlpinterp.regions import extract_regions, region_path, save_regions  # noqa: E402
from mlpinterp.train import load_model  # noqa: E402
from mlpinterp.utils import describe_device, ensure_dirs, get_device  # noqa: E402

SETTINGS = [
    ("moons", [64]),
    ("moons", [32, 32]),
    ("spiral", [64]),
    ("spiral", [32, 32]),
    ("mnist", [128]),
    ("mnist", [64, 64]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    device = get_device()
    print(f"device: {describe_device(device)}\n")
    print(f"{'setting':22s} {'split':6s} {'inputs':>8s} {'regions':>9s} {'R/N':>7s} {'상한':>12s}")
    print("-" * 70)

    for name, hidden in SETTINGS:
        model, _ = load_model(name, hidden, args.seed, device=device)
        ds = get_dataset(name, seed=args.seed)

        for split, x in (("train", ds.x_train), ("test", ds.x_test)):
            path = region_path(name, hidden, args.seed, split)
            if path.exists() and not args.force:
                from mlpinterp.regions import load_regions

                rs = load_regions(name, hidden, args.seed, split)
            else:
                rs = extract_regions(model, x, name, args.seed, split)
                save_regions(rs)

            # 1층 모델에 한해 Zaslavsky 상한과 대조할 수 있다.
            if len(hidden) == 1 and ds.input_dim <= 3:
                ub = f"{max_regions_one_hidden(hidden[0], ds.input_dim):,}"
            else:
                ub = "-"
            tag = f"{name} h={'x'.join(map(str, hidden))}"
            print(
                f"{tag:22s} {split:6s} {rs.n_inputs:8,d} {rs.n_regions:9,d} "
                f"{rs.n_regions / rs.n_inputs:7.3f} {ub:>12s}"
            )
        print()


if __name__ == "__main__":
    main()
