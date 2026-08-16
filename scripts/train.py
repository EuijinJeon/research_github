"""Stage 1 모델 학습.

깊이 축은 총 ReLU 유닛 수를 맞춰서 비교한다:
    [64]  vs [32, 32]   -> 둘 다 64비트 활성 패턴, 64개 초평면
    [128] vs [64, 64]   -> 둘 다 128비트

유닛 수가 다르면 "층이 깊어서"인지 "유닛이 많아서"인지 구분할 수 없다.
조합론적 상한을 고정한 채 기하(직선 경계 vs 꺾인 경계)만 바꾸는 것이 목적이다.

사용법:
    python scripts/train.py                  # 전체 그리드
    python scripts/train.py --only moons     # 특정 데이터셋만
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlpinterp.data import get_dataset  # noqa: E402
from mlpinterp.models import max_regions_one_hidden  # noqa: E402
from mlpinterp.train import ckpt_path, evaluate, save_model, train_model  # noqa: E402
from mlpinterp.utils import describe_device, ensure_dirs, get_device  # noqa: E402

# (dataset, hidden, epochs, batch_size)
GRID = [
    ("moons", [64], 150, 256),
    ("moons", [32, 32], 150, 256),
    ("spiral", [64], 400, 256),
    ("spiral", [32, 32], 400, 256),
    ("mnist", [128], 30, 256),
    ("mnist", [64, 64], 30, 256),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None, help="이 데이터셋만 학습")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="체크포인트가 있어도 재학습")
    args = ap.parse_args()

    ensure_dirs()
    device = get_device()
    print(f"device: {describe_device(device)}\n")

    for name, hidden, epochs, bs in GRID:
        if args.only and name != args.only:
            continue

        path = ckpt_path(name, hidden, args.seed)
        if path.exists() and not args.force:
            print(f"[skip] {path.name} — 이미 존재 (--force로 재학습)")
            continue

        ds = get_dataset(name, seed=args.seed)
        n_units = sum(hidden)
        print(f"[train] {name} hidden={hidden} ({n_units} units) — {ds}")
        if ds.input_dim <= 3:
            ub = max_regions_one_hidden(n_units, ds.input_dim)
            print(f"         1층 기준 영역 상한 (Zaslavsky): {ub:,}")

        model, history = train_model(
            ds, hidden, seed=args.seed, epochs=epochs, batch_size=bs, device=device
        )

        tr_acc, _ = evaluate(model, ds.x_train.to(device), ds.y_train.to(device))
        te_acc, _ = evaluate(model, ds.x_test.to(device), ds.y_test.to(device))

        # 저장 직전에 piecewise-linear 항등식을 검산한다.
        # 여기서 실패하면 이후 Stage 1 전체가 무의미하므로 즉시 중단한다.
        probe = ds.x_test[:512].to(device)
        v = model.verify_affine(probe)
        status = "PASS" if v["passed"] else "FAIL"
        print(
            f"         affine 검산: {status} "
            f"(max_abs={v['max_abs_err']:.2e}, max_rel={v['max_rel_err']:.2e})"
        )
        if not v["passed"]:
            raise SystemExit(
                "A_r·x + b_r != model(x). effective_affine 구현을 먼저 고쳐야 합니다."
            )

        p = save_model(model, ds, args.seed, history)
        print(f"         saved -> {p.relative_to(path.parents[2])}")
        print(f"         final: train {tr_acc:.4f} | test {te_acc:.4f}\n")


if __name__ == "__main__":
    main()
