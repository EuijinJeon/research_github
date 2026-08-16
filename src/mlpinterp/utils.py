"""재현성, device 선택, 산출물 경로.

설계 원칙 (Stage 0에서 정한 것):
- device-agnostic: cuda / mps / cpu 어디서든 같은 코드가 돈다.
- float64를 기본으로 쓰지 않는다 (MPS 미지원). 고정밀 검산이 필요하면 CPU로 승격.
- 중간 산출물은 전부 artifacts/ 아래. 뒷단계가 앞단계를 재실행하지 않게 한다.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------- 경로

# 이 파일: <repo>/src/mlpinterp/utils.py  →  parents[2] == <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"
CKPT_DIR = ARTIFACTS / "checkpoints"
REGION_DIR = ARTIFACTS / "regions"
FIG_DIR = ARTIFACTS / "figures"
LOG_DIR = ARTIFACTS / "logs"
DATA_DIR = REPO_ROOT / "data"


def ensure_dirs() -> None:
    for d in (CKPT_DIR, REGION_DIR, FIG_DIR, LOG_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 재현성


def set_seed(seed: int) -> None:
    """python / numpy / torch 난수를 한 번에 고정.

    cudnn deterministic까지 켠다. 우리 모델은 전부 작아서 속도 손해가 없고,
    "같은 seed면 같은 폴리토프"가 보장되지 않으면 Stage 1의 검산이 의미를 잃는다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------- device


def get_device(prefer: str | None = None) -> torch.device:
    """사용 가능한 최선의 device를 고른다.

    prefer로 강제할 수 있다 ("cpu"를 넘기면 고정밀 검산용으로 쓸 수 있음).
    환경변수 MLPINTERP_DEVICE로도 강제 가능 — 맥에서 MPS 이슈가 나면
    코드를 고치지 않고 CPU로 내릴 수 있게 하는 탈출구.
    """
    if prefer is None:
        prefer = os.environ.get("MLPINTERP_DEVICE")
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device.index or 0)})"
    return device.type


# ---------------------------------------------------------------- 그림


def setup_matplotlib(prefer_korean: bool = True) -> str | None:
    """한글이 깨지지 않는 폰트를 고른다. 없으면 조용히 기본값으로 둔다.

    리눅스(Noto Sans CJK KR)와 macOS(Apple SD Gothic Neo)를 모두 커버한다.
    폰트가 없다고 그림 생성이 실패하면 안 되므로 실패는 무시한다.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    matplotlib.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지
    if not prefer_korean:
        return None

    available = {f.name for f in font_manager.fontManager.ttflist}
    for cand in (
        "Noto Sans CJK KR",
        "NanumGothic",
        "Apple SD Gothic Neo",
        "AppleGothic",
        "Malgun Gothic",
    ):
        if cand in available:
            plt.rcParams["font.family"] = [cand]
            return cand
    return None


# ---------------------------------------------------------------- 저장/로드


def save_checkpoint(path: Path, payload: dict) -> Path:
    """체크포인트 저장. 텐서는 전부 CPU로 내려서 저장한다.

    GPU 텐서를 그대로 저장하면 로드 시 device가 강제되어
    4060 ↔ M4 Pro 사이에서 깨진다. 저장 시점에 CPU로 통일하는 게
    map_location에만 의존하는 것보다 확실하다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def to_cpu(obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(to_cpu(v) for v in obj)
        return obj

    torch.save(to_cpu(payload), path)
    return path


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> dict:
    """항상 map_location을 명시해서 로드한다."""
    return torch.load(path, map_location=device, weights_only=False)
