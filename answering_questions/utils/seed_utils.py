import hashlib
import os
import random
from contextlib import contextmanager
from typing import Optional

_BASE_SEED: int = 0


def _normalize_seed(seed: int) -> int:
    return seed % (2**32)


def _apply_seed(seed: int) -> None:
    normalized = _normalize_seed(seed)
    random.seed(normalized)
    try:
        import numpy as np  # type: ignore

        np.random.seed(normalized)
    except ImportError:
        pass

    try:
        import torch  # type: ignore

        torch.manual_seed(normalized)
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            torch.cuda.manual_seed_all(normalized)  # type: ignore[attr-defined]
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except ImportError:
        pass


def seed_everything(seed: int) -> None:
    global _BASE_SEED
    _BASE_SEED = _normalize_seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(_BASE_SEED))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _apply_seed(_BASE_SEED)


def base_seed() -> int:
    return _BASE_SEED


def seed_from_material(material: str) -> int:
    digest = hashlib.sha256(f"{_BASE_SEED}:{material}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def reseed_for_context(material: str) -> int:
    seed = seed_from_material(material)
    _apply_seed(seed)
    return seed


@contextmanager
def temporary_seed(seed: Optional[int] = None):
    state_random = random.getstate()
    try:
        import numpy as np  # type: ignore

        state_numpy = np.random.get_state()
    except Exception:
        state_numpy = None
    try:
        import torch  # type: ignore

        state_cpu = torch.random.get_rng_state()
        state_cuda = (
            torch.cuda.get_rng_state_all()  # type: ignore[attr-defined]
            if torch.cuda.is_available()  # type: ignore[attr-defined]
            else None
        )
    except Exception:
        state_cpu = None
        state_cuda = None

    if seed is None:
        seed = _BASE_SEED
    _apply_seed(seed)
    try:
        yield
    finally:
        random.setstate(state_random)
        if state_numpy is not None:
            try:
                import numpy as np  # type: ignore

                np.random.set_state(state_numpy)
            except Exception:
                pass
        if state_cpu is not None:
            try:
                import torch  # type: ignore

                torch.random.set_rng_state(state_cpu)
                if (
                    state_cuda is not None
                    and torch.cuda.is_available()  # type: ignore[attr-defined]
                ):
                    torch.cuda.set_rng_state_all(state_cuda)  # type: ignore[attr-defined]
            except Exception:
                pass
