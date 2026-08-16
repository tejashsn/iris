"""Environment metadata capture with graceful degradation."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from typing import Any


def collect_metadata() -> dict[str, Any]:
    """Collect optional runtime metadata; missing GPU/tooling is not an error."""
    meta: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": None,
        "gpu_arch": None,
        "rocm_probe": None,
    }

    try:
        import torch

        meta["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            meta["gpu_arch"] = getattr(props, "gcnArchName", None) or props.name
    except Exception:
        pass

    meta["rocm_probe"] = _probe_rocm()
    return meta


def _probe_rocm() -> dict[str, Any] | None:
    rocm_smi = shutil.which("rocm-smi")
    if not rocm_smi:
        return None

    try:
        completed = subprocess.run(
            [rocm_smi, "--showproductname"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if completed.returncode != 0:
            return {"available": False, "error": completed.stderr.strip() or "non-zero exit"}
        return {"available": True, "product_name": completed.stdout.strip()}
    except Exception as exc:
        return {"available": False, "error": str(exc)}
