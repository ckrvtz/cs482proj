"""Check that the project environment is installed correctly (P0-T02).

Usage (from the repository root, with the environment active):

    python scripts/check_env.py                # imports + versions + GPU if present
    python scripts/check_env.py --require-gpu  # also fail unless the GPU works (local machine)

Exit code 0 means every check passed. Paste the output into the Decision log (docs/DECISIONS.md)
when recording the working environment.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import platform
import sys
import time

# Import name -> distribution label, for the packages the project relies on.
PACKAGES = {
    "desertification": "desertification",
    "sentinelhub": "sentinelhub",
    "pystac_client": "pystac-client",
    "planetary_computer": "planetary-computer",
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "scipy": "scipy",
    "xarray": "xarray",
    "rioxarray": "rioxarray",
    "rasterio": "rasterio",
    "geopandas": "geopandas",
    "shapely": "shapely",
    "pyproj": "pyproj",
    "statsmodels": "statsmodels",
    "pymannkendall": "pymannkendall",
    "sklearn": "scikit-learn",
    "imblearn": "imbalanced-learn",
    "lightgbm": "lightgbm",
    "shap": "shap",
    "optuna": "optuna",
    "torch": "torch",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "folium": "folium",
    "dotenv": "python-dotenv",
    "yaml": "PyYAML",
    "tqdm": "tqdm",
    "papermill": "papermill",
    "nbconvert": "nbconvert",
}


def check_imports() -> bool:
    ok = True
    for module, label in PACKAGES.items():
        try:
            mod = importlib.import_module(module)
            try:
                version = importlib.metadata.version(label)
            except importlib.metadata.PackageNotFoundError:
                version = getattr(mod, "__version__", "?")
            print(f"  ok    {label:<20} {version}")
        except Exception as exc:  # noqa: BLE001 - report any import failure
            ok = False
            print(f"  FAIL  {label:<20} {type(exc).__name__}: {exc}")
    return ok


def check_gpu(require_gpu: bool) -> bool:
    import torch

    from desertification.hardware import arch_list_supports, select_device

    print(f"  torch {torch.__version__}, built for CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("  CUDA not available: running on CPU.")
        return not require_gpu

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    archs = torch.cuda.get_arch_list()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU: {name}, compute capability {capability[0]}.{capability[1]}, {vram_gb:.1f} GB")
    print(f"  torch compiled for: {' '.join(archs)}")
    if not arch_list_supports(capability, archs):
        print("  FAIL  this torch build has no kernels for this GPU (see requirements-torch.txt)")
        return False

    device = select_device()
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(2048, 2048, generator=gen)
    b = torch.randn(2048, 2048, generator=gen)
    expected = a @ b
    start = time.perf_counter()
    result = (a.to(device) @ b.to(device)).cpu()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    max_err = (result - expected).abs().max().item()
    # float32 matmul on GPU vs CPU differs by accumulated rounding error only.
    passed = device == "cuda" and torch.allclose(result, expected, rtol=1e-3, atol=1e-2)
    status = "ok  " if passed else "FAIL"
    print(
        f"  {status}  2048x2048 matmul on {device}: {elapsed_ms:.1f} ms, "
        f"max |GPU-CPU| {max_err:.2e}"
    )
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--require-gpu", action="store_true", help="fail if the GPU is not usable")
    args = parser.parse_args()

    print(f"Python {sys.version.split()[0]} on {platform.platform()}")
    print("Imports:")
    imports_ok = check_imports()
    print("GPU:")
    gpu_ok = check_gpu(args.require_gpu) if imports_ok else False
    ok = imports_ok and gpu_ok
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
