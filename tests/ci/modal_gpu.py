import pathlib
import subprocess

import modal


def _repo_root() -> pathlib.Path:
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return pathlib.Path.cwd()


REPO_ROOT = _repo_root()
REMOTE_REPO = "/root/rushlite"
IGNORE = [
    "**/.git",
    "**/build",
    "**/.venv",
    "**/__pycache__",
    "**/.pytest_cache",
    "**/.ruff_cache",
    "**/.mypy_cache",
    "**/*.pyc",
    "data",
    "dist",
]

app = modal.App("rushlite-ci-gpu")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .apt_install(
        "build-essential",
        "cmake",
        "git",
        "python3-venv",
        "python3-pip",
        "pybind11-dev",
        "python3-pybind11",
        "libboost-all-dev",
    )
    .pip_install("uv")
    .add_local_dir(str(REPO_ROOT), REMOTE_REPO, copy=True, ignore=IGNORE)
)


def _run(cmd: str) -> None:
    print(f"\n+ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, cwd=REMOTE_REPO, check=True)


@app.function(gpu="T4", image=image, timeout=60 * 60)
def run_gpu_ci() -> None:
    _run("nvidia-smi")
    _run("uv lock")
    _run("uv sync --extra cu128")
    _run(
        'CC=gcc CXX=g++ SKBUILD_CMAKE_DEFINE="LMP_ENABLE_CUDA=ON;'
        'CMAKE_C_COMPILER=gcc;CMAKE_CXX_COMPILER=g++" '
        "uv pip install ."
    )
    _run(
        'uv run python -c "'
        "import rushlite; "
        "rushlite.Variable([[0.0]], requires_grad=False, "
        "device=rushlite.device.cuda, dtype=rushlite.dtype.float64); "
        "print('rushlite CUDA device OK')\""
    )
    _run("uv run pytest tests -v")


@app.local_entrypoint()
def main() -> None:
    run_gpu_ci.remote()
