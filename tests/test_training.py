from pathlib import Path
from tests import _PROJECT_ROOT


def test_models_dir_contains_pth_file():
    """Ensure there is at least one trained model (a .pth file) in the project `models/` folder at the cwd root."""
    models_dir = _PROJECT_ROOT / "models"

    # Directory exists
    assert models_dir.exists() and models_dir.is_dir(), f"Models directory not found at: {models_dir}"

    # Find all files with .pth suffix (case-insensitive)
    pth_files = [p for p in models_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pth"]

    assert pth_files, f"No .pth files found in models folder: {models_dir}"

    # Optional: check that at least one found file is non-empty
    assert any(p.stat().st_size > 0 for p in pth_files), f"All .pth files in {models_dir} are empty"
