"""Installation page findings.

- The model folder was hard-coded to ~/.synap/models; JS reads SYNAP_MODELS_DIR.
- The wheel had no py.typed, so type checkers ignored the SDK's hints.
"""
from pathlib import Path

import maximem_synap
from maximem_synap.cache import encoders


def test_synap_models_dir_moves_the_download_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("SYNAP_MODELS_DIR", str(tmp_path / "models"))
    assert encoders.models_dir() == tmp_path / "models"
    encoders.load_encoder("unknown:x")
    assert (tmp_path / "models").is_dir(), "the loader uses the folder at call time"


def test_without_it_the_default_is_home(monkeypatch):
    monkeypatch.delenv("SYNAP_MODELS_DIR", raising=False)
    assert encoders.models_dir() == Path.home() / ".synap" / "models"


def test_the_package_is_marked_typed():
    assert (Path(maximem_synap.__file__).parent / "py.typed").exists()
