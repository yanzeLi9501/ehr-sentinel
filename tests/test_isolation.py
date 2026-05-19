"""Repository isolation verification.

Asserts that no file under ``src/ehr_sentinel/`` imports from the original
analysis scripts and that no real-data identifiers are embedded in the
package source.
"""
from __future__ import annotations

import re
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src" / "ehr_sentinel"

FORBIDDEN_IMPORTS = ("NC_revision", "gpu_reproduction_test", "DeepseekRevision")


def _iter_py_files():
    return [p for p in SRC.rglob("*.py") if p.is_file()]


def test_no_forbidden_imports():
    for f in _iter_py_files():
        text = f.read_text(encoding="utf-8")
        for token in FORBIDDEN_IMPORTS:
            # Allow plain mentions in docstrings; forbid in import statements
            for line in text.splitlines():
                stripped = line.strip()
                if (stripped.startswith("import ") or stripped.startswith("from ")) and token in stripped:
                    raise AssertionError(f"Forbidden import {token!r} in {f}: {line}")


def test_no_real_data_paths_in_source():
    # Common patterns of real cohort data paths used in the original scripts
    forbidden_patterns = [
        re.compile(r"history_features\.csv"),
        re.compile(r"train_data\.csv"),
        re.compile(r"WHU_primary"),
        re.compile(r"42k_cohort"),
    ]
    for f in _iter_py_files():
        text = f.read_text(encoding="utf-8")
        for pat in forbidden_patterns:
            assert not pat.search(text), f"Forbidden data identifier {pat.pattern!r} found in {f}"


def test_synthetic_mrn_prefix_only():
    # The synthetic generator must only emit MRNs of the form SYN-######
    from ehr_sentinel import generate_admissions
    df = generate_admissions(n_patients=5, seed=0, n_admissions_per_patient=(2, 3))
    assert df["mrn"].str.match(r"^SYN-\d{6}$").all()
