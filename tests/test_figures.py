import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from importlib import import_module

figs = import_module("05_figures")


def test_generate_all_creates_files():
    generated = figs.generate_all()
    assert len(generated) >= 8
    for fp in generated:
        assert os.path.exists(fp), fp
        min_size = 1000 if fp.endswith(".png") else 100
        assert os.path.getsize(fp) > min_size, fp
