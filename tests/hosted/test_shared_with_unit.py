import importlib.util
import unittest
from pathlib import Path


SHARED_MODULES = (
    "test_citation_resolution.py",
)

_UNIT = Path(__file__).resolve().parents[1] / "unit"


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    for name in SHARED_MODULES:
        spec = importlib.util.spec_from_file_location(
            f"hosted_shared_{Path(name).stem}", _UNIT / name
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite.addTests(loader.loadTestsFromModule(module))
    return suite
