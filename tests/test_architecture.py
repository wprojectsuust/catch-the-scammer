import ast
import unittest
from pathlib import Path


class DependencyTests(unittest.TestCase):
    def test_inner_layers_only_depend_on_domain_and_standard_library(self):
        import sys
        root = Path(__file__).resolve().parents[1] / "antifraud"
        for name in ("domain.py", "application.py"):
            tree = ast.parse((root / name).read_text())
            for node in ast.walk(tree):
                imported = []
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imported = [node.module or ""]
                for module in imported:
                    with self.subTest(file=name, imported=module):
                        self.assertTrue(module.split(".")[0] in sys.stdlib_module_names
                                        or module == "antifraud.domain", module)
