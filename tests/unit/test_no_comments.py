import ast
import io
import tokenize
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDERS = ("src", "profiles", "scripts", "plugins", "skills", "tests")
OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _reads_doc(tree):
    return any(
        (isinstance(node, ast.Name) and node.id == "__doc__")
        or (isinstance(node, ast.Attribute) and node.attr == "__doc__")
        for node in ast.walk(tree)
    )


class NoCommentsTests(unittest.TestCase):
    def test_the_python_files_carry_no_comments_or_docstrings(self):
        found = []
        for folder in FOLDERS:
            for path in sorted((ROOT / folder).rglob("*.py")):
                name = path.relative_to(ROOT).as_posix()
                text = path.read_text(encoding="utf-8")
                for token in tokenize.generate_tokens(io.StringIO(text).readline):
                    shebang = token.start[0] == 1 and token.string.startswith("#!")
                    if token.type == tokenize.COMMENT and not shebang:
                        found.append(f"{name}:{token.start[0]} comment")
                tree = ast.parse(text)
                if _reads_doc(tree):
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, OWNERS) and ast.get_docstring(node, clean=False) is not None:
                        found.append(f"{name}:{getattr(node, 'lineno', 1)} docstring")
        self.assertEqual([], found)


if __name__ == "__main__":
    unittest.main()
