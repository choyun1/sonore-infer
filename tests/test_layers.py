"""The differentiable core must not depend on sonore (design D2), and the
package's module-level imports must form no cycle (as sonore's own
tests/test_layers.py requires).

sonore is allowed only in the modules that make stimuli or plots. Every other
module is imported in a fresh interpreter, and the test fails if that pulls
sonore in.
"""

import ast
import pkgutil
import subprocess
import sys
from pathlib import Path

import sonore_infer

# Modules allowed to import sonore, by dotted name prefix.
SONORE_ALLOWED = ("sonore_infer.stimuli",)


def _core_modules() -> list[str]:
    names = [sonore_infer.__name__]
    for module in pkgutil.walk_packages(sonore_infer.__path__, prefix="sonore_infer."):
        if not module.name.startswith(SONORE_ALLOWED):
            names.append(module.name)
    return names


def test_core_does_not_import_sonore():
    for name in _core_modules():
        code = f"import sys, {name}; sys.exit('sonore' in sys.modules)"
        result = subprocess.run([sys.executable, "-c", code])
        assert result.returncode == 0, f"{name} imports sonore"


def _module_level_imports(path: Path, package: str) -> set[str]:
    """The sonore_infer modules ``path`` imports at module level (not inside functions)."""
    imported = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ImportFrom):
            if node.level:  # relative: resolve against the module's own package
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                target = f"{base}.{node.module}" if node.module else base
            else:
                target = node.module or ""
            if target.startswith("sonore_infer"):
                imported.add(target)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names if alias.name.startswith("sonore_infer"))
    return imported


def test_no_import_cycles():
    root = Path(sonore_infer.__file__).parent
    graph = {}
    for path in root.rglob("*.py"):
        parts = path.relative_to(root.parent).with_suffix("").parts
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        graph[name] = _module_level_imports(path, package)

    visiting, done = [], set()

    def visit(name):
        if name in done or name not in graph:
            return
        assert name not in visiting, "import cycle: " + " -> ".join(visiting[visiting.index(name) :] + [name])
        visiting.append(name)
        for target in graph[name]:
            visit(target)
        visiting.pop()
        done.add(name)

    for name in graph:
        visit(name)
