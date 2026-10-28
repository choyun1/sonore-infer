"""Build the project website from the repository's Markdown.

The site is made of the files already in the repository, not copies kept by
hand: ``docs/overview.md`` is the home page, then ``docs/methods.md``, the
results pages (``results/``), the design documents (``docs/``) and the main
README as the code page. This script gathers them into a staging folder at
the same relative paths, rewrites each relative link so it still points at
its target (on the site if the target is part of it, on GitHub otherwise),
writes the MkDocs configuration with a navigation built from the files
found, and runs ``mkdocs build``. A new results page or design document
therefore appears on the site with no change here.

    pip install -r website/requirements.txt
    python website/build.py            # builds website/_build/site
    python website/build.py --serve    # the same, served at localhost:8000
"""

from __future__ import annotations

import argparse
import posixpath
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HERE = REPO / "website"
STAGE = HERE / "_build" / "pages"
SITE = HERE / "_build" / "site"
GITHUB = "https://github.com/choyun1/sonore-infer"

# repository path -> path on the site; every other file of the site keeps its path
RENAMED = {"docs/overview.md": "index.md", "README.md": "code.md"}

# a Markdown link or image target: ](target) or ](target "title"), not a URL
LINK = re.compile(r"(\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))")
FENCE = re.compile(r"^\s*(```|~~~)")


def site_files() -> list[str]:
    """Repository paths (POSIX, relative) of every file the site holds."""
    paths = ["README.md"]
    paths += [p.relative_to(REPO).as_posix() for p in sorted((REPO / "docs").rglob("*")) if p.is_file()]
    paths += [p.relative_to(REPO).as_posix() for p in sorted((REPO / "results").rglob("*")) if p.is_file()]
    return paths


def site_path(repo_path: str) -> str:
    return RENAMED.get(repo_path, repo_path)


def rewrite_target(target: str, source: str, on_site: set[str]) -> str:
    """The link ``target``, written in repository file ``source``, as it must be written on the site."""
    if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith(("#", "/")):
        return target
    path, _, anchor = target.partition("#")
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source), path))
    suffix = f"#{anchor}" if anchor else ""
    if resolved in on_site:
        new = posixpath.relpath(site_path(resolved), posixpath.dirname(site_path(source)) or ".")
        return new + suffix
    if (REPO / resolved).is_dir():
        # a folder of the site links to its README (MkDocs makes it the folder's page)
        readme = posixpath.join(resolved, "README.md")
        if readme in on_site:
            return posixpath.relpath(site_path(readme), posixpath.dirname(site_path(source)) or ".") + suffix
        return f"{GITHUB}/tree/main/{resolved}{suffix}"
    return f"{GITHUB}/blob/main/{resolved}{suffix}"


def rewrite(text: str, source: str, on_site: set[str]) -> str:
    """``text`` with every relative link outside code blocks rewritten by :func:`rewrite_target`."""
    out, in_fence = [], False
    for line in text.splitlines(keepends=True):
        if FENCE.match(line):
            in_fence = not in_fence
        if not in_fence:
            line = LINK.sub(
                lambda m: m.group(1) + rewrite_target(m.group(2), source, on_site) + m.group(3), line
            )
        out.append(line)
    return "".join(out)


def title(path: Path) -> str:
    """The page's first level-1 heading, or its file name."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def nav(on_site: set[str]) -> list:
    """The navigation: home, methods, results, design, code."""

    def entries(folder: str) -> list[dict[str, str]]:
        pages = sorted(
            p
            for p in on_site
            if posixpath.dirname(p) == folder and p.endswith(".md") and not p.endswith("README.md")
        )
        return [{title(REPO / p): site_path(p)} for p in pages]

    return [
        {"Overview": "index.md"},
        {"Methods": "docs/methods.md"},
        {"Results": [{"All experiments": "results/README.md"}, *entries("results")]},
        {"Design": [{"All design documents": "docs/README.md"}, *entries("docs/design")]},
        {"Code": "code.md"},
    ]


def write_config(on_site: set[str]) -> Path:
    import yaml  # PyYAML comes with MkDocs

    config = yaml.safe_load((HERE / "mkdocs.yml").read_text(encoding="utf-8"))
    config |= {"docs_dir": str(STAGE), "site_dir": str(SITE), "nav": nav(on_site)}
    path = HERE / "_build" / "mkdocs.yml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def stage() -> set[str]:
    """Copy the site's files into ``STAGE``, links rewritten; return their repository paths."""
    paths = site_files()
    on_site = set(paths)
    if STAGE.exists():
        shutil.rmtree(STAGE)
    for path in paths:
        target = STAGE / site_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.endswith(".md"):
            text = (REPO / path).read_text(encoding="utf-8")
            target.write_text(rewrite(text, path, on_site), encoding="utf-8")
        else:
            shutil.copyfile(REPO / path, target)
    for extra in (HERE / "assets").iterdir():
        target = STAGE / "assets" / extra.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(extra, target)
    return on_site


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--serve", action="store_true", help="serve the site at localhost:8000 instead")
    args = parser.parse_args()
    config = write_config(stage())
    command = ["serve"] if args.serve else ["build", "--strict"]
    sys.exit(subprocess.call([sys.executable, "-m", "mkdocs", *command, "-f", str(config)]))


if __name__ == "__main__":
    main()
