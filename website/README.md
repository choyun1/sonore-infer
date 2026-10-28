# Project website

The website is built from the repository's own Markdown, so it never holds
a copy that can fall behind. `build.py` gathers the pages, rewrites their
links, adds the navigation and runs [MkDocs](https://www.mkdocs.org/):

| On the site | From |
|---|---|
| Overview (home page) | `docs/overview.md` |
| Methods | `docs/methods.md` |
| Results | `results/README.md` and every page in `results/`, with `results/plots/` and `results/data/` |
| Design | `docs/README.md` and every document in `docs/design/` |
| Code | the main `README.md` |

A new results page or design document appears on the site with no change
here: the navigation lists the folders' Markdown files, titled by their
first heading. Links to files that are not on the site (code, tools,
tests) go to the file on GitHub. Mathematics is written as `$...$` and
`$$...$$`, which GitHub renders too; on the site MathJax renders it.

## Build it

    pip install -r website/requirements.txt
    python website/build.py            # writes website/_build/site; fails on a broken link
    python website/build.py --serve    # the same, at http://localhost:8000

The workflow `.github/workflows/website.yml` builds the site on every pull
request and push to main, and keeps it as a download (the run's "website"
artifact).

## Publishing

The site is published at https://choyun1.github.io/sonore-infer/ when
both of these are set: in the repository's Settings,
Pages, set the source to "GitHub Actions"; then in Settings, Secrets and
variables, Actions, Variables, add `PUBLISH_WEBSITE` with the value `true`.
The next push to main deploys. Removing the variable stops further deploys;
unpublishing the site is done in the Pages settings.
