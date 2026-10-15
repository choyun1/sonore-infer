"""SVG plots for the experiment pages in results/, drawn from the committed outputs in results/data/.

Each plot is one self-contained SVG file in results/plots/, named after its
page, which shows it with a Markdown image link. Everything is read from
results/data/, so the plots can be redrawn from the repository alone:

    python tools/results_plots.py

Output is deterministic: rerunning it on unchanged data writes identical
files. Colors follow the reader's light or dark theme. Hovering a point in a
browser that opens the SVG on its own shows its value; GitHub shows the
image without that.

Paper values are read by eye from Fig. 7D of Cusimano, Hewitt & McDermott
(2024), as in the table "Milestone (a) at a glance" in results/README.md.
"""

import argparse
import glob
import html
import math
import os
import re
import sys

from sonore_infer.thresholds import probability_from_log_odds, threshold

TOOLS = os.path.dirname(os.path.abspath(__file__))
# the summaries this script draws from live with their experiments' scripts
sys.path[:0] = [os.path.join(TOOLS, folder) for folder in ("bass", "grammar", "online")]

import aba_map  # noqa: E402
import mistuning_thresholds  # noqa: E402
import two_notes_summary  # noqa: E402

RESULTS = os.path.join(TOOLS, "..", "results")
DATA = os.path.join(RESULTS, "data")

# Paper's model thresholds (% of f0), Fig. 7D read by eye; None is not measurable.
PAPER = {
    (100, 1): 45,
    (100, 2): None,
    (100, 3): None,
    (200, 1): 10,
    (200, 2): 20,
    (200, 3): 21,
    (400, 1): 5,
    (400, 2): 10,
    (400, 3): 13,
}
NOT_MEASURABLE = 49.9  # a threshold this high means two sources were never preferred

STYLE = """
:root { --background: #f6f7f5; --text: #1b1d1c; --muted: #5d625f; --rule: #dcdfdb;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #b07c00; --s5: #6b6f6c;
  --q1: #b9d3f2; --q2: #85b1e8; --q3: #4f8bdc; --q4: #2a69bd; --q5: #163f78; --diamond: #1b1d1c }
@media (prefers-color-scheme: dark) { :root { --background: #1f2221; --text: #e9ebe8; --muted: #a3a8a4;
  --rule: #333836; --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #9a9f9b;
  --q1: #24446e; --q2: #2f5d9a; --q3: #3f7ccc; --q4: #78a8e8; --q5: #b8d3f5; --diamond: #e9ebe8 } }
.background { fill: var(--background) }
text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--muted) }
.tick { font-size: 10px; font-variant-numeric: tabular-nums }
.label { font-size: 10.5px }
.title { font-size: 11px; font-weight: 600; fill: var(--text) }
.note { font-size: 9.5px }
.legend { font-size: 12px }
.grid { stroke: var(--rule); stroke-width: 1 }
.axis { stroke: var(--muted); stroke-width: 1 }
.zero { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 2 3 }
.line { fill: none; stroke-width: 2; stroke-linejoin: round }
.vertical { stroke-width: 1.5 }
.dash { stroke-dasharray: 5 4 }
.marker { stroke: var(--background); stroke-width: 1.5 }
.diamond { fill: none; stroke: var(--diamond); stroke-width: 2 }
.s1 { stroke: var(--s1) } circle.s1 { fill: var(--s1) }
.s2 { stroke: var(--s2) } circle.s2 { fill: var(--s2) }
.s3 { stroke: var(--s3) } circle.s3 { fill: var(--s3) }
.s4 { stroke: var(--s4) } circle.s4 { fill: var(--s4) }
.s5 { stroke: var(--s5) } circle.s5 { fill: var(--s5) }
circle.q1 { fill: var(--q1) } circle.q2 { fill: var(--q2) } circle.q3 { fill: var(--q3) }
circle.q4 { fill: var(--q4) } circle.q5 { fill: var(--q5) }
"""


def logistic(log_odds):
    return 1 / (1 + math.exp(-max(min(log_odds, 700), -700)))


def escape(value):
    return html.escape(str(value))


def line_element(kind, x1, x2, y1, y2):
    return f'<line class="{kind}" x1="{x1:.1f}" x2="{x2:.1f}" y1="{y1:.1f}" y2="{y2:.1f}"/>'


def text_element(kind, x, y, content, anchor="start", transform=""):
    transform = f' transform="{transform}"' if transform else ""
    position = f'x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"'
    return f'<text class="{kind}" {position}{transform}>{escape(content)}</text>'


def circle_element(kind, x, y, radius, tip=None):
    title = f"<title>{escape(tip)}</title>" if tip else ""
    return f'<circle class="{kind}" cx="{x:.1f}" cy="{y:.1f}" r="{radius}">{title}</circle>'


def diamond(x, y, size, tip):
    return (
        f'<path class="diamond" d="M{x:.1f} {y - size:.1f}L{x + size:.1f} {y:.1f}L{x:.1f} {y + size:.1f}'
        f'L{x - size:.1f} {y:.1f}Z"><title>{escape(tip)}</title></path>'
    )


class Panel:
    """One plot area: data coordinates mapped to an SVG box of width × height."""

    def __init__(self, width, height, x_range, y_range, left=38, right=10, top=22, bottom=30, x_scale=None):
        self.width, self.height = width, height
        self.x_range, self.y_range = x_range, y_range
        self.left, self.right, self.top, self.bottom = left, right, top, bottom
        self.x_scale = x_scale or (lambda value: value)
        self.items = []

    @property
    def plot_right(self):
        return self.width - self.right

    @property
    def plot_bottom(self):
        return self.height - self.bottom

    def x(self, value):
        start, stop = self.x_scale(self.x_range[0]), self.x_scale(self.x_range[1])
        return self.left + (self.x_scale(value) - start) / (stop - start) * (self.plot_right - self.left)

    def y(self, value):
        value = min(max(value, self.y_range[0]), self.y_range[1])
        span = self.y_range[1] - self.y_range[0]
        return self.top + (self.y_range[1] - value) / span * (self.plot_bottom - self.top)

    def axes(self, x_ticks, y_ticks, x_label="", y_label="", title="", x_format=None, y_format=None):
        x_format = x_format or (lambda value: f"{value:g}")
        y_format = y_format or (lambda value: f"{value:g}")
        parts = []
        for value in y_ticks:
            y = self.y(value)
            parts.append(line_element("grid", self.left, self.plot_right, y, y))
            parts.append(text_element("tick", self.left - 5, y + 3.5, y_format(value), "end"))
        base = self.plot_bottom
        parts.append(line_element("axis", self.left, self.plot_right, base, base))
        for value in x_ticks:
            x = self.x(value)
            parts.append(line_element("axis", x, x, base, base + 4))
            parts.append(text_element("tick", x, base + 15, x_format(value), "middle"))
        middle_x = (self.left + self.plot_right) / 2
        middle_y = (self.top + self.plot_bottom) / 2
        if x_label:
            parts.append(text_element("label", middle_x, self.height - 2, x_label, "middle"))
        if y_label:
            parts.append(
                text_element("label", 10, middle_y, y_label, "middle", f"rotate(-90 10 {middle_y:.1f})")
            )
        if title:
            parts.append(text_element("title", self.left, 14, title))
        self.items[:0] = parts

    def line(self, points, color, dash=False, markers=True, tip=None):
        points = [(x, y) for x, y in points if y is not None]
        if not points:
            return
        path = " ".join(
            f"{'M' if index == 0 else 'L'}{self.x(x):.1f},{self.y(y):.1f}"
            for index, (x, y) in enumerate(points)
        )
        self.items.append(f'<path class="line {color}{" dash" if dash else ""}" d="{path}"/>')
        if markers:
            for x, y in points:
                self.items.append(
                    circle_element(f"marker {color}", self.x(x), self.y(y), 3, tip(x, y) if tip else None)
                )

    def vertical(self, x, color, dash=False):
        if x is not None:
            kind = f"vertical {color}{' dash' if dash else ''}"
            self.items.append(line_element(kind, self.x(x), self.x(x), self.top, self.plot_bottom))

    def horizontal(self, y, kind="zero"):
        self.items.append(line_element(kind, self.left, self.plot_right, self.y(y), self.y(y)))

    def text(self, x, y, content, kind="note", anchor="start"):
        self.items.append(text_element(kind, x, y, content, anchor))

    def error_bars(self, points, color):
        """Vertical bars of mean ± sd at each (x, mean, sd)."""
        for x, mean, sd in points:
            self.items.append(
                line_element(f"line {color}", self.x(x), self.x(x), self.y(mean - sd), self.y(mean + sd))
            )


def legend_entries(entries, width):
    """SVG for a legend of (color, name, kind) entries, wrapped to width; returns (svg, height)."""
    parts, x, y, row_height = [], 0.0, 12.0, 18
    for color, name, kind in entries:
        entry_width = 40 + 7.0 * len(name)  # 12 px text averages under 7 px a character
        if x > 0 and x + entry_width > width:
            x, y = 0.0, y + row_height
        if kind == "line":
            parts.append(line_element(f"line {color}", x + 1, x + 21, y - 4, y - 4))
        elif kind == "dash":
            parts.append(line_element(f"vertical {color} dash", x + 1, x + 21, y - 4, y - 4))
        elif kind == "dot":
            parts.append(circle_element(f"marker {color}", x + 8, y - 4, 4))
        else:
            parts.append(diamond(x + 8, y - 4, 5, name))
        parts.append(text_element("legend", x + 26, y, name))
        x += entry_width
    return "".join(parts), y + 8


def figure(panels, label, legend=(), columns=None, gap=10, padding=14):
    """One SVG document: a legend above a grid of panels of equal size."""
    columns = columns or len(panels)
    rows = math.ceil(len(panels) / columns)
    panel_width, panel_height = panels[0].width, panels[0].height
    inner_width = columns * panel_width + (columns - 1) * gap
    legend_svg, legend_height = legend_entries(legend, inner_width) if legend else ("", 0)
    width = inner_width + 2 * padding
    height = padding + legend_height + rows * panel_height + (rows - 1) * gap + padding
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height:.0f}" width="{width}" '
        f'height="{height:.0f}" role="img" aria-label="{escape(label)}">',
        f"<style>{STYLE}</style>",
        f'<rect class="background" width="{width}" height="{height:.0f}" rx="8"/>',
        f'<g transform="translate({padding} {padding})">{legend_svg}</g>',
    ]
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        x = padding + column * (panel_width + gap)
        y = padding + legend_height + row * (panel_height + gap)
        parts.append(f'<g transform="translate({x} {y:.0f})">' + "".join(panel.items) + "</g>")
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def data(*parts):
    return os.path.join(DATA, *parts)


def numeric_rows(path):
    """Whitespace-split rows of a text output whose first field is a number."""
    with open(path, encoding="utf-8") as file:
        return [line.split() for line in file if line[:7].strip().isdigit()]


# Milestone (a): the mistuned harmonic


def load_milestone_a(experiment):
    if experiment == "A1":
        paths = glob.glob(data("A1", "coarse", "f0-*.txt")) + glob.glob(data("A1", "fine", "f0-*.txt"))
    else:
        paths = glob.glob(data(experiment, "f0-*.txt"))
    conditions = {}
    for path in sorted(paths):
        f0, harmonic, rows, _ = mistuning_thresholds.read(path)
        conditions.setdefault((int(f0), harmonic), {}).update(rows)
    return conditions


def laplace_threshold(rows):
    mistunings = sorted(rows)
    value = threshold(mistunings, probability_from_log_odds([rows[m][0] for m in mistunings]))
    return None if value >= NOT_MEASURABLE else value


def mistuning_grid(experiment):
    conditions = load_milestone_a(experiment)
    panels = []
    for f0 in (100, 200, 400):
        for harmonic in (1, 2, 3):
            rows = conditions[(f0, harmonic)]
            panel = Panel(210, 160, (0, 50), (0, 1), top=34)
            panel.axes(
                [0, 10, 20, 30, 40, 50],
                [0, 0.5, 1],
                title=f"{f0} Hz, harmonic {harmonic}",
                x_label="mistuning (% of f0)" if f0 == 400 else "",
                y_label="p(two)" if harmonic == 1 else "",
            )
            paper = PAPER[(f0, harmonic)]
            panel.vertical(paper, "s2", dash=True)
            ours = laplace_threshold(rows)
            panel.vertical(ours, "s3")
            panel.line(
                [(m, logistic(rows[m][0])) for m in sorted(rows)],
                "s1",
                tip=lambda x, y, rows=rows: f"{x:g}%: log odds {rows[x][0]:+.1f}, p(two) {y:.2f}",
            )
            ours_text = f"{ours:.1f}" if ours else "n.m."
            paper_text = f"~{paper}" if paper else "n.m."
            panel.text(panel.left, 27, f"ours {ours_text} · paper {paper_text}")
            panels.append(panel)
    legend = [
        ("s1", "p(two sources), Laplace", "line"),
        ("s3", "our threshold (Eqn 2)", "line"),
        ("s2", "paper's model (Fig. 7D, by eye)", "dash"),
    ]
    return figure(panels, f"{experiment}: p(two) against mistuning", legend, columns=3)


def threshold_ladder():
    conditions = [(f0, harmonic) for f0 in (100, 200, 400) for harmonic in (1, 2, 3)]
    experiments = ["A1", "A2", "A3", "A4", "A5"]
    loaded = {experiment: load_milestone_a(experiment) for experiment in experiments}
    row_height, top = 26, 30
    panel = Panel(
        640, top + row_height * len(conditions) + 40, (0, 52), (0, 1), left=118, right=20, top=top, bottom=40
    )
    parts = []
    for value in (0, 10, 20, 30, 40, 50):
        x = panel.x(value)
        parts.append(line_element("grid", x, x, top - 6, panel.plot_bottom))
        parts.append(text_element("tick", x, panel.plot_bottom + 15, value, "middle"))
    parts.append(text_element("tick", panel.x(50), top - 12, "n.m.", "middle"))
    middle_x = (panel.left + panel.plot_right) / 2
    parts.append(text_element("label", middle_x, panel.height - 6, "threshold (% of f0)", "middle"))
    for index, (f0, harmonic) in enumerate(conditions):
        y = top + row_height * index + row_height / 2
        parts.append(text_element("tick", panel.left - 10, y + 4, f"{f0} Hz, h{harmonic}", "end"))
        parts.append(line_element("grid", panel.left, panel.plot_right, y, y))
        paper = PAPER[(f0, harmonic)]
        tip = "Paper: not measurable" if paper is None else f"Paper: about {paper}"
        parts.append(diamond(panel.x(50 if paper is None else paper), y, 7, tip))
        for order, experiment in enumerate(experiments):
            value = laplace_threshold(loaded[experiment][(f0, harmonic)])
            text = "n.m." if value is None else f"{value:.1f}%"
            x = panel.x(50 if value is None else value)
            parts.append(circle_element(f"marker q{order + 1}", x, y, 4.5, f"{experiment}: {text}"))
    panel.items = parts
    legend = [
        ("q1", "A1 gammatone", "dot"),
        ("q2", "A2 FFT", "dot"),
        ("q3", "A3 + whistle timing", "dot"),
        ("q4", "A4 + BASS gain", "dot"),
        ("q5", "A5 full model", "dot"),
        ("", "paper (by eye)", "diamond"),
    ]
    return figure([panel], "Milestone (a) thresholds by experiment", legend)


def bass_own_code():
    with open(data("A6", "thresholds.txt"), encoding="utf-8") as file:
        blocks = file.read().split("\n\n")
    panels = []
    for block, condition in zip(blocks, [(100, 1), (200, 2)], strict=False):
        rows = [line.split() for line in block.splitlines() if line.strip()[:1].isdigit()]
        mistunings = [float(row[0]) for row in rows]
        bass = [float(row[4]) for row in rows]
        ours = [float(row[5]) for row in rows]
        panel = Panel(320, 190, (0, 50), (-40, 160))
        panel.axes(
            [0, 10, 20, 30, 40, 50],
            [-40, 0, 40, 80, 120, 160],
            x_label="mistuning (% of f0)",
            y_label="log odds, two vs one",
            title=f"{condition[0]} Hz, harmonic {condition[1]}",
        )
        panel.horizontal(0)
        panel.vertical(PAPER[condition], "s3", dash=True)
        panel.line(list(zip(mistunings, bass, strict=True)), "s1", tip=lambda x, y: f"BASS {x:g}%: {y:+.1f}")
        panel.line(list(zip(mistunings, ours, strict=True)), "s2", tip=lambda x, y: f"ours {x:g}%: {y:+.1f}")
        panels.append(panel)
    legend = [
        ("s1", "BASS's code (ELBO, one seed)", "line"),
        ("s2", "ours (A5, Laplace)", "line"),
        ("s3", "paper's threshold (by eye)", "dash"),
    ]
    return figure(panels, "A6: log odds, BASS's code and ours", legend)


def variational_evidence():
    estimates = [
        (7, "s1", "Laplace"),
        (8, "s2", "IS from Laplace"),
        (16, "s3", "ELBO"),
        (17, "s4", "IS from variational"),
    ]
    panels = []
    for path in sorted(glob.glob(data("A7", "f0-*.txt"))):
        f0, harmonic = (int(value) for value in re.findall(r"f0-(\d+)_h(\d)", path)[0])
        rows = numeric_rows(path)
        mistunings = [float(row[0]) for row in rows]
        panel = Panel(250, 160, (0, 50), (0, 1))
        panel.axes(
            [0, 10, 20, 30, 40, 50],
            [0, 0.5, 1],
            x_label="mistuning (% of f0)",
            y_label="p(two)",
            title=f"{f0} Hz, harmonic {harmonic}",
        )
        panel.vertical(PAPER[(f0, harmonic)], "s5", dash=True)
        for column, color, name in estimates:
            panel.line(
                [(m, logistic(float(row[column]))) for m, row in zip(mistunings, rows, strict=True)],
                color,
                tip=lambda x, y, name=name: f"{name} {x:g}%: p(two) {y:.2f}",
            )
        panels.append(panel)
    legend = [(color, name, "line") for _, color, name in estimates] + [("s5", "paper (by eye)", "dash")]
    return figure(panels, "A7: p(two) by four evidence estimates", legend, columns=2)


def ten_seeds_by_condition():
    with open(data("A8", "summary.txt"), encoding="utf-8") as file:
        text = file.read().split("== (b)")[0]
    seeds = {}
    for block in re.split(r"\n(?=f0-)", text):
        match = re.match(r"f0-(\d+)_h(\d) (\w+)", block)
        if match:
            points = re.findall(r"^\s+(\d+)%\s+(-?[\d.]+) ±\s+([\d.]+)", block, re.MULTILINE)
            seeds[(int(match[1]), int(match[2]), match[3])] = [
                tuple(float(v) for v in point) for point in points
            ]
    return seeds


def ten_seeds():
    seeds = ten_seeds_by_condition()
    panels = []
    for f0, harmonic in [(100, 1), (200, 2), (200, 3), (400, 3)]:
        panel = Panel(250, 170, (0, 30), (-30, 110))
        panel.axes(
            [0, 5, 10, 15, 20, 25, 30],
            [-20, 0, 20, 40, 60, 80, 100],
            x_label="mistuning (% of f0)",
            y_label="log odds, two vs one",
            title=f"{f0} Hz, harmonic {harmonic}",
        )
        panel.horizontal(0)
        points = seeds[(f0, harmonic, "variational")]
        sds = {x: sd for x, _, sd in points}
        panel.error_bars(points, "s1")
        panel.line(
            [(x, mean) for x, mean, _ in points],
            "s1",
            tip=lambda x, y, sds=sds: f"{x:g}%: {y:+.1f} ± {sds[x]:.1f}",
        )
        one_seed = numeric_rows(data("A7", f"f0-{f0}_h{harmonic}.txt"))
        panel.line(
            [(float(row[0]), float(row[17])) for row in one_seed if float(row[0]) <= 30],
            "s2",
            dash=True,
            tip=lambda x, y: f"A7 one seed {x:g}%: {y:+.1f}",
        )
        panels.append(panel)
    legend = [
        ("s1", "mean ± SD over 10 seeds (variational)", "line"),
        ("s2", "A7, one seed (container)", "line"),
    ]
    return figure(panels, "A8: log odds over ten seeds", legend, columns=2)


# Milestone (b): two simultaneous notes, and milestone (c)'s kernel experiment

INTERVALS = [
    ("tritone_different_spectra", "Tritone, different spectra", "s1"),
    ("tritone", "Tritone", "s2"),
    ("just_fifth", "Just fifth", "s3"),
    ("octave", "Octave", "s4"),
]
ASYNCHRONIES = [0, 10, 20, 40, 80, 120, 160, 240]


def summary_odds(experiment, estimate):
    """Log odds of two over one per interval from data/<experiment>/summary.txt: {interval: [(ms, odds)]}."""
    with open(data(experiment, "summary.txt"), encoding="utf-8") as file:
        section = file.read().split(f"== {estimate} ==")[1].split("==")[0]
    odds = {}
    for line in section.splitlines():
        match = re.match(r"(\S+) (\d+) ms\s+(-?[\d.]+)", line.strip())
        if match:
            odds.setdefault(match[1], []).append((float(match[2]), float(match[3])))
    return {interval: sorted(points) for interval, points in odds.items()}


def asynchrony_panel(series, title, y_range, y_ticks, asynchronies=ASYNCHRONIES, width=600, height=240):
    """Log odds against onset asynchrony, points equally spaced; series is [(points, color, name, sds)]."""
    position = {value: index for index, value in enumerate(asynchronies)}
    panel = Panel(width, height, (0, len(asynchronies) - 1), y_range, left=46)
    panel.axes(
        list(range(len(asynchronies))),
        y_ticks,
        x_label="onset asynchrony of the upper note (ms; equally spaced, not to scale)",
        y_label="log odds, two vs one",
        x_format=lambda index: f"{asynchronies[int(index)]:g}",
        title=title,
    )
    panel.horizontal(0)
    for points, color, name, sds in series:
        if sds:
            panel.error_bars([(position[a], mean, sd) for a, mean, sd in sds], color)
        panel.line(
            [(position[a], value) for a, value in points],
            color,
            tip=lambda x, y, name=name: f"{name}, {asynchronies[int(x)]:g} ms: {y:+.1f}",
        )
    return panel


def two_notes(experiment, estimate, asynchronies):
    odds = summary_odds(experiment, estimate)
    series = [(odds.get(key, []), color, name, None) for key, name, color in INTERVALS]
    panel = asynchrony_panel(series, "", (-40, 90), [-40, -20, 0, 20, 40, 60, 80], asynchronies, height=260)
    panel.text(panel.x(len(asynchronies) - 1) - 4, panel.y(0) - 5, "above: two sources", anchor="end")
    legend = [(color, name, "line") for _, name, color in INTERVALS]
    return figure([panel], f"{experiment}: log odds of two notes over one", legend)


def octave_odds(paths, estimate="variational"):
    fits, floored = two_notes_summary.read(sorted(paths))
    table = two_notes_summary.odds(fits, floored, estimate)
    return sorted(
        (float(two_notes_summary.STIMULUS.match(stimulus)["value"]), values["two"][0])
        for stimulus, values in table.items()
        if stimulus.startswith("octave") and "two" in values
    )


def octave_sigma5():
    sd10 = octave_odds(glob.glob(data("B2", "octave*.txt")))
    sd5 = octave_odds(glob.glob(data("B3", "octave_*.txt")))
    sd5_elbo = octave_odds(glob.glob(data("B3", "octave_*.txt")), "elbo")
    series = [
        (sd10, "s1", "SD 10 dB (B2)", None),
        (sd5, "s2", "SD 5 dB", None),
        (sd5_elbo, "s3", "SD 5 dB, ELBO", None),
    ]
    panel = asynchrony_panel(series, "Octave", (-90, 10), [-80, -60, -40, -20, 0])
    legend = [
        ("s1", "likelihood SD 10 dB (B2), variational", "line"),
        ("s2", "SD 5 dB, variational", "line"),
        ("s3", "SD 5 dB, ELBO", "line"),
    ]
    return figure([panel], "B3: octave log odds by likelihood SD", legend)


def two_notes_seeds():
    with open(data("A8", "summary.txt"), encoding="utf-8") as file:
        text = file.read().split("== (b)")[1]
    seeds = {}
    pattern = r"^(\w+) (\d+) ms\s+(\w+)\s+(\w+)\s+(-?[\d.]+) ±\s+([\d.]+)"
    for name, asynchrony, estimate, hypothesis, mean, sd in re.findall(pattern, text, re.MULTILINE):
        seeds.setdefault((name, estimate, hypothesis), []).append((float(asynchrony), float(mean), float(sd)))
    rows = INTERVALS + [("just_fifth", "Just fifth, whistles", "s5")]
    series = []
    for key, name, color in rows:
        hypothesis = "whistles" if "whistles" in name else "two"
        points = sorted(seeds[(key, "variational", hypothesis)])
        series.append(([(a, mean) for a, mean, _ in points], color, name, points))
    panel = asynchrony_panel(series, "", (-40, 80), [-40, -20, 0, 20, 40, 60, 80], height=270)
    legend = [(color, name, "line") for _, name, color in rows]
    return figure([panel], "B4: log odds over ten seeds", legend)


def kernel_inferred():
    inferred = numeric_rows(data("C1", "f0-100_h1.txt"))
    fixed = numeric_rows(data("A7", "f0-100_h1.txt"))
    mistuned = Panel(380, 230, (0, 50), (-30, 30), left=46)
    mistuned.axes(
        [0, 10, 20, 30, 40, 50],
        [-30, -15, 0, 15, 30],
        x_label="mistuning (% of f0)",
        y_label="log odds, two vs one",
        title="100 Hz, harmonic 1",
    )
    mistuned.horizontal(0)
    mistuned.vertical(PAPER[(100, 1)], "s5", dash=True)
    mistuned.line(
        [(float(row[0]), float(row[17])) for row in fixed], "s1", tip=lambda x, y: f"fixed {x:g}%: {y:+.1f}"
    )
    mistuned.line(
        [(float(row[0]), float(row[-1])) for row in inferred],
        "s2",
        tip=lambda x, y: f"inferred {x:g}%: {y:+.1f}",
    )
    octave = asynchrony_panel(
        [
            (octave_odds(glob.glob(data("B2", "octave*.txt"))), "s1", "kernel fixed (B2)", None),
            (octave_odds(glob.glob(data("C1", "octave_*.txt"))), "s2", "kernel inferred", None),
        ],
        "Octave",
        (-50, 5),
        [-40, -20, 0],
        width=380,
        height=230,
    )
    legend = [
        ("s1", "kernel fixed (A7, B2)", "line"),
        ("s2", "kernel inferred", "line"),
        ("s5", "paper's 100/1 threshold (by eye)", "dash"),
    ]
    return figure([mistuned, octave], "C1: log odds with the GP kernel fixed and inferred", legend)


ABA_SEMITONES = (1, 3, 6, 9, 12)
ABA_INTERVALS = (67, 83, 100, 117, 134, 150)
ABA_CLIP = 20  # the color scale of the paper's Fig. 8D stops at ±20


def aba_heatmap(found):
    """The map as the paper draws it: semitones by interval, color clipped at ±20, values printed."""
    panel = Panel(600, 260, (0, len(ABA_INTERVALS)), (0, len(ABA_SEMITONES)), left=46, bottom=34)
    panel.items.append(
        "<style>.cell-two { fill: var(--s2) } .cell-one { fill: var(--s1) } "
        ".cell-value { font-size: 10.5px; fill: var(--text); font-variant-numeric: tabular-nums }</style>"
    )
    for row, semitones in enumerate(ABA_SEMITONES):
        for column, interval in enumerate(ABA_INTERVALS):
            odds = aba_map.log_odds(found, semitones, interval, "vi_importance")
            x0, x1 = panel.x(column), panel.x(column + 1)
            y0, y1 = panel.y(row + 1), panel.y(row)
            kind = "cell-two" if odds > 0 else "cell-one"
            opacity = min(abs(odds), ABA_CLIP) / ABA_CLIP
            panel.items.append(
                f'<rect class="{kind}" x="{x0 + 1:.1f}" y="{y0 + 1:.1f}" width="{x1 - x0 - 2:.1f}" '
                f'height="{y1 - y0 - 2:.1f}" fill-opacity="{opacity:.2f}">'
                f"<title>{semitones} st, {interval} ms: {odds:+.1f}</title></rect>"
            )
            panel.text((x0 + x1) / 2, (y0 + y1) / 2 + 3.5, f"{odds:+.0f}", "cell-value", "middle")
        middle = (panel.y(row) + panel.y(row + 1)) / 2
        panel.text(panel.left - 6, middle + 3.5, f"{semitones}", "tick", "end")
    for column, interval in enumerate(ABA_INTERVALS):
        panel.text(
            (panel.x(column) + panel.x(column + 1)) / 2,
            panel.plot_bottom + 14,
            f"{interval}",
            "tick",
            "middle",
        )
    panel.text(
        (panel.left + panel.plot_right) / 2,
        panel.height - 4,
        "onset-to-onset interval (ms)",
        "label",
        "middle",
    )
    middle_y = (panel.top + panel.plot_bottom) / 2
    panel.items.append(
        text_element(
            "label", 10, middle_y, "semitones (B above A)", "middle", f"rotate(-90 10 {middle_y:.1f})"
        )
    )
    panel.text(
        panel.left,
        14,
        "Scale stops at ±20 as in the paper's Fig. 8D: orange two streams, blue one",
        "title",
    )
    return panel


def aba_streaming():
    found = aba_map.collect([data("C2", "one_converged"), data("C2", "run_300steps")])
    panel = Panel(600, 260, (60, 157), (-50, 230), left=46)
    panel.axes(
        list(ABA_INTERVALS),
        [-50, 0, 50, 100, 150, 200],
        x_label="onset-to-onset interval (ms)",
        y_label="log odds, two streams vs one",
        title="The same values on their full scale (legend above)",
    )
    panel.horizontal(0)
    legend = []
    for semitones, color in zip(ABA_SEMITONES, ("s1", "s2", "s3", "s4", "s5"), strict=True):
        points = [(dt, aba_map.log_odds(found, semitones, dt, "vi_importance")) for dt in ABA_INTERVALS]
        panel.line(points, color, tip=lambda x, y, st=semitones: f"{st} st, {x:g} ms: {y:+.1f}")
        legend.append((color, f"{semitones} semitone{'s' if semitones > 1 else ''}", "line"))
    return figure(
        [aba_heatmap(found), panel],
        "C2: ABA, log odds of two streams by interval and frequency gap",
        legend,
        columns=1,
    )


# Source and room


def likelihood_slice():
    with open(data("R1", "likelihood_slices.txt"), encoding="utf-8") as file:
        lines = file.read().splitlines()
    header = next(index for index, line in enumerate(lines) if line.startswith("RT60"))
    rt60s = [float(value) for value in lines[header].split()[2:]]
    panel = Panel(600, 260, (0.1, 2.0), (-60, 5), left=46, x_scale=math.log)
    panel.axes(
        [0.1, 0.2, 0.4, 0.8, 2.0],
        [-60, -40, -20, 0],
        x_label="RT60 of the model (s, log scale); truth 0.4 s",
        y_label="log likelihood − value at truth (nats)",
    )
    panel.vertical(0.4, "s5", dash=True)
    colors = {"60": "s1", "40": "s2", "20": "s3"}
    for line in lines[header + 1 : header + 7]:
        fields = line.split()
        sigma, floor = fields[1], fields[3]
        if sigma != "3.6":
            continue
        values = [float(value) for value in re.findall(r"-?\d+\.\d", line.split("floor")[1][3:])]
        by_rt60 = dict(zip(rt60s, values, strict=True))
        panel.line(
            list(by_rt60.items()),
            colors[floor],
            tip=lambda x, y, floor=floor, by_rt60=by_rt60: (
                f"floor {floor} dB, RT60 {x:g} s: {by_rt60[x]:+.1f}"
            ),
        )
    legend = [
        ("s1", "floor 60 dB", "line"),
        ("s2", "floor 40 dB", "line"),
        ("s3", "floor 20 dB (BASS's)", "line"),
        ("s5", "true RT60", "dash"),
    ]
    return figure([panel], "R1: likelihood against the model's RT60", legend)


def interval_coverage():
    with open(data("R1", "recovery_summary.txt"), encoding="utf-8") as file:
        lines = file.read().splitlines()
    settings = ["fine-slow-small", "default", "coarse-fast-large"]
    cells = []
    for line in lines:
        match = re.match(r"(\S+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(.*)$", line)
        if match and match[1] in settings:
            coverage = [float(value) for value in re.findall(r"(\d+)% x", match[5])]
            cells.append((match[1], float(match[2]), match[3], coverage))
    quantities = ["RT60", "band correlation", "time correlation", "level sd"]
    columns = sorted({(cell[0], cell[1]) for cell in cells}, key=lambda key: (settings.index(key[0]), key[1]))
    panels = []
    for k, quantity in enumerate(quantities):
        panel = Panel(300, 190, (-0.5, len(columns) - 0.5), (0, 100), bottom=44)
        panel.axes(
            list(range(len(columns))),
            [0, 50, 90, 100],
            y_label="% of runs in 90% interval",
            title=quantity,
            x_format=lambda index: f"{columns[int(index)][1]:g}",
        )
        panel.horizontal(90)
        for sigma, color, offset in (("3.6", "s1", -0.12), ("7.2", "s2", 0.12)):
            for index, column in enumerate(columns):
                cell = next(cell for cell in cells if (cell[0], cell[1]) == column and cell[2] == sigma)
                tip = f"{column[0]}, RT60 {column[1]:g} s, sigma {sigma}: {cell[3][k]:g}%"
                panel.items.append(
                    circle_element(f"marker {color}", panel.x(index + offset), panel.y(cell[3][k]), 3.5, tip)
                )
        for group, name in enumerate(["fine-slow", "default", "coarse-fast"]):
            panel.text(panel.x(3 * group + 1), panel.height - 18, name, anchor="middle")
        panels.append(panel)
    legend = [("s1", "sigma 3.6 dB (measured)", "dot"), ("s2", "sigma 7.2 dB (twice)", "dot")]
    return figure(panels, "R1: coverage of 90% intervals", legend, columns=2)


def atypical_rooms():
    with open(data("R2", "atypical_rooms_summary.txt"), encoding="utf-8") as file:
        lines = file.read().splitlines()
    header = next(index for index, line in enumerate(lines) if line.startswith("condition"))
    rows = [line.split() for line in lines[header + 1 :] if line.strip()]
    rooms = [
        "control",
        "exaggerated",
        "reduced",
        "linear_matched_end",
        "linear_matched_start",
        "inverted",
        "time_reversed",
    ]
    names = ["normal", "exaggerated", "reduced", "linear, end", "linear, start", "inverted", "time-reversed"]
    colors = {"60": "s1", "40": "s2", "20": "s3"}
    panels = []
    for rt60 in ("0.4", "0.8"):
        top, row_height = 28, 26
        panel = Panel(
            330,
            top + row_height * len(rooms) + 40,
            (0.1, 4),
            (0, 1),
            left=96,
            right=12,
            top=top,
            bottom=40,
            x_scale=math.log,
        )
        parts = []
        for value in (0.1, 0.2, 0.5, 1, 2, 4):
            x = panel.x(value)
            kind = "zero" if value == 1 else "grid"
            parts.append(line_element(kind, x, x, top - 6, panel.plot_bottom))
            parts.append(text_element("tick", x, panel.plot_bottom + 15, f"{value:g}", "middle"))
        middle_x = (panel.left + panel.plot_right) / 2
        parts.append(
            text_element("label", middle_x, panel.height - 6, "RT60 read / true (log scale)", "middle")
        )
        parts.append(text_element("title", panel.left, 14, f"true RT60 {rt60} s"))
        for index, (room, name) in enumerate(zip(rooms, names, strict=True)):
            y = top + row_height * index + row_height / 2
            parts.append(text_element("tick", panel.left - 8, y + 4, name, "end"))
            matching = [row for row in rows if row[0] == room and row[1] == rt60]
            for row in matching:
                floor, posterior = row[2], float(row[4])
                tip = f"{name}, floor {floor} dB: {posterior:.2f}"
                parts.append(circle_element(f"marker {colors[floor]}", panel.x(posterior), y, 4, tip))
            t20 = float(matching[0][5])
            parts.append(diamond(panel.x(t20), y, 6, f"T20 x 3: {t20:.2f}"))
        panel.items = parts
        panels.append(panel)
    legend = [
        ("s1", "model, floor 60 dB", "dot"),
        ("s2", "floor 40 dB", "dot"),
        ("s3", "floor 20 dB", "dot"),
        ("", "T20 × 3 of the same rooms", "diamond"),
    ]
    return figure(panels, "R2: RT60 read in atypical rooms", legend)


# Online inference


TRIPLETS = [1, 2, 3, 4, 5, 6]


def online_odds(path):
    """Online log odds at each triplet's end, from the ``odds:`` lines of online_buildup.py: [(n, odds)]."""
    with open(path, encoding="utf-8") as file:
        rows = [line.split() for line in file if re.match(r"odds: [0-9.inf]+ ", line)]
    return [(int(row[2]), float(row[3])) for row in rows]


def batch_evidence(path):
    """Batch Laplace log evidence plus structure prior from the ``batch:`` lines: {(n, hypothesis): value}."""
    with open(path, encoding="utf-8") as file:
        rows = [line.split() for line in file if re.match(r"batch: [0-9]", line)]
    return {(int(row[1]), row[2]): float(row[3]) + float(row[4]) for row in rows}


def batch_odds(path):
    evidence = batch_evidence(path)
    return [(n, evidence[n, "two"] - evidence[n, "one"]) for n in TRIPLETS if (n, "two") in evidence]


def ten_thousand_steps_odds(semitones):
    """Batch log odds at 6 triplets with the one-stream fit run to 10,000 steps."""
    with open(data("O1", f"one_stream_10000_steps_st{semitones}.txt"), encoding="utf-8") as file:
        one = float(re.search(r"laplace\+structure (\S+)", file.read()).group(1))
    return batch_evidence(data("O1", f"st{semitones}_batch3000.txt"))[6, "two"] - one


def leaky_odds(path, tau, lead=0.05):
    """Leaky log odds (decision O10) at each triplet's end, structure prior included."""
    from leaky_odds import read_windows

    from sonore_infer.online import leaky_log_odds

    runs = read_windows(path)
    (lag,) = {lag for lag, _ in runs}
    one = {end: (term, structure) for end, term, structure in runs[lag, "one"]}
    two = {end: (term, structure) for end, term, structure in runs[lag, "two"]}
    ends = sorted(set(one) & set(two))
    curve = leaky_log_odds(ends, [two[end][0] for end in ends], [one[end][0] for end in ends], tau)
    points = []
    for n in TRIPLETS:
        end = round(lead + 0.5 * n, 9)
        if end in ends:
            k = ends.index(end)
            points.append((n, curve[k] + two[end][1] - one[end][1]))
    return points


def buildup_panel(title, y_range, y_ticks, width=340, height=230):
    panel = Panel(width, height, (0.7, 6.3), y_range, left=44)
    panel.axes(TRIPLETS, y_ticks, x_label="triplets heard", y_label="log odds, two vs one", title=title)
    panel.horizontal(0)
    return panel


def online_buildup():
    panels = []
    axes = ((4, (-5, 20), [-5, 0, 5, 10, 15, 20]), (8, (0, 140), [0, 35, 70, 105, 140]))
    for semitones, y_range, y_ticks in axes:
        panel = buildup_panel(f"{semitones} semitones", y_range, y_ticks)
        series = [
            (batch_odds(data("O1", f"st{semitones}_batch3000.txt")), "s1", "batch"),
            (online_odds(data("O1", f"st{semitones}_w50_p50_lag1.txt")), "s2", "online"),
            (leaky_odds(data("O1", f"st{semitones}_w50_p50_lag1.txt"), 2.5), "s3", "leaky"),
        ]
        for points, color, name in series:
            panel.line(points, color, tip=lambda x, y, name=name: f"{name}, {x:g} triplets: {y:+.1f}")
        odds = ten_thousand_steps_odds(semitones)
        panel.items.append(diamond(panel.x(6), panel.y(odds), 5, f"batch, 10,000 steps: {odds:+.1f}"))
        panels.append(panel)
    legend = [
        ("s1", "batch, 3000 steps", "line"),
        ("s5", "batch, 10,000 steps (one stream)", "diamond"),
        ("s2", "online, lag 1 s", "line"),
        ("s3", "online, leaky τ 2.5 s", "line"),
    ]
    return figure(panels, "O1: buildup of two streams, batch, online and leaky", legend)


def online_sweep():
    panels = []
    axes = ((4, (-5, 40), [0, 10, 20, 30, 40]), (8, (0, 200), [0, 50, 100, 150, 200]))
    for semitones, y_range, y_ticks in axes:
        panel = buildup_panel(f"{semitones} semitones", y_range, y_ticks)
        series = [
            (f"st{semitones}_w50_p50_lag0.5.txt", "s4", False, "lag 0.5 s"),
            (f"st{semitones}_w50_p50_lag1.txt", "s2", False, "lag 1 s"),
            (f"st{semitones}_w50_p50_lag2.txt", "s5", False, "lag 2 s"),
        ]
        if semitones == 8:
            series += [
                ("st8_w10_p50_lag1.txt", "s3", True, "window 10 ms"),
                ("st8_w100_p50_lag1.txt", "s1", True, "window 100 ms"),
            ]
        for name, color, dash, label in series:
            panel.line(
                online_odds(data("O1", name)),
                color,
                dash=dash,
                tip=lambda x, y, label=label: f"{label}, {x:g} triplets: {y:+.1f}",
            )
        panels.append(panel)
    legend = [
        ("s4", "lag 0.5 s", "line"),
        ("s2", "lag 1 s", "line"),
        ("s5", "lag 2 s", "line"),
        ("s3", "window 10 ms (8 st)", "dash"),
        ("s1", "window 100 ms (8 st)", "dash"),
    ]
    return figure(panels, "O1: online log odds by lag and window, windows 50 ms unless named", legend)


PLOTS = {
    "milestone-a-thresholds": threshold_ladder,
    "A1-gammatone-baseline": lambda: mistuning_grid("A1"),
    "A2-fft-cochleagram": lambda: mistuning_grid("A2"),
    "A3-whistle-timing": lambda: mistuning_grid("A3"),
    "A4-bass-gain": lambda: mistuning_grid("A4"),
    "A5-full-model": lambda: mistuning_grid("A5"),
    "A6-bass-own-code": bass_own_code,
    "A7-variational-evidence": variational_evidence,
    "A8-ten-seeds": ten_seeds,
    "B1-two-notes-laplace": lambda: two_notes("B1", "laplace", [0, 10, 20, 40, 80]),
    "B2-two-notes-variational": lambda: two_notes("B2", "variational", ASYNCHRONIES),
    "B3-octave-sigma5": octave_sigma5,
    "B4-two-notes-seeds": two_notes_seeds,
    "C1-kernel-inferred": kernel_inferred,
    "C2-aba-streaming": aba_streaming,
    "O1-online-buildup": online_buildup,
    "O1-lag-window": online_sweep,
    "R1-likelihood-slice": likelihood_slice,
    "R1-interval-coverage": interval_coverage,
    "R2-atypical-rooms": atypical_rooms,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(RESULTS, "plots"), help="folder for the SVG files")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, plot in PLOTS.items():
        path = os.path.join(args.out, f"{name}.svg")
        with open(path, "w", encoding="utf-8", newline="\n") as file:
            file.write(plot())
        print(path)


if __name__ == "__main__":
    main()
