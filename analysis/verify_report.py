"""Check every number in the technical report (report/report_en.tex) against the repository file it comes from.

The report adds no new numbers: each number it prints is taken from a committed file (README.md, the archived
Markdown report, media/README.md, the result JSON files, the pinned code constants). This script extracts every number
from the body of report/report_en.tex and asserts that it appears, as a whole number token, in the file that SOURCES maps
it to. A number that is in the report but not in SOURCES fails ("unmapped"); a mapping that is no longer used fails too
("stale"), so the map stays in step with the text. Modelled on the garment repository's report checker.

Skipped as not claims: the preamble and the bibliography, LaTeX labels / references / citations, file paths and code
names (\\code, \\cmd, \\texttt, \\url), figure file names and widths, table column specifications, and digits glued to letters
(names such as G1, N1.6 in "GR00T N1.6", 103-D is checked as 103). The TikZ figure is layout; it must print no numbers.

Further checks: the report states that no success rate is reported, that the Eagle embedding does not influence the
target, and that this is not a vision-language-action model.

    python -m analysis.verify_report [--md out.md]       # standard library only; exit 1 on any failure
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = "report/report_en.tex"
README = "README.md"
ARCHIVED = "report/archive/report.md"
MEDIA = "media/README.md"
SWEEP = "results/walk_sweep_cpu/walk_sweep.json"
RERUN = "results/demo_rerun_cpu/episodes.json"
NAVIGATOR = "src/g1nav/navigation/action_module.py"
NAVIGATE = "src/g1nav/navigation/navigate.py"
WALK_EVAL = "src/g1nav/locomotion/walk_eval.py"
SCENE_GEN = "src/g1nav/envs/scene_gen.py"
INSTRUCTIONS = "src/g1nav/envs/instructions.py"
SCENE_DOC = "docs/dataset.md"

# number as printed in the report -> the repository file it was taken from
SOURCES = {
    # dates, counts, interfaces
    "2026": README, "0": README, "1": README, "2": README, "3": README, "4": README, "5": README, "10": README,
    "29": README, "50": README, "500": README, "103": README, "216": README, "0.5": README, "6": SCENE_GEN,
    "512": ARCHIVED, "256": ARCHIVED, "128": ARCHIVED, "58": ARCHIVED,
    # training
    "154.8": README, "154828800": README, "77": README, "8192": README, "100": README, "1.0": README, "16": README,
    "10321920": ARCHIVED, "2.3": ARCHIVED, "2.0": ARCHIVED,
    # raw-environment evaluation
    "0.6": README, "250": WALK_EVAL, "0.3": WALK_EVAL, "0.4": WALK_EVAL, "41.3": README, "0.52": README,
    "0.74": README, "0.58": README, "144.5": README, "87": README, "2.59": README, "0.519": README, "2.07": README,
    "0.745": README,
    # emergence table (archived report, section 4; values from walk_t4_metrics.jsonl and walk_sweep.json)
    "6.30": ARCHIVED, "0.36": ARCHIVED, "0.78": ARCHIVED, "10.3": ARCHIVED, "2.83": ARCHIVED, "0.28": ARCHIVED,
    "31.0": ARCHIVED, "2.09": ARCHIVED, "0.27": ARCHIVED, "0.76": ARCHIVED, "0.03": ARCHIVED, "0.32": README,
    "1.54": ARCHIVED, "0.73": README, "61.9": ARCHIVED, "3.99": ARCHIVED, "0.50": ARCHIVED, "2.51": ARCHIVED,
    "8.75": ARCHIVED, "2.89": ARCHIVED, "9.29": README,
    # target selection, Navigator, episode protocol
    "286": README, "2048": README, "0.45": NAVIGATOR, "40.0": NAVIGATOR, "1.2": NAVIGATOR, "0.04": NAVIGATOR,
    "0.05": NAVIGATOR, "0.55": README, "25": NAVIGATE, "12": MEDIA,
    # closed-loop episodes (media/README.md; CPU re-run in results/demo_rerun_cpu)
    "0.39": README, "0.41": README, "0.54": README, "0.75": README, "630": README, "387": README, "668": README,
    "386": ARCHIVED,
    # engineering log (archived report, section 5)
    "19": ARCHIVED, "1.6": ARCHIVED, "6000": ARCHIVED,
}

NUM = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)(?![0-9])")


def normSource(text: str) -> str:
    """Drop thousands separators (1,524 / 1{,}524 / 1\\,524 -> 1524)."""
    tText = text
    while True:
        tNew = re.sub(r"(\d)\\,(\d{3})(?!\d)", r"\1\2", tText)
        tNew = re.sub(r"(\d)\{,\}(\d{3})(?!\d)", r"\1\2", tNew)
        tNew = re.sub(r"(\d),(\d{3})(?![\d])", r"\1\2", tNew)
        if tNew == tText:
            return tText
        tText = tNew


def bodyOf(tex: str) -> str:
    tTex = re.sub(r"(?<!\\)%.*", "", tex)
    tTex = tTex[tTex.index("\\begin{document}"):]
    if "\\begin{thebibliography}" in tTex:
        tTex = tTex[:tTex.index("\\begin{thebibliography}")]
    return tTex


def numbersIn(tex: str) -> tuple[list[str], list[str]]:
    """(numbers in the text, numbers printed inside TikZ figures)."""
    tBody = normSource(bodyOf(tex))
    tTikz = "".join(re.findall(r"\\begin\{tikzpicture\}(.*?)\\end\{tikzpicture\}", tBody, re.S))
    tBody = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", tBody, flags=re.S)
    for tCmd in ("label", "ref", "cite", "code", "cmd", "texttt", "url", "includegraphics", "bibitem", "hypersetup",
                 "setlength"):
        tBody = re.sub(r"\\" + tCmd + r"(\[[^\]]*\])?\{[^{}]*\}(\{[^{}]*\})?", " ", tBody)
    tBody = re.sub(r"\\begin\{(tabular|table\*?|figure\*?)\}(\[[^\]]*\])?(\{(?:[^{}]|\{[^{}]*\})*\})?", " ", tBody)
    tBody = re.sub(r"\\\\\[[^\]]*\]", " ", tBody)                         # \\[1mm]
    tBody = re.sub(r"width=[0-9.]*\\(textwidth|columnwidth)", " ", tBody)
    tBody = tBody.replace("\\\\", " ")
    tBody = re.sub(r"\\[A-Za-z]+", " ", tBody)                            # remaining command names
    tPrinted = " ".join(re.findall(r"\)\s*\{((?:[^{}]|\{[^{}]*\})*)\}", tTikz))   # node texts of the TikZ figure
    tPrinted += " " + " ".join(re.findall(r"node\[[^\]]*\]\{((?:[^{}]|\{[^{}]*\})*)\}", tTikz))
    tPrinted = re.sub(r"\\[A-Za-z]+", " ", tPrinted.replace("\\\\", " "))
    return NUM.findall(tBody), NUM.findall(tPrinted)


def main(argv: list[str] | None = None) -> int:
    tArgs = sys.argv[1:] if argv is None else argv
    tRows = []

    def check(section, claim, ok, source):
        tRows.append((section, claim, source, "PASS" if ok else "FAIL"))

    tTex = (ROOT / REPORT).read_text()
    tNumbers, tTikzNumbers = numbersIn(tTex)
    tCache = {}
    for tNumber in dict.fromkeys(tNumbers):
        tSource = SOURCES.get(tNumber)
        if tSource is None:
            check("numbers", f"{tNumber}: unmapped (no source)", False, "-")
            continue
        if tSource not in tCache:
            tCache[tSource] = normSource((ROOT / tSource).read_text())
        tHit = re.search(r"(?<![0-9.])" + re.escape(tNumber) + r"(?![0-9])(?!\.\d)", tCache[tSource]) is not None
        check("numbers", f"{tNumber} appears in its source", tHit, tSource)
    for tNumber in sorted(set(SOURCES) - set(tNumbers)):
        check("map", f"{tNumber}: mapped but not in the report (stale entry)", False, SOURCES[tNumber])
    check("figure", f"TikZ figure prints no numbers (found: {sorted(set(tTikzNumbers))})", not tTikzNumbers, REPORT)

    tFlat = " ".join(tTex.split())
    check("status", "states that no success rate is reported", "no success rate is reported" in tFlat, REPORT)
    check("status", "states that the Eagle embedding does not influence the target",
          "does not influence the target" in tFlat, REPORT)
    check("status", "states that this is not a vision-language-action model",
          "not a vision-language-action model" in tFlat, REPORT)
    check("status", "states that target selection is a keyword rule with a ground-truth position",
          "keyword matching" in tFlat and "ground truth" in tFlat, REPORT)

    tFailed = sum(tRow[-1] == "FAIL" for tRow in tRows)
    if "--md" in tArgs:
        tLines = ["| section | claim | source | status |", "|---|---|---|---|"]
        tLines += [f"| {a} | {b} | `{c}` | {d} |" for a, b, c, d in tRows]
        tLines.append(f"\n{len(tRows)} claims checked, {tFailed} failed.")
        Path(tArgs[tArgs.index("--md") + 1]).write_text("\n".join(tLines) + "\n")
    for tRow in tRows:
        if tRow[-1] == "FAIL":
            print("FAIL:", tRow)
    print(f"{len(tRows)} claims checked, {tFailed} failed.")
    return 1 if tFailed else 0


if __name__ == "__main__":
    raise SystemExit(main())
