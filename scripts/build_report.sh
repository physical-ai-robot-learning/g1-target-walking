#!/usr/bin/env bash
# Rebuild the archived Markdown engineering report report/archive/report.pdf from report/archive/report.md
# (pandoc -> LaTeX -> tectonic/XeTeX). The current technical report is report/report_en.tex (see report/README.md).
# Fonts: STIX Two Text + STIXGeneral fallback glyphs and Menlo (macOS system fonts);
# pass MAINFONT=/MONOFONT= to use others.
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
pandoc report/archive/report.md -s -o "$TMP/report.tex" \
  -V geometry:margin=2cm -V fontsize=10pt -V colorlinks=true \
  -V mainfont="${MAINFONT:-STIXGeneral}" -V monofont="${MONOFONT:-Menlo}" \
  --metadata title="G1Nav: Engineering Report" --metadata pagetitle="G1Nav report"
# the markdown already starts with its own title heading
sed -i.bak '/\\maketitle/d' "$TMP/report.tex"
(cd "$TMP" && tectonic -c minimal report.tex)
cp "$TMP/report.pdf" report/archive/report.pdf
rm -rf "$TMP"
echo "wrote report/archive/report.pdf"
