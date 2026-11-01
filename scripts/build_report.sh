#!/usr/bin/env bash
# Rebuild docs/report.pdf from docs/report.md (pandoc -> LaTeX -> tectonic/XeTeX).
# Fonts: STIX Two Text + STIXGeneral fallback glyphs and Menlo (macOS system fonts);
# pass MAINFONT=/MONOFONT= to use others.
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
pandoc docs/report.md -s -o "$TMP/report.tex" \
  -V geometry:margin=2cm -V fontsize=10pt -V colorlinks=true \
  -V mainfont="${MAINFONT:-STIXGeneral}" -V monofont="${MONOFONT:-Menlo}" \
  --metadata title="G1Nav: Engineering Report" --metadata pagetitle="G1Nav report"
# the markdown already starts with its own title heading
sed -i.bak '/\\maketitle/d' "$TMP/report.tex"
(cd "$TMP" && tectonic -c minimal report.tex)
cp "$TMP/report.pdf" docs/report.pdf
rm -rf "$TMP"
echo "wrote docs/report.pdf"
