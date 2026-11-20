#!/usr/bin/env bash
# Build a PDF from a Markdown report with pandoc (P0-T02).
#
#   scripts/build_pdf.sh reports/milestone1/milestone1_report.md [output.pdf]
#
# Output defaults to the same path with a .pdf extension. Images are resolved relative
# to the report's folder, then the repository root, so both `../figures/x.png` and
# `reports/figures/x.png` work. Requires pandoc and a PDF engine; the default engine is
# xelatex (TeX Live / MiKTeX). Override with: PDF_ENGINE=lualatex scripts/build_pdf.sh ...
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <report.md> [output.pdf]" >&2
  exit 2
fi

input="$1"
output="${2:-${input%.md}.pdf}"
engine="${PDF_ENGINE:-xelatex}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
input_dir="$(cd "$(dirname "$input")" && pwd)"

if [[ ! -f "$input" ]]; then
  echo "error: $input not found" >&2
  exit 1
fi
for tool in pandoc "$engine"; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: '$tool' is not installed. See README (Building report PDFs)." >&2
    exit 1
  fi
done

pandoc "$input" \
  --from markdown \
  --pdf-engine="$engine" \
  --resource-path="$input_dir:$repo_root" \
  --standalone \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V colorlinks=true \
  -V linkcolor=blue \
  -V urlcolor=blue \
  -o "$output"

echo "wrote $output"
