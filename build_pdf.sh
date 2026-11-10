#!/usr/bin/env bash
# Build a shareable PDF of the manuscript (e.g. to send an advisor) from the
# LaTeX-flavoured markdown draft. One command, repeatable:
#
#     bash build_pdf.sh            ->  paper_draft.pdf
#
# Pipeline:
#   paper_full_draft.md
#     -> make_reading_copy.py : resolve \cite{key} to [n], renumber the ref list
#     -> sed .svg)->.pdf)     : LaTeX embeds the vector .pdf figures, not .svg
#     -> pandoc + tectonic    : XeLaTeX-class engine; header maps the few glyphs
#                               Latin Modern lacks and shrinks the wide tables.
#
# Requires (already set up on this machine): conda env `Pandoc` with pandoc +
# tectonic.  The source draft keeps \cite{} untouched -- that's the .bib source.
set -euo pipefail
cd "$(dirname "$0")"

# make `conda` usable in a non-interactive shell, if it isn't already
if ! command -v conda >/dev/null 2>&1; then
  source /root/miniconda3/etc/profile.d/conda.sh
fi

python make_reading_copy.py
sed 's/\.svg)/.pdf)/g' paper_reading_copy.md > paper_pdf_build.md

# -f markdown-implicit_figures: don't auto-caption images; the draft already has
#   its own "**Figure N.** ..." caption paragraph under each figure.
conda run -n Pandoc pandoc paper_pdf_build.md -o paper_draft.pdf \
  --pdf-engine=tectonic \
  -f markdown-implicit_figures \
  -V geometry:margin=1.6cm -V fontsize=10pt -V colorlinks=true \
  -H pdf_header.tex

echo "wrote paper_draft.pdf"
