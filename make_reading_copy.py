#!/usr/bin/env python3
"""Build a self-contained 'reading copy' of paper_full_draft.md for PDF export.

The draft is LaTeX-flavoured markdown: it carries 57 `\\cite{key}` markers and a
hand-formatted reference list where every entry is prefixed with its cite key in
backticks (``- `unet` — Olaf Ronneberger ...``). Obsidian / pandoc won't resolve
`\\cite{}`, so a raw export shows literal "\\cite{unet}" all over the prose.

This script reads that key->reference mapping straight out of the reference list,
then emits paper_reading_copy.md with:
  * every `\\cite{a,b}` replaced by a bracketed number list, e.g. [1, 3];
  * the reference list rewritten as a plain numbered list (key prefix removed);
  * the References header cleaned up.

Numbers follow the *order of the reference list* (internally consistent: prose
[12] == list item 12). Appearance-order renumbering is a submission-time job for
the MDPI template; this is just a clean draft for a human to read.

The source draft with `\\cite{}` is left untouched -- that stays the .bib source.

Run:    python make_reading_copy.py
Output: paper_reading_copy.md  (+ a short report on stdout)
"""
import re
import sys

SRC = "paper_full_draft.md"
OUT = "paper_reading_copy.md"

# a reference list item: optional indent, a "-"/"*" bullet, `key` in backticks,
# an em/en-dash or hyphen separator (glued or spaced), then the reference text.
REF_RE = re.compile(r"^\s*[-*]\s+`(?P<key>[^`]+)`\s*[—–-]*\s*(?P<rest>.*\S)\s*$")
CITE_RE = re.compile(r"\\cite\{(?P<keys>[^}]*)\}")
HDR_RE = re.compile(r"^\s*#+\s*references\b", re.IGNORECASE)


def main():
    with open(SRC, encoding="utf-8") as f:
        lines = f.read().split("\n")

    # locate the References section header
    hdr = next((i for i, ln in enumerate(lines) if HDR_RE.match(ln)), None)
    if hdr is None:
        sys.exit("ERROR: could not find a '## References' header in " + SRC)
    body, refs = lines[:hdr], lines[hdr:]

    # pass 1: read key -> (number, text) from the reference list, in list order
    key2num, num2text, n = {}, {}, 0
    for ln in refs:
        m = REF_RE.match(ln)
        if m:
            n += 1
            key2num[m["key"]] = n
            num2text[n] = m["rest"]

    # pass 2: rewrite \cite{...} in the body to [n] (or [n, m])
    missing, n_cites = set(), 0

    def sub(m):
        nonlocal n_cites
        nums = []
        for k in (k.strip() for k in m["keys"].split(",")):
            if not k:
                continue
            n_cites += 1
            if k in key2num:
                nums.append(key2num[k])
            else:
                missing.add(k)
                nums.append("?")
        return "[" + ", ".join(str(x) for x in nums) + "]"

    body = [CITE_RE.sub(sub, ln) for ln in body]

    # rebuild the references section as a clean numbered list
    out_refs, n = [], 0
    for ln in refs:
        if HDR_RE.match(ln):
            out_refs.append("## References")
        elif REF_RE.match(ln):
            n += 1
            out_refs.append(f"{n}. {num2text[n]}")
        else:
            out_refs.append(ln)  # blank lines, trailing TODO comment, etc.

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(body + out_refs))

    print(f"wrote {OUT}")
    print(f"  references mapped : {len(key2num)}")
    print(f"  \\cite keys resolved: {n_cites - len(missing)} / {n_cites}")
    if missing:
        print("  WARNING unmapped keys (left as '?'):", ", ".join(sorted(missing)))


if __name__ == "__main__":
    main()
