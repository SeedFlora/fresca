"""
Repair LaTeX table row terminators.

Scripted edits passed through a shell heredoc can collapse the two-backslash
row terminator to a single backslash, which silently merges table rows (LaTeX
reports "Extra alignment tab has been changed to cr" but still emits a PDF).
This restores a proper terminator on every line inside a tabular that ends with
exactly one backslash. Run it after any scripted edit to the .tex.
"""
import re
import sys

BS = chr(92)                      # a single backslash, written without escaping

path = sys.argv[1] if len(sys.argv) > 1 else "FRESCA-Net-ESC50.tex"
lines = open(path, encoding="utf-8").read().split("\n")

in_tab = False
fixed = 0
for i, line in enumerate(lines):
    if BS + "begin{tabular}" in line:
        in_tab = True
    elif BS + "end{tabular}" in line:
        in_tab = False

    if not in_tab:
        continue
    stripped = line.rstrip()
    # a row terminator that survived as a single trailing backslash
    if stripped.endswith(BS) and not stripped.endswith(BS + BS):
        lines[i] = stripped + BS
        fixed += 1

open(path, "w", encoding="utf-8").write("\n".join(lines))
print("repaired %d row terminators" % fixed)

# report anything still suspicious
bad = [i + 1 for i, l in enumerate(lines)
       if l.rstrip().endswith(BS) and not l.rstrip().endswith(BS + BS)]
print("lines still ending in a lone backslash:", bad if bad else "none")
