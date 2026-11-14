#!/usr/bin/env python3
"""Replace {{snip:path.py}} placeholders in a note draft with a runnable
```python block plus the snippet's real stdout as `# Expected output:` comments.

Usage:
    python3 scripts/inline_snippets.py draft.md wiki/<category>/<topic>.md

Each snippet is executed once with the project venv (override with SNIP_PY).
A non-zero exit aborts the whole file, so a note never ships output from a
snippet that failed. Output is capped at 10 lines (CLAUDE.md rule).
"""
import os
import re
import subprocess
import sys

MAX_LINES = 10


def run_block(match: re.Match) -> str:
    path = match.group(1)
    code = open(path).read().rstrip()
    py = os.environ.get("SNIP_PY", sys.executable)
    proc = subprocess.run(
        [py, path],
        env={**os.environ, "MPLBACKEND": "Agg"},  # plt.show() must not block
        capture_output=True,
        text=True,
        timeout=int(os.environ.get("SNIP_TIMEOUT", "1800")),
    )
    if proc.returncode:
        sys.exit(f"FAILED {path}\n{proc.stderr[-2000:]}")
    lines = proc.stdout.rstrip().splitlines()
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES] + ["... trimmed"]
    out = "\n".join("# " + line for line in lines)
    return f"```python\n{code}\n# Expected output:\n{out}\n```"


def main() -> int:
    if len(sys.argv) != 3:
        return print(__doc__) or 2
    draft, note = sys.argv[1], sys.argv[2]
    text = re.sub(r"\{\{snip:([^}]+)\}\}", run_block, open(draft).read())
    open(note, "w").write(text)
    print("wrote", note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
