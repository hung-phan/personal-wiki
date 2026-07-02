# Personal Wiki

This repo is personal knowledge wiki. Notes live under `wiki/<category>/<topic>.md` and follow a fixed template so search and skim work consistently.

## Layout

```
wiki/<category>/INDEX.md       # one-line router per note in the category
wiki/<category>/<topic>.md     # the note (lowercase-hyphenated filename)
scripts/extract-manifest.py    # frontmatter search across all notes
```

## Frontmatter (required — the script reads this)

```markdown
---
name: <kebab-case-slug>
description: One sentence packed with the symptoms and trigger phrases future-me would search for. Generic descriptions never get matched.
created: YYYY-MM-DD
updated: YYYY-MM-DD
tags: [tag-one, tag-two]
---
```

Bump `updated:` on every edit. `description` is the most load-bearing field — it's what `extract-manifest.py` searches.

## Note template (use these section headings, in this order)

```markdown
# <Title — human-readable, not the slug>

## Why This Exists
**Problem.** What breaks or stays unclear without this note.
**Key insight.** The core idea in one sentence.
**Reach for this when:** bullet list of symptoms / triggers.
**Don't reach for this when:** bullet list of cases where this is the wrong tool.

## Diagrams
Mermaid diagrams when a picture pulls weight. Skip if a diagram would be ceremony.

## <Content sections>
The actual knowledge — concepts, code, patterns. Use `### Sub-headings` for each variant or step. Self-contained: don't lean on chat context that won't exist at read time.

## Trade-offs
What you give up by picking this. Every recommendation has a cost — name it.

## Common Pitfalls
Failure modes named with reasons. "Don't do X because Y" — both halves required.

## Decision Table
| Scenario | Use this | Not that | Why |
|----------|----------|----------|-----|

## References
Verified links to primary sources (`curl -sI <url> | head -1` returns 200). Never invent URLs.

## See Also
Relative links to neighboring notes: `[event-sourcing](../system-design/event-sourcing.md)`.
```

Drop a section if it would be one line of fluff. Keep it if it would be five lines of substance. The fixed order is what makes scanning fast — don't reorder.

## Teach with pictures and code, not just prose

The wiki is read in Obsidian with the [Execute Code](https://github.com/twibiral/obsidian-execute-code) plugin installed. Any concept that can be *seen* or *run* should be — reading raw formulas is the slowest way to learn.

**Diagrams.** Prefer SVG under `wiki/<category>/assets/` over ASCII art or Mermaid. SVGs render sharp in Obsidian light/dark themes and on GitHub. Include a `@media (prefers-color-scheme: dark)` block in the inline `<style>` so colors adapt. Embed as `![alt](assets/name.svg)`. Reserve Mermaid for tiny flow/graph diagrams where the source text carries meaning (e.g. a confounder DAG); pixel-precise charts belong in SVG.

**Python.** Anywhere a formula deserves verification or a picture, add a short runnable snippet:

- Use ` ```python ` fences (not `python3` / `py`) — the Execute Code plugin turns those into Run buttons and executes the block with the interpreter configured in its settings.
- **Self-contained**: imports at the top of each block; the reader can run one block without running any other.
- **Deterministic**: seed randomness (`rng = np.random.default_rng(0)`, `torch.manual_seed(0)`, `random_state=0`) so the same click always produces the same output.
- **Short**: 5–15 lines. Comparisons (analytic vs numerical, method A vs method B) teach; showcase code doesn't.
- **Fold expected output back into the block as trailing comments**, not into a separate `Result:` fence. The plugin still renders live output below the fence when clicked; the comment serves as a "here's what to expect" preview for reading. Format:

  ````
  ```python
  import numpy as np
  print(np.linalg.norm([3, -4, 12]))
  # Expected output:
  # 13.0
  ```
  ````

  Run the snippet once against the project venv, paste the captured stdout in as `#` comments — one per output line. Keep it terse: 10 lines max, add `# ... trimmed` if you cut. This co-locates the answer with the code (survives file moves, plugin off, GitHub view) without duplicating a whole fenced block.
- Prefer `matplotlib.pyplot` for plots when a chart teaches better than a static SVG — Execute Code embeds `plt.show()` output inline. Reserve hand-authored SVGs under `assets/` for diagrams that are *not* the direct output of a `plt.plot` call (schematic flows, geometry sketches, decision trees) — those stay in `assets/`.
- Use `numpy` as the default toolkit. Reach for `scipy.stats`, `sklearn`, `torch` only when they clarify the concept (autograd, lasso vs ridge, distributions).

**Interpreter setup.** The Execute Code plugin runs blocks with the Python interpreter set in Settings → Execute Code → Python. Point it at the project venv `/Users/phahng/Desktop/workarea/.venv/bin/python`. If a package is missing, install it into that venv via:

```
VIRTUAL_ENV=/Users/phahng/Desktop/workarea/.venv uv pip install <package>
```

(The venv was built by `uv`, so it doesn't ship `pip` — always use `uv pip install` from outside, not `python -m pip`.) Baseline dependencies for the math notes: `numpy scipy scikit-learn torch matplotlib`.

## Three jobs

- **Recall** ("do I have notes on X?") — start with `python3 scripts/extract-manifest.py <keyword>`, fall back to `grep -ril <keyword> wiki/`. Read the candidate in full before quoting. Cite the path. If nothing matches, say so plainly.
- **Capture** ("save this") — ask 2–4 scoping questions (topic in one sentence, where it goes, anchor sources, what's out of scope). Draft against the template above. Update the category's `INDEX.md`.
- **Refine** ("update / audit X") — read the file in full, edit in place, bump `updated:`. For multi-file topics, map every match (`grep -rln`), pick one canonical home, link the rest.

## Don't

- Quote a note from its `description` alone — read the body.
- Skip the template for a "simple" note — every note hits the same headings, that's the search affordance.
- Override a stale wiki claim silently — flag it so the note gets fixed.
- Duplicate content across notes — pick a canonical home and link.
