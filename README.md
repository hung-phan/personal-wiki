# Personal Wiki

A personal knowledge base. Notes live under `wiki/<category>/<topic>.md` and follow a fixed template so search and skim work consistently. The repo is built around `CLAUDE.md` — drop into Claude Code from this directory and the agent reads it automatically, then helps me **recall**, **capture**, or **refine** notes.

## Layout

```
CLAUDE.md                      # operating manual — auto-loaded by Claude Code
scripts/extract-manifest.py    # frontmatter search across all notes
wiki/<category>/INDEX.md       # one-line router per note in the category
wiki/<category>/<topic>.md     # the note itself
```

There are no plugins, no skills, no marketplace metadata — just markdown, one Python script, and the CLAUDE.md that tells the agent how to use them.

## How to use it

```bash
cd /path/to/personal-wiki
claude
```

Claude reads `CLAUDE.md` and is ready for one of three jobs:

| I say... | The agent does... |
|----------|-------------------|
| "Do I have notes on cache stampedes?" | Runs `extract-manifest.py`, reads matching notes in full, quotes with the path. |
| "Save this lesson about Redis cluster failover." | Asks 2–4 scoping questions, drafts against the template, updates the category `INDEX.md`. |
| "Audit my notes on caching." | Maps every match across the wiki, picks one canonical home, links the rest, bumps `updated:`. |

## The note template

Every note hits the same headings in the same order — that consistency is the search affordance:

```
# Title

## Why This Exists      ← problem, key insight, when (and when not) to reach for this
## Diagrams             ← Mermaid, only if a picture pulls weight
## <Content>            ← the actual knowledge
## Trade-offs           ← what you give up by picking this
## Common Pitfalls      ← failure modes named with reasons
## Decision Table       ← Scenario | Use this | Not that | Why
## References           ← verified primary-source links
## See Also             ← relative links to neighboring notes
```

Sections are dropped only when they'd be one line of fluff. The order is fixed.

## Frontmatter

```markdown
---
name: <kebab-case-slug>
description: One sentence packed with symptoms and trigger phrases future-me would search for.
created: YYYY-MM-DD
updated: YYYY-MM-DD
tags: [tag-one, tag-two]
---
```

`description` is the most load-bearing field — it's what `extract-manifest.py` matches against. `updated:` is bumped on every edit.

## Searching

```bash
# Frontmatter search (preferred — token-cheap, no full reads)
python3 scripts/extract-manifest.py                    # all notes
python3 scripts/extract-manifest.py caching            # one keyword
python3 scripts/extract-manifest.py caching redis      # AND
python3 scripts/extract-manifest.py --any caching cdn  # OR

# Body-text fallback when frontmatter missed the term
grep -ril "<keyword>" wiki/
```
