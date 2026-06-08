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

## Three jobs

- **Recall** ("do I have notes on X?") — start with `python3 scripts/extract-manifest.py <keyword>`, fall back to `grep -ril <keyword> wiki/`. Read the candidate in full before quoting. Cite the path. If nothing matches, say so plainly.
- **Capture** ("save this") — ask 2–4 scoping questions (topic in one sentence, where it goes, anchor sources, what's out of scope). Draft against the template above. Update the category's `INDEX.md`.
- **Refine** ("update / audit X") — read the file in full, edit in place, bump `updated:`. For multi-file topics, map every match (`grep -rln`), pick one canonical home, link the rest.

## Don't

- Quote a note from its `description` alone — read the body.
- Skip the template for a "simple" note — every note hits the same headings, that's the search affordance.
- Override a stale wiki claim silently — flag it so the note gets fixed.
- Duplicate content across notes — pick a canonical home and link.
