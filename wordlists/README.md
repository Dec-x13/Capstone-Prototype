# Word lists (preliminary seeds)

These lists drive the rule-based language tagger in `core/language_tagger.py`.

**They are seeds, not finished lexicons.**

- `bikol.txt` and `tagalog.txt` start with the seed words from the manuscript. Below the seeds are extra
  starter words the prototype added so the pipeline runs end to end. Those extra words still **need review**.
- `english.txt` is a starter list of about 350 common English words.
- `abbreviations.txt` maps chat abbreviations (e.g. `lng`, `u`, `pls`, `nmn`, `d2`) to a full word. The tagger uses
  the full word for language lookup. `features.py` counts the abbreviation itself as a writing habit.

## How tagging works

- A word in **more than one** list is tagged `ambiguous`, for example `ako`, `iyo`, `man`, `na`, `ko`, `pa` and `sa`.
- Laughter (haha, hehe, lol, ...), emoji, numbers and words in no list are tagged `other`.

## Planned work

1. The team (native Bikol Naga speakers) expands `bikol.txt` and `tagalog.txt`.
2. A **second speaker** checks every entry.
3. The tagger will later be checked against **500 hand-labeled words**. Until then, no accuracy is claimed for it.

## File format

One lowercase word per line. Lines starting with `#` are comments. Duplicate lines are harmless.
