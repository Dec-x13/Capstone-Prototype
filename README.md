# BehaviorGuard: feasibility prototype

Prototype for the senior thesis *"Learning Individual Communication Patterns for Detecting Suspicious Chat
Interactions: A Hybrid Content and Behavioral Deviation Approach for Code-Switched Filipino Messaging"*
(BS Computer Science, Ateneo de Naga University).

It implements the **behavior-only pipeline** of the manuscript (Section 4.2.3, Table 4.5). Each Messenger user gets a
baseline built from their own sent messages. A new window of messages is compared with that baseline in six behavior
groups: timing, bubble pattern, language mix (Bikol / Tagalog / English), writing habits, recipients, and entropy.
The app then estimates a risk level (Low / Medium / High) and lists plain-language reasons.

> **This is a feasibility demo.** It reports no performance figures. A change in behavior is a risk signal worth
> verifying, never proof of anything. The word lists are preliminary seeds. The risk cutoffs are **prototype cutoffs**,
> not the validated thresholds of Section 4.2.2. Those will be set later on validation data.

## Setup

Requires Python 3.11. The launchers create `.venv` and install the pinned `requirements.txt` on first run. After that,
everything runs offline.

| OS | Command |
|---|---|
| Windows | double-click `run_demo.bat` (or run it from a terminal) |
| macOS / Linux | `./run_demo.sh` |

Then open <http://localhost:8501>.

Manual setup:

```
uv venv --python 3.11 .venv             # or: py -3.11 -m venv .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python -m streamlit run app.py
.venv/Scripts/python -m pytest
```

Dependencies: pandas, numpy, scikit-learn, streamlit, plotly, pytest (pinned in `requirements.txt`).

## Privacy design

| Rule | How the code follows it |
|---|---|
| Keep only the owner's sent messages | `export_reader.load_owner_messages` drops other senders' messages while each thread file is read, and then clears each thread's message list. |
| No contact names | `anonymize.RecipientCoder` maps every contact or group to a random code (R07, R42, ...). The salt is random per session and kept only in memory. |
| Mask what is displayed | `anonymize.mask` replaces PH phone numbers (09xx, +63), emails, URLs and digit runs of 6 or more with `[PHONE]`, `[EMAIL]`, `[URL]`, `[NUMBER]`. The app masks every message it shows, and the messages are hidden in a collapsed panel by default. |
| No raw text on disk | Text lives only in memory (`st.session_state`). `features.prepare_messages` turns messages into numbers early. The app uses no `st.cache_data` at all, and nothing is logged or saved. |
| Offline | No network imports (checked by `tests/test_offline_privacy.py`). Streamlit runs headless, bound to localhost, with usage stats off (`.streamlit/config.toml` and the launcher flags). |
| Participant extraction | `extract/extract_features.py` writes only timestamps, recipient codes, numeric features and salted-hash counts. A test checks that no original token appears in the output. |

Real exports belong outside the repository, or in `exports/`, which `.gitignore` excludes. They can stay zipped: the reader opens only the message JSON inside the zips, in memory.

## Modules (pipeline order, Section 4.2.3 / Table 4.5)

| Pipeline step | Module | What it does |
|---|---|---|
| Data acquisition | `core/export_reader.py`, `core/zip_source.py` | Auto-detects the export layout (registry; currently `legacy` `messages/inbox`, `dyi_2024` `your_facebook_activity/messages`, checked against a real 2026 export, and a **provisional** `e2ee` reader). Reads every thread subfolder, including `e2ee_cutover`. Accepts an unzipped folder, one .zip, or the folder of .zip parts Facebook splits a large export into; zips are read in place and only the message JSON is opened, so nothing is extracted. Fixes Facebook's latin-1/UTF-8 encoding bug. Keeps the owner's text messages and skips media, calls, unsent messages and system lines ("You joined the call", "You pinned a message", ...). Returns `ts` (Asia/Manila), `recipient_code`, `thread_type`, `text`. |
| Anonymization | `core/anonymize.py` | Random recipient codes and display masking. |
| Language tagging | `core/language_tagger.py`, `wordlists/` | Tags each token as bikol, tagalog, english, ambiguous or other. Stretched letters are shortened (3+ repeats become 1, then 2) before lookup, and abbreviations are expanded. Also computes code-switching and the dominant mix rule (below). |
| Windowing | `core/windows.py` | Windows of k messages (1/3/5/10). The baseline is the previous N messages (50/100/150/200), split into sliding windows of size k. A burst means each message comes within 60 s of the previous one. |
| Feature extraction | `core/features.py` | Table 4.3 features in six groups (see below). |
| Baseline | `core/baseline.py` | Baseline windows, with leave-one-out n-gram distance and leave-one-out "new recipient". |
| Deviation | `core/deviation.py` | z-scores with a std floor, n-gram cosine distance, entropy ΔH, agreement score, topic-independent vs length-based summary scores, and the context-matched baseline. |
| Scoring | `core/scorer.py` | Pluggable detector (Isolation Forest now; Extended Isolation Forest and One-Class SVM are compared later in the thesis), max-\|z\| fallback, and prototype percentile cutoffs. |
| Explanation | `core/explain.py` | Neutral plain-language reasons, the agreement line, and a habits-vs-length line. |
| Demo simulation | `core/simulate.py` | Inserts user B's window into A's timeline under normal or takeover timing (Section 4.1.3). |
| Orchestration | `core/pipeline.py` | Runs baseline → deviation → score → reasons for one window. |
| Interface | `app.py`, `ui/` | Streamlit pages: Load data, Your baseline, Check a window, Simulate a takeover, About. It also has a pipeline strip for the seven Table 4.5 stages, the risk result card, presenter mode, light and dark themes (system fonts only, `ui/theme.css`), and one shared Plotly template (`ui/theme.py`). |
| Participant tool | `extract/extract_features.py` | Offline feature extraction with `--preview`. |

All working values live in `config.toml`.

### Features (Table 4.3)

| Group | Features |
|---|---|
| Timing | mean p_h of the window's hours under the baseline hour profile (Laplace-smoothed); messages per hour; median gap |
| Bubble pattern | average words per bubble; share of bubbles over 15 words; bubbles per burst |
| Language mix | Bikol / Tagalog / English / ambiguous shares; code-switching rate |
| Writing habits | stretched-letter rate; laughter style shares (HAHAHA, hahaha, hehe, lol, none); ? and ! per message; capital-letter share; emoji per message; abbreviation rate; char 2-4-gram profile distance |
| Recipients | share to recipients not in the baseline; distinct recipients; near-identical text to 2+ recipients |
| Entropy | Shannon entropy (bits) of hours, recipients, languages |

### Choices the manuscript leaves open (implemented as follows)

- **Dominant mix:** this rule looks only at Bikol, Tagalog and English words. If the second most common language
  has at least 20% of those words (`[language].mix_second_share`), the mix is the pair: Bi-lish, Tag-col or Taglish.
  Otherwise it is the single top language.
- **Ambiguous words:** `ako`, `iyo`, `man`, `na`, `ko`, `pa` and `sa` are in more than one list, so they are tagged ambiguous.
- **Entropy:** ΔH = H(window) − mean H(baseline windows of size k).
- **Leave-one-out:** each baseline window's n-gram distance and "new recipient" share are measured against the
  rest of the baseline, so a window is never compared with itself.
- **Context matching:** this keys on the window's most frequent recipient. With at least 10 baseline windows to that
  recipient, z-scores, the agreement score and the reasons use those windows. The detector is always trained on the
  whole baseline (N − k + 1 windows), because 10 heavily overlapping windows are too few to train it on. The UI shows
  which reference was used.
- **Detector input:** this is `[scorer].detector_input`. The default `group_max` gives the Isolation Forest the
  largest |z| in each of the six groups. The full 24 z-scores (`features`) hid large deviations on the synthetic sample.
  Isolation Forest picks split features at random and cannot split on features that never vary in the baseline.
- **Risk level:** this is the percentile of the window's detector score among the owner's own baseline windows. Above
  the 99th percentile with agreement ≥ 2 is High, above the 95th is Medium, and anything else is Low. The max-|z|
  fallback is always shown beside the result.
- **Bursts, messages per hour:** bubbles per burst is computed inside the window. Messages per hour floors the span
  at 1 minute. Timing features are undefined for k = 1 and are skipped.
- **Takeover simulation:** this picks 2–6 recipients (capped at k), with gaps of 5–90 s at a random hour. B's first
  message goes to each recipient, so several near-identical messages appear. Any remaining slots get B's next messages.

## Synthetic sample

`data/sample/SYNTHETIC/` holds two invented users, regenerated by `python data/sample/generate_sample.py`.
User A uses the older export layout and user B the newer one. The UI labels it **"Synthetic sample (for testing only)"**.
No real person's messages are in it.

## Tests

`python -m pytest` covers the following:

- encoding fix (Filipino text, ñ, emoji)
- layout detection
- masking
- tagger rules
- window, burst and code-switching math
- features, z-scores, agreement, context matching and entropy
- scorer and wording
- simulation
- a headless app smoke test
- the extractor's no-text-leak test with a negative control
- static offline and privacy checks

## Limitations

- Feasibility only: there are no labeled takeover data and no performance figures.
- The word lists are seeds. A native speaker must review them, and the tagger must be checked against 500 hand-labeled words.
- The cutoffs are prototype cutoffs.
- The separate E2EE-download reader is provisional. The real 2026 export put encrypted chats in `e2ee_cutover/`, which the main reader handles.
- A behavior change has many innocent causes, such as a new topic, a new group or a different device. The output says "unusual for this user", not "not the user".
- Sliding baseline windows overlap, so they are not independent samples.
