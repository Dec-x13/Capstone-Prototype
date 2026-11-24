# Live demo script: graded defense (7 to 8 minutes)

**Before the panel arrives (5 min)**

- Run `run_demo.bat`, open <http://localhost:8501>, and zoom the browser to 125–150% for the projector.
- Keep both exports unzipped **outside** the repo, e.g. `C:\thesis-data\me\` and `C:\thesis-data\partner\`.
- Rehearse once with the synthetic sample (fallback path below). Press **⟲ Reset** before the panel starts.
- Turn off Wi-Fi to show that the demo runs offline. This is optional but a strong point to make.
- Keep a second terminal open in the project folder for step 6.

Sidebar defaults: **k = 5**, **N = 100**, context-matched baseline **on**.

---

### 1. Load my export (≈1 min)

**Tab 1 · Load → Account owner (user A) → "Export folder on this computer"** → paste the path.

> "The app detects the export layout and suggests the most frequent sender as the owner. I confirm my name.
> Only my own sent messages are kept. Everyone else's messages are dropped in memory while each file is read."

Click **Load messages**, then point at the counts: messages kept, recipients, date range.
> "There is no message text on this screen, only counts. Contacts are already random codes like R07."

Load your partner's export into **user B** now, since step 4 needs it.

### 2. Show my baseline (≈1 min)

**Tab 2 · Baseline.**
> "The baseline is the 100 messages right before the window being checked, cut into overlapping windows of 5.
> These are my habits: when I am active, how long my bubbles are, my language mix, and how I laugh."

Point at the hour-of-day chart, the language-mix bar (dominant mix), the laughter style, and the habits table.

### 3. Check one of my own real windows (≈1 min)

**Tab 3 · Check a window → Pick from the timeline.** Keep the latest window or one you have rehearsed.
> "This is my own real window. The estimated risk is Low. The reasons line says how many of the six behavior
> groups changed. A few small changes are normal, and that is natural variation."

Point at the **Reference** caption (context-matched or whole baseline) and the per-group chart with the |z| = 2 line.
Optionally open **Show this window's messages (masked)** to show `[PHONE]` / `[URL]` masking.

### 4. Insert my partner's window: normal timing, then takeover timing (≈2 min)

**Tab 4 · Simulate takeover.** Point at the banner: "Everything here is SIMULATED."

1. **Normal timing** → **Run simulation**.
   > "My partner's 5 messages take my timestamps and my recipients. Only the writing can differ: bubbles,
   > language and habits."
2. **Takeover timing** → **Run simulation**.
   > "Now the same writing is sent as near-identical messages to several recipients, seconds apart, at a random hour.
   > This is the pattern Section 4.1.3 describes."

Compare the left side (my own later window) with the right side (the inserted window).

### 5. Read the reasons aloud (≈1 min)

Read the right-hand reasons. For example:
> "Mostly Taglish, usually Bi-lish. Long paragraphs, usually short bubbles. Laughs with 'lol', usually 'HAHAHA'.
> Three of six behavior groups changed. The change includes topic-independent habits, not only message length."

> "The system does not say who wrote it. It says this is unusual for this user and worth verifying."

### 6. Extraction script preview: no text is shared (≈1 min)

In the second terminal:

```
.venv\Scripts\python extract\extract_features.py "C:\thesis-data\me" --owner "My Name" --preview
```

> "This is what participants will run after ethics clearance. The summary lists exactly what would be shared:
> timestamps, random recipient codes, numeric features, and salted hashes of words. The salt is thrown away,
> so the hashes cannot be turned back into words. In preview mode nothing is saved. Our tests check that no
> original word appears anywhere in the output file."

### 7. Close with the limitations (≈30 s)

**Tab 5 · About / limitations.**
> "This is a feasibility demo on our own two exports. It gives no performance figures. A change is a risk signal, not proof.
> The word lists are preliminary, and the Low/Medium/High cutoffs are prototype cutoffs. The validated thresholds
> come later from validation data."

---

## Fallback path: if a live export fails

1. Press **⟲ Reset**.
2. Tab 1: choose **"Synthetic sample (for testing only)"** for user A and **Load**, then do the same for user B.
   The blue label stays visible in the sidebar.
3. Run steps 2–5 exactly as above. Say: *"These are two invented users made for testing. User A writes Bi-lish
   with 'hiiii' and 'HAHAHA'. User B writes Taglish paragraphs with 'lol'."*
4. Step 6: `.venv\Scripts\python extract\extract_features.py data\sample\SYNTHETIC\user_a --owner "Ana Synthetica" --preview`

If the app itself will not start: `.venv\Scripts\python -m pytest` shows the pipeline tests passing, and the
synthetic steps above can be walked through from screenshots.

## Likely panel questions

**"Isn't a long paragraph just the owner explaining something?"**
It can be, which is why one feature never decides alone. The **agreement score** counts how many of the six behavior
groups changed. The app also says whether the change is in **topic-independent habits** (stretched letters, laughter,
punctuation, capitals, emoji, abbreviations) or **only in length**. Only-length changes are described as such. We also
plan an **owner circumstance-shift test**: we will check the owner's own windows from periods when their situation changed
(busy weeks, new group chats) to see how often natural change raises the level.

**"How accurate is it?"**
We do not claim any figure. This prototype shows the pipeline works end to end on our own data. Evaluation needs labeled
data and the validated thresholds of Section 4.2.2, and those come after data collection.

**"Why Isolation Forest?"**
It is trained only on the owner's own baseline windows, so it needs no examples of takeovers. The detector is pluggable.
The thesis compares an Extended Isolation Forest and a One-Class SVM next.

**"Why does High need agreement of at least 2?"**
A single odd feature (one emoji-heavy window, for example) should not reach High on its own. High needs both a score that is rare
for the owner and changes in at least two behavior groups.

**"What is the context-matched baseline?"**
If at least 10 baseline windows went to the same recipient, the window is compared with those windows, because people
write differently to different people. Otherwise the whole baseline is used. The app shows which one it used.

**"How do you tag Bikol vs Tagalog when words overlap?"**
Words in more than one list (ako, iyo, man) are tagged *ambiguous* and are left out of code-switching pairs. The lists
are seeds that native speakers will expand. A second speaker will check them, and the tagger will be checked against 500 hand-labeled words.

**"Do you store our messages?"**
No. Text is read into memory, turned into numbers, and never written to disk, logs or caches. Contacts become random codes.
Anything shown is masked. Participants share only the extraction output.

**"What if the owner lets a friend type for them?"**
The system would likely mark it as unusual. That is correct behavior for a risk signal, and it is why the output says
"worth verifying" and not a verdict.
