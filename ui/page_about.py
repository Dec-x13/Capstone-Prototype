"""Page 5: About. Plain prose, no performance figures."""
from __future__ import annotations

import streamlit as st

from core.config import cfg

STAGE_LINES = [
    ("Export reader", "Reads the Messenger export, fixes the text encoding, keeps only the owner's own sent "
                      "messages and replaces contacts with codes."),
    ("Language tagger", "Tags each word as Bikol, Tagalog, English, ambiguous or other, using seed word lists."),
    ("Feature extractor", "Turns each window of k messages into numbers in six behavior groups: timing, "
                          "bubble pattern, language mix, writing habits, recipients and entropy."),
    ("Baseline builder", "Summarises the N messages before the window as overlapping windows of the same size."),
    ("Deviation scorer", "Measures how far the window is from the baseline (z-scores), counts how many groups "
                         "changed, and scores it with an Isolation Forest trained only on the owner's baseline."),
    ("Explanation generator", "Writes the reasons in plain, neutral language."),
    ("Interface", "This app: shows the result, the reasons and the charts, with text masked."),
]


def render() -> None:
    st.subheader("What this prototype does")
    st.markdown("BehaviorGuard asks one question: **does this window of Messenger messages look like the account "
                "owner?** It learns each owner's habits from their own sent messages (when they write, how long "
                "their bubbles are, how they mix Bikol, Tagalog and English, how they laugh and abbreviate, and "
                "whom they message), then estimates how unusual a new window is for that owner.")

    st.subheader("The seven-stage pipeline (Table 4.5)")
    for i, (name, line) in enumerate(STAGE_LINES, 1):
        st.markdown(f"{i}. **{name}.** {line}")

    st.subheader("How to read the result")
    st.markdown(f"""
- **Low, Medium or High** compares the window's score with the owner's own baseline windows. Above the
  {cfg('risk', 'high_percentile')}th percentile *and* at least {cfg('risk', 'high_min_agreement')} changed behavior
  groups is High. Above the {cfg('risk', 'medium_percentile')}th percentile is Medium. Everything else is Low.
- **"3 of 6 behavior groups changed"** counts groups with at least one feature more than
  {cfg('deviation', 'z_threshold'):g} standard deviations from the owner's usual value.
- **Writing habits vs length.** Stretched letters, laughter, punctuation, capitals, emoji and abbreviations
  barely depend on the topic. A change there says more than a change in message length alone.
- **Context-matched** means the window was compared only with earlier windows sent to the same recipient.
- The result means "unusual for this user, worth verifying". It never says who wrote the messages.
""")

    st.subheader("Privacy design")
    st.markdown("""
- Only the owner's own sent messages are kept. Other people's messages are dropped while each file is read.
- Contacts become random codes such as R07. The code table exists only in memory for this session.
- Message text stays in memory, is never written to disk, logs or caches, and is masked
  (phone numbers, emails, links, long numbers) whenever it is shown.
- The app runs offline: no network calls, no API keys, no usage statistics.
- The participant extraction script shares only timestamps, codes, numeric features and salted hashes.
""")

    st.subheader("Limitations")
    st.markdown(f"""
- This is a **feasibility demo** on the two researchers' own exports and a synthetic sample. It reports
  **no performance figures**.
- **A change is a risk signal, not proof.** New topics, new group chats, being in a hurry, or a friend typing with
  permission can all look unusual.
- **Word lists are preliminary.** Native speakers will expand and check them, and the tagger will be checked
  against 500 hand-labeled words.
- **Cutoffs are prototype cutoffs**, not the validated thresholds of Section 4.2.2, which will be set on
  validation data later.
- The thesis will compare the Isolation Forest with an Extended Isolation Forest and a One-Class SVM.
""")
