"""Generate the SYNTHETIC sample Messenger exports used for development and tests.

Everything here is invented: names, contacts and text. Nothing comes from a real person.

  user_a  "Ana Synthetica"  -> older layout   messages/inbox/<thread>/message_1.json
  user_b  "Ben Synthetico"  -> newer layout   your_facebook_activity/messages/inbox/<thread>/message_1.json

User A: mostly Bi-lish (Bikol + English), short frequent bubbles, "hiiii", "HAHAHA", active 8 AM to 11 PM,
        plus legitimate "pa-GCash" requests (hard negatives).
User B: mostly Taglish, longer paragraphs, "lol", no stretched letters, active early afternoon to ~1 AM.

Text is written the way Facebook writes it: UTF-8 bytes mis-read as latin-1, then JSON-escaped,
so the reader's encoding fix is exercised.

Run:  python data/sample/generate_sample.py
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "SYNTHETIC"
MANILA = timezone(timedelta(hours=8))
START = datetime(2026, 6, 1, tzinfo=MANILA)

# ---------------------------------------------------------------- text banks (invented)
A_FRAGS = [
    "dai pa ko tapos", "garo okay man", "tano ka late", "sige omw na", "igwa ka class later?",
    "maray man ang result", "dakol pa assignment ko", "bako man ako", "sain ka na", "wait lng",
    "ngani so true", "pls send the file", "baga tired na ko", "mayo pa ko food", "iyo see u later",
    "thank u tabi", "ok ok", "dai ko aram", "garo ma-uran", "nice one", "maray ta", "tara na",
    "sure sure", "igwa pa meeting", "dakol salamat", "send ko later", "u there?", "check mo tabi",
    "bako ini ang file", "garo late ako", "where na u", "maray na lang", "dai man problem",
]
A_GREET = ["hiiii", "hiii", "heyyy", "helloooo", "goodmorniiing"]
A_LAUGH = ["HAHAHA", "HAHAHAHA", "HAHAHA grabe", "HAHAHA dai ko kaya"]
A_EMOJI = ["\U0001F602", "\U0001F64F", "❤️", "\U0001F62D"]
A_GCASH = [
    "pa-gcash man tabi 200, bayad ko bukas",
    "pa gcash tabi sa 09171234567 pang load lang",
    "pwede pa-gcash 150? dai pa ko sweldo HAHAHA",
    "pa-gcash tabi pambayad sa printing, thank u",
]

B_SENT = [
    "hindi ko alam kung bakit pero sobrang busy talaga ako this week",
    "sige lang pwede naman natin pag-usapan bukas pagkatapos ng class",
    "ano ba plan natin for the weekend kasi wala pa akong idea",
    "kasi naman wala pa yung update from the team kaya hindi pa ako makapag-start",
    "gusto ko sana mag-meeting tayo later para ma-finalize na natin",
    "nasaan na yung files na sinend mo kanina hindi ko makita sa drive",
    "okay lang yun basta ma-submit natin bago mag deadline",
    "bakit naman ganun yung sinabi niya parang hindi fair sa atin",
    "pwede mo ba akong samahan bukas sa mall may bibilhin lang ako",
    "talaga ba hindi ko alam na ganun pala yung nangyari",
    "kailan ba yung next na meeting natin para ma-prepare ko na",
    "ang dami ko pang gagawin pero sige gagawin ko na lang mamaya",
    "sana maayos na natin ito kasi nakaka-stress na talaga",
    "Señor ng tita ko yung may-ari ng shop kaya pwede tayo doon",
]
B_SHORT = ["lol", "sige", "okay", "lol sige", "ano na", "wait lang"]

OTHERS_TEXT = ["ok", "sige", "see you", "thanks!", "nice", "haha", "where na", "omw"]

A_CONTACTS = ["Carla Inventa", "Dino Fabrica", "Elle Ficta", "Fritz Imaginaria", "Gina Placeholder",
              "Hugo Mockado", "Iris Fakedo"]
B_CONTACTS = ["Jojo Inventado", "Kara Ficticia", "Luis Sampol", "Mara Ejemplo", "Nico Prueba", "Olga Testa"]
A_GROUP = ("Barkada Synthetic GC", ["Carla Inventa", "Dino Fabrica", "Elle Ficta"])
B_GROUP = ("Thesis Synthetic GC", ["Jojo Inventado", "Kara Ficticia", "Luis Sampol"])


def fb_mangle(s: str) -> str:
    """Reproduce Facebook's bug: UTF-8 bytes stored as if each byte were a latin-1 character."""
    return s.encode("utf-8").decode("latin-1")


def stretch(word: str, rng: random.Random) -> str:
    i = rng.randrange(len(word))
    if not word[i].isalpha():
        return word
    return word[:i] + word[i] * rng.randint(3, 5) + word[i + 1:]


def a_message(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.08:
        return rng.choice(A_GREET)
    if r < 0.20:
        return rng.choice(A_LAUGH)
    if r < 0.24:
        return rng.choice(A_GCASH)
    words = rng.choice(A_FRAGS).split()
    if rng.random() < 0.25:
        j = rng.randrange(len(words))
        words[j] = stretch(words[j], rng)
    text = " ".join(words)
    if rng.random() < 0.10:
        text += " " + rng.choice(A_EMOJI)
    if rng.random() < 0.10:
        text += " HAHAHA"
    return text


def b_message(rng: random.Random) -> str:
    if rng.random() < 0.22:
        return rng.choice(B_SHORT)
    n = rng.randint(1, 3)
    sents = rng.sample(B_SENT, n)
    text = ". ".join(s[0].upper() + s[1:] for s in sents)
    if rng.random() < 0.25:
        text += " lol"
    return text


def sessions(rng: random.Random, n_target: int, hours: list[int], burst: tuple[int, int],
             gap: tuple[int, int]):
    """Yield (local datetime, session_id) for owner messages, grouped in short bursts."""
    t = START
    produced = 0
    sid = 0
    while produced < n_target:
        t = t + timedelta(hours=rng.uniform(1.5, 9))
        if t.hour not in hours:
            continue
        size = rng.randint(*burst)
        cur = t
        for _ in range(size):
            yield cur, sid
            produced += 1
            cur = cur + timedelta(seconds=rng.randint(*gap))
        sid += 1
        t = cur


def build_user(owner: str, contacts: list[str], group, make_text, rng: random.Random, n_target: int,
               hours: list[int], burst, gap):
    threads: dict[str, dict] = {}
    for c in contacts:
        threads[c] = {"participants": [c, owner], "messages": [], "group": False, "title": c}
    gname, gmembers = group
    threads[gname] = {"participants": gmembers + [owner], "messages": [], "group": True, "title": gname}
    keys = list(threads)
    weights = [6, 5, 4, 3, 2, 1, 1][: len(contacts)] + [3]

    current = None
    last_sid = -1
    for when, sid in sessions(rng, n_target, hours, burst, gap):
        if sid != last_sid:
            current = rng.choices(keys, weights=weights)[0]
            last_sid = sid
            # someone else talks first sometimes (must be dropped by the reader)
            if rng.random() < 0.5:
                other = rng.choice([p for p in threads[current]["participants"] if p != owner])
                threads[current]["messages"].append(
                    {"sender_name": other, "ts": when - timedelta(seconds=rng.randint(20, 300)),
                     "content": rng.choice(OTHERS_TEXT)})
        threads[current]["messages"].append({"sender_name": owner, "ts": when, "content": make_text(rng)})
        # occasional non-text owner items that the reader must skip
        r = rng.random()
        extra_ts = when + timedelta(seconds=2)
        if r < 0.02:
            threads[current]["messages"].append({"sender_name": owner, "ts": extra_ts,
                                                 "photos": [{"uri": "photos/synthetic.jpg"}]})
        elif r < 0.035:
            threads[current]["messages"].append({"sender_name": owner, "ts": extra_ts,
                                                 "sticker": {"uri": "stickers/synthetic.png"}})
        elif r < 0.045:
            threads[current]["messages"].append({"sender_name": owner, "ts": extra_ts,
                                                 "is_unsent": True, "content": ""})
        elif r < 0.05:
            threads[current]["messages"].append({"sender_name": owner, "ts": extra_ts,
                                                 "call_duration": 63,
                                                 "content": f"{owner} called you."})
    return threads


def write_threads(threads: dict, inbox: Path):
    for i, (name, th) in enumerate(threads.items()):
        folder = inbox / f"{name.lower().replace(' ', '')}_{1000 + i}"
        folder.mkdir(parents=True, exist_ok=True)
        msgs = []
        for m in sorted(th["messages"], key=lambda m: m["ts"], reverse=True):  # FB stores newest first
            out = {"sender_name": fb_mangle(m["sender_name"]),
                   "timestamp_ms": int(m["ts"].timestamp() * 1000)}
            for k in ("content", "photos", "sticker", "is_unsent", "call_duration"):
                if k in m:
                    out[k] = fb_mangle(m[k]) if k == "content" else m[k]
            msgs.append(out)
        doc = {
            "participants": [{"name": fb_mangle(p)} for p in th["participants"]],
            "messages": msgs,
            "title": fb_mangle(th["title"]),
            "is_still_participant": True,
            "thread_path": f"inbox/{folder.name}",
            "magic_words": [],
        }
        with open(folder / "message_1.json", "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=True, indent=1)


def main():
    rng_a = random.Random(2026_01)
    rng_b = random.Random(2026_02)
    a = build_user("Ana Synthetica", A_CONTACTS, A_GROUP, a_message, rng_a, 520,
                   hours=list(range(8, 24)), burst=(2, 7), gap=(4, 45))
    b = build_user("Ben Synthetico", B_CONTACTS, B_GROUP, b_message, rng_b, 450,
                   hours=list(range(13, 24)) + [0], burst=(1, 4), gap=(30, 240))
    write_threads(a, ROOT / "user_a" / "messages" / "inbox")
    write_threads(b, ROOT / "user_b" / "your_facebook_activity" / "messages" / "inbox")
    (ROOT / "README.txt").write_text(
        "SYNTHETIC sample (for testing only).\n"
        "Invented users, contacts and text generated by data/sample/generate_sample.py.\n"
        "No real person's messages are in this folder.\n", encoding="utf-8")
    print("Synthetic sample written to", ROOT)


if __name__ == "__main__":
    main()
