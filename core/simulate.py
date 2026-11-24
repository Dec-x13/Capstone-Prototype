"""Demo simulation (manuscript Section 4.1.3): insert user B's window into user A's timeline.

The window of k messages written by B is placed right after A's baseline (position `at`), under:

* normal timing   - the inserted messages copy A's own timestamps and recipients at that position,
                    so only writing, bubbles and language can differ;
* takeover timing - several similar messages go to different recipients within a short time:
                    2-6 recipients (random), gaps of 5-90 seconds (random), at a random hour.
                    B's first message is sent to each recipient (near-identical text); any remaining
                    slots are filled with B's next messages, still cycling through those recipients.

Every result is labelled SIMULATED. Use pipeline.evaluate(owner, at, ...) on A's own real window at
the same position for the natural-variation comparison.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from .features import Prepared
from .pipeline import WindowResult, evaluate

CONDITIONS = ("normal", "takeover")


@dataclass
class SimulationInfo:
    condition: str
    b_start: int
    at: int
    k: int
    n_recipients: int
    gaps_s: list[int]
    hour: int

    def describe(self) -> str:
        if self.condition == "normal":
            return (f"SIMULATED · normal timing · {self.k} of user B's messages placed at A's position {self.at}, "
                    "with A's own timestamps and recipients")
        return (f"SIMULATED · takeover timing · {self.n_recipients} recipients · gaps "
                f"{min(self.gaps_s)}-{max(self.gaps_s)} s · starting around {self.hour:02d}:00")


def build_simulated_timeline(owner: Prepared, other: Prepared, at: int, b_start: int, k: int,
                             condition: str = "normal", seed: int | None = None) -> tuple[Prepared, SimulationInfo]:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if b_start + k > len(other):
        raise ValueError("User B's window runs past the end of B's timeline.")
    rng = random.Random(seed)
    head = owner.take(range(at))

    if condition == "normal":
        if at + k > len(owner):
            raise ValueError("Normal timing needs k of A's own messages after the baseline to copy timing from.")
        win = other.take(range(b_start, b_start + k))
        src = owner.table.iloc[at:at + k]
        for col in ("ts", "hour", "recipient_code", "thread_type"):
            win.table[col] = src[col].to_numpy()
        info = SimulationInfo("normal", b_start, at, k, int(src["recipient_code"].nunique()), [], int(src["hour"].iloc[0]))
        return Prepared.concat([head, win]), info

    # takeover timing
    n_rec = min(rng.randint(2, 6), k) if k > 1 else 1
    pool = list(dict.fromkeys(owner.table["recipient_code"]))
    rng.shuffle(pool)
    recipients = pool[:n_rec]
    while len(recipients) < n_rec:  # owner has very few contacts
        recipients.append(f"R{900 + len(recipients)}")
    order = [b_start] * n_rec + [b_start + 1 + i for i in range(k - n_rec)]
    win = other.take(order)
    hour = rng.randint(0, 23)
    last = head.table["ts"].iloc[-1]
    t0 = (last + pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=hour, minutes=rng.randint(0, 50))
    gaps = [rng.randint(5, 90) for _ in range(k - 1)]
    stamps = [t0]
    for g in gaps:
        stamps.append(stamps[-1] + pd.Timedelta(seconds=g))
    win.table["ts"] = pd.Series(stamps)  # tz-aware, same zone as the owner's timeline
    win.table["hour"] = [s.hour for s in stamps]
    win.table["recipient_code"] = [recipients[i % n_rec] for i in range(k)]
    win.table["thread_type"] = "one-to-one"
    info = SimulationInfo("takeover", b_start, at, k, n_rec, gaps, hour)
    return Prepared.concat([head, win]), info


def simulate(owner: Prepared, other: Prepared, at: int, b_start: int, k: int, n: int,
             condition: str = "normal", seed: int | None = None, use_context: bool = True
             ) -> tuple[WindowResult, SimulationInfo]:
    timeline, info = build_simulated_timeline(owner, other, at, b_start, k, condition, seed)
    res = evaluate(timeline, at, k, n, use_context=use_context, simulated=True, label=info.describe())
    return res, info
