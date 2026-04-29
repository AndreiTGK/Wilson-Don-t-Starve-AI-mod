"""
death_memory.py
───────────────
Wilson's death memory bank. The seed of deja vu.

Every time Wilson dies, his final hidden state and the circumstances
of his death are written here. On each subsequent life, the current
hidden state is compared against this bank. The similarity score
is the deja vu signal — fed back into Wilson's perception without
any label or explanation.

He doesn't know why certain situations feel familiar.
He just feels it. And over many deaths, that feeling becomes dread.
"""

import math
import json
from pathlib import Path

MAX_MEMORIES = 50


class DeathMemory:
    def __init__(self):
        self.memories: list[dict] = []
        # Each memory:
        # {
        #   "death_id":   int,
        #   "hidden":     list[float],   ← Wilson's hidden state at death
        #   "emotion":    list[float],   ← emotion vector at death
        #   "obs_slice":  list[float],   ← self stats at death
        #   "day":        int,
        #   "cause":      str,           ← inferred from state (darkness, hunger, etc)
        # }

    def record(self, hidden: list, emotion: list, obs: list, state: dict):
        """Imprint a death into the memory bank."""
        s = state.get("self", {})
        w = state.get("world", {})

        # Infer cause of death from state at time of death
        hp_pct  = s.get("hp", 0)    / max(s.get("hp_max", 150), 1)
        hun_pct = s.get("hunger", 0) / max(s.get("hunger_max", 150), 1)
        san_pct = s.get("sanity", 0) / max(s.get("sanity_max", 200), 1)
        sounds  = state.get("sounds", {})

        if sounds.get("darkness"):
            cause = "darkness"
        elif hun_pct < 0.05:
            cause = "starvation"
        elif san_pct < 0.1:
            cause = "insanity"
        elif sounds.get("hound_attack") or sounds.get("damage_taken"):
            cause = "creature"
        else:
            cause = "unknown"

        memory = {
            "death_id": len(self.memories) + 1,
            "hidden":   list(hidden),
            "emotion":  list(emotion),
            "obs_slice":[hp_pct, hun_pct, san_pct],
            "day":      w.get("day", 0),
            "phase":    w.get("phase", "?"),
            "cause":    cause,
        }

        if len(self.memories) >= MAX_MEMORIES:
            self.memories.pop(0)
        self.memories.append(memory)

        print(f"[DeathMemory] Death #{memory['death_id']} imprinted. "
              f"Day {memory['day']+1}, cause: {cause}. "
              f"Bank size: {len(self.memories)}")

    def deja_vu_signal(self, hidden: list) -> float:
        """
        Compare current hidden state against all death memories.
        Returns a float in [0, 1] — how strongly this moment
        resembles a past death.

        This is the deja vu signal. Wilson doesn't know why it fires.
        """
        if not self.memories:
            return 0.0

        max_sim = 0.0
        for mem in self.memories:
            sim = cosine_similarity(hidden, mem["hidden"])
            if sim > max_sim:
                max_sim = sim

        return max_sim

    def weighted_deja_vu(self, hidden: list) -> float:
        """
        Like deja_vu_signal but weights recent deaths more heavily.
        The most recent deaths are the freshest wounds.
        """
        if not self.memories:
            return 0.0

        total_weight = 0.0
        weighted_sim = 0.0
        n = len(self.memories)

        for i, mem in enumerate(self.memories):
            weight = (i + 1) / n       # recent memories weighted more
            sim    = cosine_similarity(hidden, mem["hidden"])
            weighted_sim  += weight * sim
            total_weight  += weight

        return weighted_sim / total_weight if total_weight > 0 else 0.0

    def most_similar_death(self, hidden: list) -> dict | None:
        """Return the death memory most similar to the current state."""
        if not self.memories:
            return None
        return max(
            self.memories,
            key=lambda m: cosine_similarity(hidden, m["hidden"])
        )

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.memories, f, indent=2)

    def load(self, path: str):
        p = Path(path)
        if p.exists() and p.stat().st_size > 0:
            with open(path) as f:
                self.memories = json.load(f)
            print(f"[DeathMemory] Loaded {len(self.memories)} memories from {path}")

    def __len__(self):
        return len(self.memories)

    def summary(self) -> str:
        if not self.memories:
            return "no deaths yet"
        causes = {}
        for m in self.memories:
            c = m.get("cause", "unknown")
            causes[c] = causes.get(c, 0) + 1
        parts = [f"{c}×{n}" for c, n in sorted(causes.items(), key=lambda x: -x[1])]
        return f"{len(self.memories)} deaths — {', '.join(parts)}"


def cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two vectors. Returns 0 if either is zero."""
    if len(a) != len(b):
        return 0.0
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


if __name__ == "__main__":
    dm = DeathMemory()

    # Fake a few deaths
    import random
    for i in range(5):
        hidden  = [random.gauss(0, 1) for _ in range(128)]
        emotion = [random.gauss(0, 1) for _ in range(16)]
        obs     = [random.random() for _ in range(3)]
        state   = {
            "self":  {"hp": 0, "hp_max": 150, "hunger": 20, "hunger_max": 150,
                      "sanity": 50, "sanity_max": 200},
            "world": {"day": i * 3, "phase": "night"},
            "sounds": {"darkness": 1 if i % 2 == 0 else 0},
        }
        dm.record(hidden, emotion, obs, state)

    # Test deja vu
    current_hidden = [random.gauss(0, 1) for _ in range(128)]
    signal = dm.deja_vu_signal(current_hidden)
    print(f"\nDeja vu signal: {signal:.4f}")
    print(f"Summary: {dm.summary()}")
