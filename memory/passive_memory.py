"""
passive_memory.py
─────────────────
Append-only permanent log of every (state, action, reward, done) tuple
across all of Wilson's runs.

Written as JSONL — one JSON object per line — so it survives crashes
and can be streamed or replayed without loading the whole file.

Full raw state is stored so offline replay / batch training is possible.
"""

import json
import os
from pathlib import Path


class PassiveMemory:
    """
    Append-only JSONL log.

    record(state, action, reward, done)  — write one experience tuple
    tail(n)                              — last n entries (no full-file load)
    __len__                              — cached entry count
    """

    def __init__(self, path: str):
        self.path  = Path(path)
        self._count = self._count_lines()

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, state: dict, action: int, reward: float, done: bool):
        """Append one experience. Full raw state stored for replay."""
        entry = {
            "state":  state,
            "action": action,
            "reward": reward,
            "done":   bool(done),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self._count += 1

    def tail(self, n: int) -> list:
        """Return the last n entries without loading the full file."""
        if n <= 0:
            return []
        lines = []
        with open(self.path, "rb") as f:
            f.seek(0, 2)
            pos   = f.tell()
            buf   = b""
            found = 0
            while pos > 0 and found < n + 1:
                chunk = min(4096, pos)
                pos  -= chunk
                f.seek(pos)
                buf   = f.read(chunk) + buf
                found = buf.count(b"\n")
            parts = buf.split(b"\n")

        result = []
        for part in parts:
            part = part.strip()
            if part:
                try:
                    result.append(json.loads(part))
                except json.JSONDecodeError:
                    pass
        return result[-n:]

    def __len__(self) -> int:
        return self._count

    # ── Internal ──────────────────────────────────────────────────────────────

    def _count_lines(self) -> int:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return 0
        count = 0
        with open(self.path, "rb") as f:
            for _ in f:
                count += 1
        return count


if __name__ == "__main__":
    import os, tempfile

    print("── PassiveMemory smoke tests ────────────────────────────────")

    fake_state = {
        "self":  {"hp": 100, "hp_max": 150, "hunger": 80, "hunger_max": 150,
                  "sanity": 180, "sanity_max": 200, "wetness": 0,
                  "x": 10.0, "z": -5.0},
        "world": {"day": 2, "phase": "day", "season": "autumn",
                  "is_raining": False},
        "inventory": ["axe", "", "log"],
        "perceived":  [],
        "remembered": [],
        "sounds":     {},
        "death_count": 0,
    }

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        tmp = tf.name
    try:
        pm = PassiveMemory(tmp)
        assert len(pm) == 0, "expected 0 entries on fresh file"

        pm.record(fake_state, action=0, reward=0.5, done=False)
        pm.record(fake_state, action=8, reward=-1000.0, done=True)

        assert len(pm) == 2, f"expected 2 entries, got {len(pm)}"
        print(f"  len() after 2 records: {len(pm)}  ✓")

        entries = pm.tail(2)
        assert len(entries) == 2, f"tail(2) returned {len(entries)}"
        assert entries[-1]["done"]   is True,     "last entry should be done=True"
        assert entries[-1]["reward"] == -1000.0,  "last reward mismatch"
        assert entries[0]["action"]  == 0,        "first action mismatch"
        assert "state" in entries[0],             "full state not stored"
        assert entries[0]["state"]["self"]["hp"] == 100
        print(f"  tail(2) content:  actions={[e['action'] for e in entries]}  ✓")
        print(f"  full state round-trip: hp={entries[0]['state']['self']['hp']}  ✓")

        # Reload from disk — count should persist
        pm2 = PassiveMemory(tmp)
        assert len(pm2) == 2, f"reload count mismatch: {len(pm2)}"
        print(f"  reload from disk: len={len(pm2)}  ✓")

        # tail(1) returns only the last entry
        last = pm.tail(1)
        assert len(last) == 1 and last[0]["done"] is True
        print(f"  tail(1): done={last[0]['done']}  ✓")

    finally:
        os.unlink(tmp)

    print("\nPassiveMemory OK")
