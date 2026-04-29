"""
thought_logger.py
─────────────────
Wilson's internal monologue. Generated entirely from his own network.
No external LLM. No API calls. No borrowed mind.

The thought decoder takes Wilson's emotion vector and maps it to words
from his own vocabulary. Early in training the words are nearly random —
fragments, wrong turns, noise. As Wilson learns to survive, the emotion
vector becomes meaningful, and the words it produces start to reflect
what he's actually experiencing.

"""

import json
import time
from pathlib import Path
from collections import deque

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from language.vocabulary import (
    decode_sequence, VOCAB_SIZE,
    START_IDX, END_IDX, UNK_IDX, PAD_IDX,
    WORD_TO_IDX,
    SENSORY_SET, STATE_SET, DIRECTIVE_SET,
)

# ── Thought log ───────────────────────────────────────────────────────────────

MAX_LOG_SIZE = 200   # entries kept in memory
THOUGHT_EVERY = 20   # generate a thought every N steps

class ThoughtLogger:
    def __init__(self, log_path: str = "wilson_thoughts.jsonl"):
        self.log_path   = Path(log_path)
        self.log        = deque(maxlen=MAX_LOG_SIZE)
        self.step_count = 0
        self.last_thought_step = -THOUGHT_EVERY

        # Ensure log file exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def maybe_think(self, net, state: dict, death_count: int, dominant_penalty: str = None, reward: float = 0.0) -> str | None:
        """
        Maybe generate a thought this step.
        Returns the thought string if one was generated, else None.
        """
        self.step_count += 1
        if self.step_count - self.last_thought_step < THOUGHT_EVERY:
            return None

        self.last_thought_step = self.step_count

        # Get emotion vector from network's last forward pass
        # (net.forward() must have been called already this step)
        # We re-run thought decoding from the cached emotion
        obs_dummy = [0.0] * net.obs_dim
        out = net.forward(obs_dummy)   # uses cached hidden state
        emotion = out["emotion"]
        thought_logits = out["thought_logits"]

        # Decode thought from logits — top words by confidence
        word_indices = self._decode_thought(thought_logits, emotion, net, death_count)
        words = decode_sequence(word_indices, strip_special=True)

        # Filter noise — remove <unk> and very low-confidence tokens
        words = [w for w in words if w != "<unk>" and w != "..."][:6]

        if not words:
            thought_str = "..."
        else:
            thought_str = self._format_thought(words, state, death_count, dominant_penalty)

        # Build log entry
        w = state.get("world", {})
        s = state.get("self", {})
        entry = {
            "step":    self.step_count,
            "day":      w.get("day", 0) + 1,
            "phase":    w.get("phase", "?"),
            "deaths":   death_count,
            "hp_pct":   round(s.get("hp", 150) / max(s.get("hp_max", 150), 1), 2),
            "emotion": [round(v, 3) for v in emotion[:4]],  # first 4 dims only
            "thought": thought_str,
            "reward":  round(reward, 3), # Added reward for dashboard polling
            "ts":      time.time(),
        }

        self.log.append(entry)
        self._write_entry(entry)

        return thought_str

    def _decode_thought(self, logits: list, emotion: list,
                        net, death_count: int) -> list:
        """
        Turn logits into word indices.
        Uses temperature sampling — hotter early on, cooler as Wilson matures.
        """
        import math, random

        # Temperature: starts high (chaotic), lowers as Wilson survives longer
        # More deaths → more formed thoughts, lower temperature
        temperature = max(0.5, 2.0 - death_count * 0.1)

        # Apply temperature
        scaled = [v / temperature for v in logits]
        m = max(scaled)
        exps = [math.exp(v - m) for v in scaled]
        s = sum(exps)
        probs = [e / s for e in exps]

        # Sample a few tokens
        n_tokens = min(3 + death_count // 5, 7)   # more words over time
        tokens = []
        for _ in range(n_tokens):
            r = random.random()
            cumulative = 0.0
            for i, p in enumerate(probs):
                cumulative += p
                if r < cumulative:
                    if i not in (PAD_IDX, START_IDX, END_IDX):
                        tokens.append(i)
                    break

        return tokens

    def _format_thought(self, words: list, state: dict, death_count: int, dominant_penalty: str = None) -> str:
        if dominant_penalty == "deja_vu":
            return "Experiencing memory_resonance_stigma; initializing spatial_recalibration."

        sensory   = [w for w in words if w in SENSORY_SET]
        state_w   = [w for w in words if w in STATE_SET]
        directive = [w for w in words if w in DIRECTIVE_SET]

        if state_w and sensory and directive:
            return f"Experiencing {state_w[0]} amidst {sensory[0]}; initializing {directive[0]}."
        elif sensory and directive:
            return f"Detecting {sensory[0]}, proceeding with {directive[0]}."
        elif state_w:
            return f"Monitoring severe {state_w[0]}..."
        elif sensory:
            return f"Observing {sensory[0]}..."
        elif directive:
            return f"Initiating {directive[0]}..."
        else:
            return " ".join(words) + "."

    def _write_entry(self, entry: dict):
        """Append thought to the log file."""
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def recent(self, n: int = 5) -> list[dict]:
        """Return the N most recent thoughts."""
        entries = list(self.log)
        return entries[-n:]

    def print_recent(self, n: int = 5):
        for entry in self.recent(n):
            phase  = entry.get("phase","?")
            day    = entry.get("day", 0)
            deaths = entry.get("deaths", 0)
            thought= entry.get("thought","...")
            print(f"  Day {day} · {phase} · deaths:{deaths}")
            print(f"  {thought}")
            print()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    from agent.network import WilsonNet
    from agent.observation import OBS_DIM
    from agent.actions import N_ACTIONS
    from language.vocabulary import VOCAB_SIZE

    net = WilsonNet(OBS_DIM, hidden_dim=128, emotion_dim=16,
                    n_actions=N_ACTIONS, vocab_size=VOCAB_SIZE)

    logger = ThoughtLogger(log_path="/tmp/wilson_thoughts.jsonl")

    fake_state = {
        "self":  {"hp": 60, "hp_max": 150, "hunger": 30, "hunger_max": 150,
                  "sanity": 100, "sanity_max": 200},
        "world": {"day": 4, "phase": "night", "season": "autumn", "is_raining": False},
        "sounds": {"darkness": 1, "low_hunger": 1},
        "perceived": [],
        "inventory": ["torch"],
        "death_count": 2,
        "death_memory": [],
    }

    # Simulate a few steps
    from agent.observation import encode_state
    obs = encode_state(fake_state)
    obs += [0.0] * (net.obs_dim - len(obs))
    net.forward(obs)

    for i in range(3):
        logger.step_count = i * THOUGHT_EVERY
        logger.last_thought_step = -THOUGHT_EVERY
        thought = logger.maybe_think(net, fake_state, death_count=2, reward=-0.1)
        if thought:
            print(f"Thought: {thought}")
