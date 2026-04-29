"""
perceptor.py
────────────
Sits between raw JSON state and Wilson's neural predictor.
Assigns meaning to raw observations without hardcoded rules.

Learns which features matter through two unsupervised mechanisms:
  1. Online normalization (Welford's algorithm) — adapts to actual
     value distributions observed in play; no assumed range.
  2. Feature importance weights — updated via EMA of |z-score × reward|,
     so features that deviate from baseline toward positive outcomes rise
     in importance over time.

Nothing is assumed important upfront. Importance emerges from experience.
After enough runs, the perceptor's top_features() will reveal what Wilson
has learned to pay attention to.
"""

import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.observation import encode_state, OBS_DIM


class Perceptor:
    """
    Learnable observation transformer.

    process(state)       → (raw_obs, perceptor_obs)  — full pipeline from JSON
    transform(raw_obs)   → perceptor_obs              — pure transform, no side effects
    update(raw_obs, r)   → updates stats and importance

    Wilson.perceive() calls transform() after encoding, so the Perceptor's
    output is what actually reaches the neural network.
    """

    def __init__(self, obs_dim: int = OBS_DIM, alpha: float = 0.005,
                 save_path: str = None):
        self.obs_dim   = obs_dim
        self.alpha     = alpha       # EMA learning rate for importance
        self.save_path = Path(save_path) if save_path else None

        # Welford online statistics — no assumed distribution
        self.n    = 0
        self.mean = [0.0] * obs_dim
        self._M2  = [0.0] * obs_dim  # sum of squared deviations

        # Feature importance — uniform start; shaped purely by reward correlation
        self.importance = [1.0] * obs_dim
        self._ema_corr  = [0.0] * obs_dim

        if self.save_path and self.save_path.exists():
            self._load(str(self.save_path))

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, state: dict) -> tuple:
        """
        Full pipeline: raw JSON state → (raw_obs, perceptor_obs).
        raw_obs is the pre-transform encoding, needed for update().
        """
        raw_obs = encode_state(state)
        return raw_obs, self.transform(raw_obs)

    def transform(self, raw_obs: list) -> list:
        """
        Normalize and apply importance weighting.
        Pure — no side effects. Called by Wilson.perceive().
        """
        norm_obs = self._normalize(raw_obs)
        return self._apply_importance(norm_obs)

    def update(self, raw_obs: list, reward: float):
        """Update running stats and importance from one (obs, reward) pair."""
        self._welford_update(raw_obs)
        self._importance_update(raw_obs, reward)

    def top_features(self, n: int = 10) -> list:
        """Return (index, importance) pairs for the n most important features."""
        indexed = sorted(enumerate(self.importance), key=lambda x: -x[1])
        return [(i, round(v, 4)) for i, v in indexed[:n]]

    # ── Normalization ─────────────────────────────────────────────────────────

    def _welford_update(self, obs: list):
        self.n += 1
        for i, x in enumerate(obs):
            d            = x - self.mean[i]
            self.mean[i] += d / self.n
            self._M2[i]  += d * (x - self.mean[i])

    def _normalize(self, obs: list) -> list:
        if self.n < 2:
            return list(obs)
        result = []
        for i, x in enumerate(obs):
            var = self._M2[i] / max(self.n - 1, 1)
            std = max(math.sqrt(var), 1e-8)
            result.append((x - self.mean[i]) / std)
        return result

    # ── Importance ────────────────────────────────────────────────────────────

    def _importance_update(self, raw_obs: list, reward: float):
        if self.n < 2:
            return
        ema_decay = 1.0 - self.alpha
        for i, x in enumerate(raw_obs):
            var = self._M2[i] / max(self.n - 1, 1)
            std = max(math.sqrt(var), 1e-8)
            z   = (x - self.mean[i]) / std
            self._ema_corr[i] = (ema_decay * self._ema_corr[i]
                                 + self.alpha * abs(z * reward))

        max_c = max(self._ema_corr) if any(v > 0 for v in self._ema_corr) else 1.0
        for i in range(self.obs_dim):
            self.importance[i] = 0.1 + 0.9 * (self._ema_corr[i] / max_c)

    def _apply_importance(self, norm_obs: list) -> list:
        w_sum = max(sum(self.importance), 1e-8)
        scale = self.obs_dim / w_sum
        return [norm_obs[i] * self.importance[i] * scale
                for i in range(self.obs_dim)]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = None):
        target = path or (str(self.save_path) if self.save_path else None)
        if not target:
            return
        data = {
            "n":          self.n,
            "mean":       self.mean,
            "_M2":        self._M2,
            "importance": self.importance,
            "_ema_corr":  self._ema_corr,
        }
        with open(target, "w") as f:
            json.dump(data, f)

    def _load(self, path: str):
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return
        with open(path) as f:
            d = json.load(f)
        saved_mean = d.get("mean", [])
        if saved_mean and len(saved_mean) != self.obs_dim:
            print(f"[Perceptor] Saved obs_dim={len(saved_mean)} != current {self.obs_dim}; starting fresh.")
            return
        self.n          = d.get("n",          0)
        self.mean       = d.get("mean",       [0.0] * self.obs_dim)
        self._M2        = d.get("_M2",        [0.0] * self.obs_dim)
        self.importance = d.get("importance", [1.0] * self.obs_dim)
        self._ema_corr  = d.get("_ema_corr",  [0.0] * self.obs_dim)
        print(f"[Perceptor] Loaded from {path} (n={self.n} samples)")


if __name__ == "__main__":
    import random, os, tempfile

    print("── Perceptor smoke tests ─────────────────────────────────────")

    # Test 1: basic process on a real fake state
    fake_state = {
        "self":  {"hp": 100, "hp_max": 150, "hunger": 80, "hunger_max": 150,
                  "sanity": 180, "sanity_max": 200, "wetness": 0,
                  "x": 10.0, "z": -5.0},
        "world": {"day": 2, "phase": "day", "season": "autumn",
                  "is_raining": False},
        "inventory": ["axe", "", "log", "", "", "", "", "", "", ""],
        "perceived":  [{"prefab": "evergreen", "dist": 8.0,
                        "angle": 45, "hp_pct": None, "radius": "near"}],
        "remembered": [],
        "sounds":     {k: 0 for k in ["hound_near", "hound_attack", "low_hp",
                                       "low_sanity", "low_hunger", "dusk",
                                       "dawn", "rain", "thunder",
                                       "boss_near", "darkness", "damage_taken"]},
        "death_count": 0,
        "death_memory": [],
        "craft_slots": [],
        "pending_placement": {"active": 0, "product": None},
        "tiles": [0, 0, 0, 0, 0],
        "border": [0.5, 0.5],
        "speed": 1,
        "tick": 1,
    }

    p = Perceptor()
    raw_obs, perceptor_obs = p.process(fake_state)
    assert len(perceptor_obs) == OBS_DIM, (
        f"Expected {OBS_DIM}, got {len(perceptor_obs)}")
    print(f"  process() output dim: {len(perceptor_obs)}  ✓")

    # Test 2: importance learning — dim 0 correlates strongly with reward
    p2 = Perceptor(obs_dim=10, alpha=0.02)
    for _ in range(400):
        obs = [random.gauss(0, 0.05) for _ in range(10)]
        obs[0] = random.gauss(0, 1.5)          # dim 0 has high variance
        reward = obs[0] * 5.0 + random.gauss(0, 0.2)  # reward ≈ dim 0
        p2.update(obs, reward)

    top_idx = max(range(10), key=lambda i: p2.importance[i])
    assert top_idx == 0, f"Expected dim 0 most important, got dim {top_idx}"
    print(f"  importance learning: dim 0 correctly surfaced as most important  ✓")
    print(f"  top 3: {p2.top_features(3)}")

    # Test 3: save / load round-trip
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp = tf.name
    try:
        p2.save(tmp)
        p3 = Perceptor(obs_dim=10, save_path=tmp)
        assert p3.n == p2.n
        assert abs(p3.importance[0] - p2.importance[0]) < 1e-9
        print(f"  save/load round-trip: n={p3.n}  ✓")
    finally:
        os.unlink(tmp)

    print("\nPerceptor OK")
