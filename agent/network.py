"""
network.py
──────────
Wilson's brain. Built from scratch. No pretrained weights.

Architecture:
  Observation vector
      ↓
  Encoder MLP  (obs → hidden state)
      ↓
  GRU cell     (hidden state persists across ticks — Wilson has memory)
      ↓
  ┌───────────────────────────┐
  │  Policy head (→ actions)  │
  │  Value head  (→ V(s))     │
  │  Emotion head(→ emotion   │
  │               vector)     │
  └───────────────────────────┘

The emotion vector is a learned latent space that emerges entirely
from training. It is never labeled or supervised. What it represents
is whatever turned out to be useful for survival.

The thought decoder is a separate small network that reads the
emotion vector and produces word indices from Wilson's own vocabulary.
It trains alongside everything else.
"""

import math
import struct
import os

# ── Pure Python tensor (no numpy/torch dependency) ───────────────────────────
# Wilson's network runs in pure Python for maximum portability.
# For actual training speed, swap this with numpy or torch — the
# interface is identical.

class Tensor:
    """Minimal flat float tensor. Row-major."""
    def __init__(self, data: list, shape: tuple):
        self.data  = list(data)
        self.shape = shape

    @classmethod
    def zeros(cls, *shape):
        size = 1
        for d in shape: size *= d
        return cls([0.0] * size, shape)

    @classmethod
    def randn(cls, *shape, scale=0.01):
        import random
        size = 1
        for d in shape: size *= d
        data = [random.gauss(0, scale) for _ in range(size)]
        return cls(data, shape)

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]
    def __setitem__(self, i, v): self.data[i] = v

    def row(self, i, cols):
        return self.data[i*cols:(i+1)*cols]


def relu(x: float) -> float:
    return max(0.0, x)

def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)

def tanh_(x: float) -> float:
    return math.tanh(x)

def softmax(vec: list) -> list:
    m = max(vec)
    exps = [math.exp(v - m) for v in vec]
    s = sum(exps)
    return [e / s for e in exps]

def matmul_vec(W: Tensor, x: list, rows: int, cols: int) -> list:
    """y = W @ x  where W is (rows x cols) stored row-major."""
    out = []
    for i in range(rows):
        acc = 0.0
        base = i * cols
        for j in range(cols):
            acc += W.data[base + j] * x[j]
        out.append(acc)
    return out

def add_vec(a: list, b: list) -> list:
    return [x + y for x, y in zip(a, b)]

def mul_vec(a: list, b: list) -> list:
    return [x * y for x, y in zip(a, b)]


# ── Linear layer ──────────────────────────────────────────────────────────────

class Linear:
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        self.in_dim  = in_dim
        self.out_dim = out_dim
        scale = math.sqrt(2.0 / in_dim)
        self.W = Tensor.randn(out_dim, in_dim, scale=scale)
        self.b = [0.0] * out_dim if bias else None
        # Gradient accumulators
        self.dW = Tensor.zeros(out_dim, in_dim)
        self.db = [0.0] * out_dim if bias else None

    def forward(self, x: list) -> list:
        out = matmul_vec(self.W, x, self.out_dim, self.in_dim)
        if self.b:
            out = add_vec(out, self.b)
        return out

    def parameters(self):
        params = [self.W]
        if self.b is not None:
            params.append(self.b)
        return params


# ── GRU Cell ──────────────────────────────────────────────────────────────────

class GRUCell:
    """
    Single GRU cell. Wilson's short-term memory lives here.
    Hidden state persists between ticks within a single life.
    Resets on death — he starts each life fresh, but carries
    the death memory bank forward.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        # Reset gate
        self.Wr = Linear(input_dim + hidden_dim, hidden_dim, bias=True)
        # Update gate
        self.Wz = Linear(input_dim + hidden_dim, hidden_dim, bias=True)
        # New gate
        self.Wn = Linear(input_dim + hidden_dim, hidden_dim, bias=True)

    def forward(self, x: list, h: list) -> list:
        xh = x + h
        r  = [sigmoid(v) for v in self.Wr.forward(xh)]
        z  = [sigmoid(v) for v in self.Wz.forward(xh)]
        xrh = x + mul_vec(r, h)
        n  = [tanh_(v) for v in self.Wn.forward(xrh)]
        # h' = (1-z)*n + z*h
        h_new = [
            (1.0 - z[i]) * n[i] + z[i] * h[i]
            for i in range(self.hidden_dim)
        ]
        return h_new


# ── Wilson's full network ─────────────────────────────────────────────────────

class WilsonNet:
    """
    Wilson's complete neural architecture.

    Parameters are intentionally small — he needs to be able to
    run on modest hardware and train in reasonable time.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128,
                 emotion_dim: int = 16, n_actions: int = 16,
                 vocab_size: int = 200):

        self.obs_dim     = obs_dim
        self.hidden_dim  = hidden_dim
        self.emotion_dim = emotion_dim
        self.n_actions   = n_actions
        self.vocab_size  = vocab_size

        # ── Encoder: obs → pre-hidden ──
        self.enc1 = Linear(obs_dim,     hidden_dim)
        self.enc2 = Linear(hidden_dim,  hidden_dim)

        # ── Recurrent memory ──
        self.gru  = GRUCell(hidden_dim, hidden_dim)

        # ── Emotion head: hidden → emotion vector ──
        # This is the latent emotional state. Never supervised.
        self.emotion_head = Linear(hidden_dim, emotion_dim)

        # ── Policy head: hidden → action logits ──
        self.policy_head  = Linear(hidden_dim, n_actions)

        # ── Value head: hidden → scalar V(s) ──
        self.value_head   = Linear(hidden_dim, 1)

        # ── Thought decoder: emotion → word sequence ──
        # Trained alongside everything else.
        # Maps emotion vector to a short sequence of word indices.
        self.thought_dec1 = Linear(emotion_dim, emotion_dim * 2)
        self.thought_dec2 = Linear(emotion_dim * 2, vocab_size)

        # ── Hidden state (persists within a life) ──
        self.h = [0.0] * hidden_dim

    def reset_hidden(self):
        """Call on death — clears short-term memory but keeps weights."""
        self.h = [0.0] * self.hidden_dim

    def forward(self, obs: list) -> dict:
        """
        Forward pass. Returns a dict with:
          action_probs  — probability over actions
          value         — estimated state value
          emotion       — latent emotion vector
          thought_logits— logits over vocabulary (for thought decoding)
          hidden        — current hidden state (for inspection)
        """
        # Encode observation
        x = [relu(v) for v in self.enc1.forward(obs)]
        x = [relu(v) for v in self.enc2.forward(x)]

        # Update recurrent state
        self.h = self.gru.forward(x, self.h)

        # Emotion (unsupervised latent)
        emotion = [tanh_(v) for v in self.emotion_head.forward(self.h)]

        # Policy
        logits = self.policy_head.forward(self.h)
        action_probs = softmax(logits)

        # Value
        value = self.value_head.forward(self.h)[0]

        # Thought (decoded from emotion, not from hidden directly)
        t1 = [relu(v) for v in self.thought_dec1.forward(emotion)]
        thought_logits = self.thought_dec2.forward(t1)

        return {
            "action_probs":   action_probs,
            "value":          value,
            "emotion":        emotion,
            "thought_logits": thought_logits,
            "hidden":         list(self.h),
        }

    def select_action(self, obs: list, greedy: bool = False) -> int:
        """Sample or argmax an action from the policy."""
        import random
        out = self.forward(obs)
        probs = out["action_probs"]
        if greedy:
            return probs.index(max(probs))
        # Weighted sample
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                return i
        return len(probs) - 1

    def decode_thought(self, emotion: list, max_len: int = 8) -> list:
        """
        Greedily decode a thought from the emotion vector.
        Returns a list of word indices.
        Stops at <end> token or max_len.
        """
        from language.vocabulary import START_IDX, END_IDX, VOCAB_SIZE
        t1 = [relu(v) for v in self.thought_dec1.forward(emotion)]
        logits = self.thought_dec2.forward(t1)
        # For now: top-k sample from logits (not autoregressive)
        # A true decoder would feed back previous token — kept simple for v1
        probs = softmax(logits)
        indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        # Return top words above a threshold, stop at max_len
        thought = []
        for idx in indices[:max_len]:
            thought.append(idx)
        return thought

    def save(self, path: str):
        """Save all parameters to a binary file."""
        import pickle
        params = self._collect_params()
        with open(path, "wb") as f:
            pickle.dump(params, f)
        print(f"[WilsonNet] Saved to {path}")

    def load(self, path: str):
        """Load parameters from a binary file."""
        import pickle
        with open(path, "rb") as f:
            params = pickle.load(f)
        self._apply_params(params)
        print(f"[WilsonNet] Loaded from {path}")

    def _collect_params(self) -> dict:
        layers = [
            "enc1","enc2","emotion_head",
            "policy_head","value_head",
            "thought_dec1","thought_dec2",
        ]
        gru_layers = ["Wr","Wz","Wn"]
        params = {}
        for name in layers:
            layer = getattr(self, name)
            params[f"{name}.W"] = list(layer.W.data)
            if layer.b:
                params[f"{name}.b"] = list(layer.b)
        for name in gru_layers:
            layer = getattr(self.gru, name)
            params[f"gru.{name}.W"] = list(layer.W.data)
            if layer.b:
                params[f"gru.{name}.b"] = list(layer.b)
        params["hidden"] = list(self.h)
        return params

    def _apply_params(self, params: dict):
        layers = [
            "enc1","enc2","emotion_head",
            "policy_head","value_head",
            "thought_dec1","thought_dec2",
        ]
        gru_layers = ["Wr","Wz","Wn"]
        for name in layers:
            layer = getattr(self, name)
            layer.W.data = params[f"{name}.W"]
            if layer.b and f"{name}.b" in params:
                layer.b = params[f"{name}.b"]
        for name in gru_layers:
            layer = getattr(self.gru, name)
            layer.W.data = params[f"gru.{name}.W"]
            if layer.b and f"gru.{name}.b" in params:
                layer.b = params[f"gru.{name}.b"]
        if "hidden" in params:
            self.h = params["hidden"]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from agent.observation import OBS_DIM
    from agent.actions import N_ACTIONS
    from language.vocabulary import VOCAB_SIZE

    net = WilsonNet(
        obs_dim=OBS_DIM,
        hidden_dim=128,
        emotion_dim=16,
        n_actions=N_ACTIONS,
        vocab_size=VOCAB_SIZE,
    )

    # Fake obs
    obs = [0.0] * OBS_DIM
    out = net.forward(obs)

    print(f"WilsonNet initialised")
    print(f"  obs_dim:     {OBS_DIM}")
    print(f"  hidden_dim:  128")
    print(f"  emotion_dim: 16")
    print(f"  n_actions:   {N_ACTIONS}")
    print(f"  vocab_size:  {VOCAB_SIZE}")
    print(f"  action_probs (sum): {sum(out['action_probs']):.4f}")
    print(f"  value:       {out['value']:.4f}")
    print(f"  emotion vec: {[f'{v:.3f}' for v in out['emotion']]}")
