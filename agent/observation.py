"""
observation.py
──────────────
Converts the raw JSON state from the Lua mod into a flat tensor
that Wilson's neural net can process.

Everything is normalised to [0, 1] or encoded as a small integer.
No pretrained features. Just numbers derived directly from what Wilson perceives.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from language.vocabulary import encode, VOCAB_SIZE

# ── Observation layout ────────────────────────────────────────────────────────
#
# [0]      hp_pct           0–1
# [1]      hunger_pct       0–1
# [2]      sanity_pct       0–1
# [3]      wetness_pct      0–1
# [4]      temperature_norm 0–1  ((temp+20)/90, range [-20,70])
# [5]      phase_pct        0–1  (time elapsed in current phase)
# [6]      phase_day        0 or 1
# [7]      phase_dusk       0 or 1
# [8]      phase_night      0 or 1
# [9]      season_autumn    0 or 1
# [10]     season_winter    0 or 1
# [11]     season_spring    0 or 1
# [12]     season_summer    0 or 1
# [13]     is_raining       0 or 1
# [14]     day_normalised   0–1  (day / 100, capped)
#
# [15 .. 15+INV_DIM]  Explicit Inventory Slots (N_INV_SLOTS × N_INV_FEATS)
#   Maps 1-to-1 with INV_1 through INV_10 actions (4 features each):
#     [vocab_idx_norm, count_norm, dur_pct, fresh_pct]
#
# [15+INV_DIM .. 15+INV_DIM+N_ENTITIES*ENTITY_DIM]
#   Up to N_ENTITIES nearby entities, each encoded as:
#     [sin(angle/180π), cos(angle/180π), dist_norm, hp_pct, is_memory, vocab_idx_norm]
#
# [tail: SOUND_DIM]  sound pulses, one per sound event (0 or 1)
# [tail+SOUND_DIM]   deja_vu_strength  0–1  (death_count / 50, capped)
#
# After craft/pending/spatial:
# [BORDER_START]     wx_norm  0–1  (x position within world, 0=west edge, 1=east edge)
# [BORDER_START+1]   wz_norm  0–1  (z position within world, 0=south edge, 1=north edge)

N_ENTITIES  = 12          # max entities to encode
ENTITY_DIM  = 6           # features per entity
SOUND_NAMES = [
    "hound_near", "hound_attack", "low_hp", "low_sanity",
    "low_hunger", "dusk", "dawn", "rain", "thunder",
    "boss_near", "darkness", "damage_taken",
]
SOUND_DIM   = len(SOUND_NAMES)

N_INV_SLOTS    = 10  # Explicitly matches actions.py INV_1 to INV_10
N_INV_FEATS    = 4   # vocab_idx_norm, count_norm, dur_pct, fresh_pct
N_CRAFT_SLOTS  = 8
CRAFT_SLOT_DIM = 4   # available, product_vidx, ingredients_met, is_placeable
CRAFT_TOTAL    = N_CRAFT_SLOTS * CRAFT_SLOT_DIM
PENDING_DIM    = 2   # has_pending, pending_product_vidx
SPATIAL_DIM    = 6
BORDER_DIM     = 2   # wx_norm, wz_norm — Wilson's normalised position in world [0,1]
RECIPE_PROGRESS_DIM = 10  # 5 recipes × (recipe_id_norm, progress_score)

SELF_DIM    = 15
INV_DIM     = N_INV_SLOTS * N_INV_FEATS
ENTITY_TOTAL= N_ENTITIES * ENTITY_DIM
OBS_DIM     = SELF_DIM + INV_DIM + ENTITY_TOTAL + SOUND_DIM + 1 + CRAFT_TOTAL + PENDING_DIM + SPATIAL_DIM + BORDER_DIM + RECIPE_PROGRESS_DIM

DEJA_VU_IDX   = SELF_DIM + INV_DIM + ENTITY_TOTAL + SOUND_DIM
SPATIAL_START = OBS_DIM - RECIPE_PROGRESS_DIM - BORDER_DIM - SPATIAL_DIM
BORDER_START  = OBS_DIM - RECIPE_PROGRESS_DIM - BORDER_DIM
RECIPE_START  = OBS_DIM - RECIPE_PROGRESS_DIM


def _phase_vec(phase: str) -> list:
    return [
        1.0 if phase == "day"   else 0.0,
        1.0 if phase == "dusk"  else 0.0,
        1.0 if phase == "night" else 0.0,
    ]

def _season_vec(season: str) -> list:
    return [
        1.0 if season == "autumn" else 0.0,
        1.0 if season == "winter" else 0.0,
        1.0 if season == "spring" else 0.0,
        1.0 if season == "summer" else 0.0,
    ]

def _encode_entity(ent: dict) -> list:
    angle   = ent.get("angle", 0) * math.pi / 180.0
    dist    = min(ent.get("dist", 40), 40) / 40.0
    hp      = ent.get("hp_pct") if ent.get("hp_pct") is not None else 1.0
    is_mem  = 1.0 if ent.get("radius") == "memory" else 0.0
    prefab  = ent.get("prefab", "<unk>")
    vidx    = encode(prefab) / max(VOCAB_SIZE - 1, 1)
    return [
        math.sin(angle),
        math.cos(angle),
        dist,
        float(hp),
        is_mem,
        vidx,
    ]


def _encode_craft_slots(craft_slots: list, pending: dict) -> list:
    vec = []
    for i in range(N_CRAFT_SLOTS):
        if i < len(craft_slots):
            slot         = craft_slots[i]
            available    = float(slot.get("available", 0))
            product_vidx = encode(slot.get("product", "<unk>")) / max(VOCAB_SIZE - 1, 1)
            ingr_met     = float(slot.get("ingredients_met", 0.0))
            is_placeable = float(slot.get("is_placeable", 0))
        else:
            available, product_vidx, ingr_met, is_placeable = 0.0, 0.0, 0.0, 0.0
        vec += [available, product_vidx, ingr_met, is_placeable]

    has_pending  = float((pending or {}).get("active", 0))
    pending_vidx = encode((pending or {}).get("product") or "<unk>") / max(VOCAB_SIZE - 1, 1)
    vec += [has_pending, pending_vidx]
    return vec


def encode_state(state: dict) -> list:
    """
    Convert a raw state dict from the Lua mod into a flat float list
    of length OBS_DIM. Ready to be turned into a tensor.
    """
    s  = state.get("self", {})
    w  = state.get("world", {})
    inv= state.get("inventory", [])
    perceived    = state.get("perceived", [])
    remembered   = state.get("remembered", [])
    sounds       = state.get("sounds", {})
    deaths       = state.get("death_count", 0)
    craft_slots  = state.get("craft_slots", [])
    pending      = state.get("pending_placement", {}) or {}

    # ── Self ──
    hp_pct   = s.get("hp", 150)    / max(s.get("hp_max", 150), 1)
    hun_pct  = s.get("hunger", 150) / max(s.get("hunger_max", 150), 1)
    san_pct  = s.get("sanity", 200) / max(s.get("sanity_max", 200), 1)
    wet_pct  = min(s.get("wetness", 0), 100) / 100.0
    temp_raw = s.get("temperature", 35)
    temp_norm = max(0.0, min((temp_raw + 20) / 90.0, 1.0))
    phase_pct = float(w.get("phase_pct", 0.5))
    day_n    = min(w.get("day", 0), 100) / 100.0
    rain     = 1.0 if w.get("is_raining") else 0.0

    obs = (
        [hp_pct, hun_pct, san_pct, wet_pct, temp_norm, phase_pct]
        + _phase_vec(w.get("phase", "day"))
        + _season_vec(w.get("season", "autumn"))
        + [rain, day_n]
    )
    # Should be SELF_DIM = 15 at this point

    # ── Explicit Slot Encoding (4 features per slot) ──
    inv_vec = []
    for i in range(N_INV_SLOTS):
        if i < len(inv):
            slot = inv[i]
            if isinstance(slot, dict):
                prefab    = slot.get("p", "") or ""
                count     = slot.get("n", 0) or 0
                dur_pct   = slot.get("d") if slot.get("d") is not None else 0.0
                fresh_pct = slot.get("f") if slot.get("f") is not None else 0.0
            elif isinstance(slot, str) and slot:
                prefab, count = slot, 1
                dur_pct, fresh_pct = 0.0, 0.0
            else:
                prefab, count, dur_pct, fresh_pct = "", 0, 0.0, 0.0
        else:
            prefab, count, dur_pct, fresh_pct = "", 0, 0.0, 0.0

        vidx = encode(prefab) / max(VOCAB_SIZE - 1, 1) if prefab else 0.0
        inv_vec += [vidx, min(count, 40) / 40.0, float(dur_pct), float(fresh_pct)]

    obs += inv_vec

    # ── Nearby entities ──
    all_ents = list(perceived) + list(remembered)
    all_ents = sorted(all_ents, key=lambda e: e.get("dist", 999))[:N_ENTITIES]
    entity_vec = []
    for ent in all_ents:
        entity_vec += _encode_entity(ent)
    # Pad to N_ENTITIES * ENTITY_DIM
    pad_needed = ENTITY_TOTAL - len(entity_vec)
    entity_vec += [0.0] * pad_needed
    obs += entity_vec

    # ── Sound pulses ──
    for snd in SOUND_NAMES:
        obs.append(float(sounds.get(snd, 0)))

    # ── Deja vu strength ──
    deja_vu = min(deaths, 50) / 50.0
    obs.append(deja_vu)

    # ── Craft slots + pending placement ──
    obs += _encode_craft_slots(craft_slots, pending)

    # ── Tile hazard sensing (current + N/+x/S/-x cardinal neighbors) ──
    raw_tiles  = state.get("tiles", [])
    spatial_vec = [float(t) for t in raw_tiles[:SPATIAL_DIM]]
    while len(spatial_vec) < SPATIAL_DIM:
        spatial_vec.append(0.0)

    # ── World border position ──
    raw_border = state.get("border", [])
    border_vec = [
        float(raw_border[0]) if len(raw_border) > 0 else 0.5,
        float(raw_border[1]) if len(raw_border) > 1 else 0.5,
    ]

    recipe_padding = [0.0] * RECIPE_PROGRESS_DIM

    obs += spatial_vec
    obs += border_vec
    obs += recipe_padding

    assert len(obs) == OBS_DIM, f"OBS_DIM mismatch: got {len(obs)}, expected {OBS_DIM}"
    return obs


if __name__ == "__main__":
    # Smoke test with a fake state (new dict inventory format)
    fake_state = {
        "self":  {"hp": 100, "hp_max": 150, "hunger": 50, "hunger_max": 150,
                  "sanity": 180, "sanity_max": 200, "wetness": 10,
                  "temperature": 20},
        "world": {"day": 3, "phase": "dusk", "season": "autumn",
                  "is_raining": False, "phase_pct": 0.3},
        "inventory": [
            {"p": "axe",     "n": 1,  "d": 0.8,  "f": None},
            {"p": "log",     "n": 5,  "d": None,  "f": None},
            {"p": "berries", "n": 1,  "d": None,  "f": 0.9},
            {"p": "",        "n": 0,  "d": None,  "f": None},
            {"p": "",        "n": 0,  "d": None,  "f": None},
            {"p": "flint",   "n": 3,  "d": None,  "f": None},
        ],
        "perceived": [
            {"prefab": "spider", "dist": 8.0, "angle": 45, "hp_pct": 1.0, "radius": "near"},
            {"prefab": "tree",   "dist": 3.0, "angle": 180,"hp_pct": None,"radius": "near"},
        ],
        "remembered": [],
        "sounds": {"hound_near": 0, "low_hp": 0, "dawn": 0, "dusk": 1},
        "death_count": 2,
        "death_memory": [],
    }
    obs = encode_state(fake_state)
    print(f"Observation vector length: {len(obs)}  (expected {OBS_DIM})")
    print(f"Self slice:   {obs[:SELF_DIM]}")
    print(f"Inv slice:    {obs[SELF_DIM:SELF_DIM+INV_DIM]}")
    print(f"Deja vu:      {obs[DEJA_VU_IDX]}")
