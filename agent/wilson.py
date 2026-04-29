"""
wilson.py
─────────
Wilson. The agent. Built from nothing.

This is the top-level class that ties together:
  - The neural network (network.py)
  - The observation encoder (observation.py)
  - The death memory / deja vu system (death_memory.py)
  - The thought logger (thought_logger.py)
  - The action space (actions.py)

One hardcoded truth: death = -1000.
Everything else is learned.
"""

import json
import math
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.network       import WilsonNet
from agent.observation   import encode_state, OBS_DIM, RECIPE_PROGRESS_DIM, SPATIAL_DIM, DEJA_VU_IDX, SPATIAL_START, RECIPE_START
from agent.actions       import Action, N_ACTIONS, ACTION_LABELS
from agent.item_knowledge import get_purpose, ItemValueTracker
from memory.death_memory  import DeathMemory
from memory.thought_logger import ThoughtLogger
from language.vocabulary  import VOCAB_SIZE

def _parse_inv_item(item) -> tuple:
    """Returns (prefab, count). Handles both old string and new dict inventory formats."""
    if isinstance(item, dict):
        return item.get("p", "") or "", item.get("n", 1) or 1
    if isinstance(item, str) and item:
        parts = item.split(":")
        return parts[0], int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
    return "", 0


_FOOD_PREFABS  = {"berries", "berrybush", "berrybush2", "carrot", "carrot_planted", "mushroom", "pumpkin", "watermelon"}
_WOOD_PREFABS  = {"tree", "evergreen", "deciduoustree", "palmtree", "sapling", "birchnutt", "marsh_tree"}
_STONE_PREFABS = {"rocks", "rock1", "rock2", "flint", "boulder", "rock_flintless"}
_FIRE_PREFABS  = {"campfire", "firepit", "endothermic_fire", "endothermic_firepit", "walrus_campfire"}
_DAY1_PRIORITY = {"twigs", "cutgrass"}

# ── Reward shaping ────────────────────────────────────────────────────────────
DEATH_REWARD    = -1000.0
SURVIVAL_REWARD =    -0.1   # per-step bonus just for staying alive
HOARDER_BONUS   =    1.0   # picking up at least one new item this step
BUMP_PENALTY              =   -0.3
DAMAGE_SCALE              =   25.0
PHOTON_DEPRIVATION_PENALTY=   -2.0
PAIN_PENALTY              =   -1.0  # flat hit any time Wilson takes damage
EXPLORE_REWARD            =    0.3  # scales down with sqrt(visit_count) per cell
NEW_ENTITY_REWARD         =    2.0  # scales down with sqrt(encounter_count) per prefab
PURPOSE_DISCOVERY_BONUS   =    8.0  # first item of a new purpose category


class Wilson:
    def __init__(self,
                 save_dir: str = ".",
                 hidden_dim: int = 128,
                 emotion_dim: int = 16,
                 learning_rate: float = 1e-3):

        self.save_dir   = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.hidden_dim  = hidden_dim
        self.emotion_dim = emotion_dim

        # ── Core network ──
        self.net = WilsonNet(
            obs_dim     = OBS_DIM,
            hidden_dim  = hidden_dim,
            emotion_dim = emotion_dim,
            n_actions   = N_ACTIONS,
            vocab_size  = VOCAB_SIZE,
        )

        # ── Memory systems ──
        self.death_memory  = DeathMemory()
        self.thought_log   = ThoughtLogger(
            log_path=str(self.save_dir / "wilson_thoughts.jsonl")
        )

        # ── Previous-step tracking (for reward shaping) ──
        self.previous_x_z: tuple | None = None
        self.previous_inventory_count: int = 0
        self.previous_hp_pct: float = 1.0

        self.discovered_item_types: set = set()
        self.consecutive_idle_steps = 0
        self._last_deja_vu = 0.0

        self.explored_cells: set = set()
        self.seen_entity_types: set = set()
        self._pending_curiosity_reward: float = 0.0

        # ── Curiosity & item knowledge ──
        self.item_tracker = ItemValueTracker()
        self.discovered_purposes: set = set()
        self.entity_encounter_count: dict = {}
        self.cell_visit_count: dict = {}
        self.previous_inv_prefabs: set = set()

        # ── Recipe database ──
        recipe_path = Path(__file__).parent / "recipe_book.json"
        if recipe_path.exists():
            with open(recipe_path) as f:
                self.recipes: dict = json.load(f)
        else:
            self.recipes: dict = {}
        sorted_names = sorted(self.recipes.keys())
        self._recipe_name_to_id = {
            name: i / max(len(sorted_names), 1)
            for i, name in enumerate(sorted_names)
        }
        self.ever_crafted: set = set()
        self._pending_mastery_reward: float = 0.0

        # ── Global spatial memory ──
        self.world_map = {"food": set(), "wood": set(), "stone": set()}

        # ── Training state ──
        self.lr            = learning_rate
        self.step          = 0
        self.episode       = 0
        self.total_deaths  = 0
        self.alive         = True

        # PPO rollout buffer
        self.rollout = []          # list of (obs, action, reward, value, log_prob)
        self.gamma   = 0.99        # discount
        self.lam     = 0.95        # GAE lambda

        # ── Perceptor (optional; set externally by pipeline) ──
        self.perceptor = None

        # ── Load previous life if it exists ──
        self._load_if_exists()

    # ── Crafting progress ─────────────────────────────────────────────────────

    def _researchlab_nearby(self, perceived: list) -> bool:
        return any(
            ent.get("prefab") == "researchlab" and ent.get("dist", 999) <= 40
            for ent in perceived
        )

    def get_craftable_progress(self, current_inventory: list, perceived: list = None) -> list:
        counts: dict = {}
        for item in current_inventory:
            prefab, count = _parse_inv_item(item)
            if prefab:
                counts[prefab] = counts.get(prefab, 0) + count

        lab_nearby = self._researchlab_nearby(perceived or [])

        result = []
        for name, recipe in self.recipes.items():
            if recipe.get("tech", 0) >= 1 and not lab_nearby:
                result.append((name, 0.0))
                continue
            ingredients = recipe.get("ingredients", {})
            if not ingredients:
                result.append((name, 0.0))
                continue
            total_required = sum(ingredients.values())
            total_have = sum(
                min(counts.get(ing, 0), needed)
                for ing, needed in ingredients.items()
            )
            result.append((name, total_have / total_required))
        return result

    # ── Perception ────────────────────────────────────────────────────────────

    def perceive(self, state: dict) -> list:
        obs = encode_state(state)

        px = state["self"].get("x", 0.0)
        pz = state["self"].get("z", 0.0)
        current_deaths = state.get("deaths", 0)

        # --- Manual Temporal Decay Logic ---
        deja_vu = 0.0
        for death in self.death_memory.memories:
            dx = death.get("x", 0.0)
            dz = death.get("z", 0.0)
            # Use the record's ID to calculate age
            death_id = death.get("death_id", 0)
            dist = math.sqrt((px - dx)**2 + (pz - dz)**2)
            if dist < 20.0:
                # Clamp age to at least 1 to prevent division by zero
                age = max(1, current_deaths - death_id + 1)
                age_factor = 1.0 / age
                deja_vu += (1.0 - (dist / 20.0)) * age_factor

        deja_vu = min(deja_vu, 1.0)
        obs[DEJA_VU_IDX] = deja_vu
        self._last_deja_vu = deja_vu

        # Map entities
        entities = list(state.get("perceived", [])) + list(state.get("remembered", []))
        for ent in entities:
            prefab = ent.get("prefab", "")
            dist   = ent.get("dist", 0.0)
            angle_rad = ent.get("angle", 0) * math.pi / 180.0
            ex = round((px + dist * math.cos(angle_rad)) / 5) * 5
            ez = round((pz + dist * math.sin(angle_rad)) / 5) * 5
            if prefab in _FOOD_PREFABS: self.world_map["food"].add((ex, ez))
            elif prefab in _WOOD_PREFABS: self.world_map["wood"].add((ex, ez))
            elif prefab in _STONE_PREFABS: self.world_map["stone"].add((ex, ez))

        # Spatial vectors with 2.0 unit Arrival Buffer
        spatial = []
        for category in ("food", "wood", "stone"):
            coords = self.world_map[category]
            if coords:
                best_dist = min(math.sqrt((px - cx) ** 2 + (pz - cz) ** 2) for cx, cz in coords)
                best_coord = min(coords, key=lambda c: (px - c[0]) ** 2 + (pz - c[1]) ** 2)
                # Buffer: If close enough, distance is effectively zero
                dist_val = max(0.0, best_dist - 2.0)
                dist_norm  = min(dist_val, 100.0) / 100.0
                angle_norm = math.atan2(best_coord[1] - pz, best_coord[0] - px) / math.pi
            else:
                dist_norm  = 1.0
                angle_norm = 0.0
            spatial += [dist_norm, angle_norm]

        obs[SPATIAL_START : SPATIAL_START + SPATIAL_DIM] = spatial

        inv = state.get("inventory", [])
        perceived_entities = list(state.get("perceived", []))
        inv_prefabs = {_parse_inv_item(item)[0] for item in inv if _parse_inv_item(item)[0]}
        lab_nearby = self._researchlab_nearby(perceived_entities)
        for name, recipe in self.recipes.items():
            if recipe.get("product", "") in inv_prefabs and name not in self.ever_crafted:
                if recipe.get("tech", 0) == 0 or lab_nearby:
                    self.ever_crafted.add(name)
                    self._pending_mastery_reward += 10.0

        for ent in perceived_entities:
            prefab = ent.get("prefab", "")
            if prefab:
                count = self.entity_encounter_count.get(prefab, 0) + 1
                self.entity_encounter_count[prefab] = count
                if count == 1:
                    self.seen_entity_types.add(prefab)
                bonus = NEW_ENTITY_REWARD / math.sqrt(count)
                if bonus >= 0.05:
                    self._pending_curiosity_reward += bonus

        progress = self.get_craftable_progress(inv, perceived_entities)
        unbuilt = [(name, score) for name, score in progress if name not in self.ever_crafted]
        top5 = sorted(unbuilt, key=lambda x: x[1], reverse=True)[:5]

        recipe_obs: list = []
        for name, score in top5:
            recipe_obs += [self._recipe_name_to_id.get(name, 0.0), score]
        recipe_obs += [0.0, 0.0] * (5 - len(top5))

        obs[RECIPE_START : RECIPE_START + RECIPE_PROGRESS_DIM] = recipe_obs

        if self.perceptor is not None:
            obs = self.perceptor.transform(obs)
        return obs

    # ── Decision ──────────────────────────────────────────────────────────────

    def decide(self, state: dict) -> dict:
        obs = self.perceive(state)
        out = self.net.forward(obs)

        action_idx = self._sample_action(out["action_probs"])
        action     = Action(action_idx)

        cur_x         = state["self"]["x"]
        cur_z         = state["self"]["z"]
        hp_max        = state["self"]["hp_max"] or 1
        cur_hp_pct    = state["self"]["hp"] / hp_max
        cur_inv_count = len(state.get("inventory", []))

        # --- THE DEATH FAIL-SAFE ---
        if cur_hp_pct <= 0.0 and self.previous_hp_pct > 0.0:
            print("[Wilson] FAILSAFE: HP is 0. Triggering death manually.")
            self.on_death(state)
            return {
                "action": Action.STOP,
                "action_label": "STOP",
                "emotion": out["emotion"],
                "deja_vu": self._last_deja_vu,
                "thought": "Terminal state reached; neuro-reset triggered.",
                "value": out["value"]
            }
        # ---------------------------

        phase   = state.get("world", {}).get("phase", "day")
        day     = state.get("world", {}).get("day", 1)
        is_day1 = (day == 1)

        inv_prefabs_list = [_parse_inv_item(item)[0] for item in state.get("inventory", []) if _parse_inv_item(item)[0]]
        torch_equipped   = "torch" in inv_prefabs_list

        is_safe_from_darkness = torch_equipped or any(
            ent.get("prefab") in _FIRE_PREFABS and ent.get("dist", 999) <= 20
            for ent in state.get("perceived", [])
        )

        env_penalty = 0.0
        if phase == "dusk":
            env_penalty = -0.1
        if phase == "night":
            env_penalty = -2.0
        if is_safe_from_darkness:
            env_penalty = 0.0
        if torch_equipped:
            env_penalty -= 0.02

        deja_vu_clamped = max(0.0, min(self._last_deja_vu, 1.0))
        deja_vu_mag     = deja_vu_clamped * 2.0
        damage_mag      = max((self.previous_hp_pct - cur_hp_pct) * DAMAGE_SCALE, 0.0)
        photon_mag      = abs(env_penalty)

        move_dist = 0.0
        if self.previous_x_z is not None:
            move_dist = math.sqrt((cur_x - self.previous_x_z[0]) ** 2 + (cur_z - self.previous_x_z[1]) ** 2)

        if self.previous_x_z is not None and move_dist < 0.2:
            self.consecutive_idle_steps += 1
        else:
            self.consecutive_idle_steps = 0

        idle_mag = self.consecutive_idle_steps * 0.005

        bump_occurred = False
        if action_idx <= 7 and self.previous_x_z is not None:
            if move_dist < 0.1:
                bump_occurred = True
        bump_mag = abs(BUMP_PENALTY) if bump_occurred else 0.0

        dominant_penalty = None
        if idle_mag > max(deja_vu_mag, damage_mag, photon_mag, bump_mag) and self.consecutive_idle_steps > 50:
            dominant_penalty = "idle"
        elif deja_vu_mag > 0 and deja_vu_mag >= max(damage_mag, photon_mag, bump_mag):
            dominant_penalty = "deja_vu"

        thought = self.thought_log.maybe_think(
            self.net, state, self.total_deaths, dominant_penalty
        )

        reward = SURVIVAL_REWARD + self._pending_mastery_reward + self._pending_curiosity_reward
        self._pending_mastery_reward = 0.0
        self._pending_curiosity_reward = 0.0

        cell = (int(cur_x / 20), int(cur_z / 20))
        cell_visits = self.cell_visit_count.get(cell, 0) + 1
        self.cell_visit_count[cell] = cell_visits
        if cell_visits == 1:
            self.explored_cells.add(cell)
        cell_bonus = EXPLORE_REWARD / math.sqrt(cell_visits)
        if cell_bonus >= 0.01:
            reward += cell_bonus

        newly_acquired = [p for p in inv_prefabs_list if p not in self.previous_inv_prefabs]
        if cur_inv_count > self.previous_inventory_count and newly_acquired:
            max_utility = max((self.item_tracker.get_utility(p) for p in newly_acquired), default=0.0)
            utility_scale = 1.0 + max(max_utility, 0.0) * 0.5
            if is_day1 and any(p in _DAY1_PRIORITY for p in newly_acquired):
                reward += HOARDER_BONUS * 3 * utility_scale
            else:
                reward += HOARDER_BONUS * utility_scale

        for item in state.get("inventory", []):
            prefab, _ = _parse_inv_item(item)
            if not prefab or prefab in self.discovered_item_types:
                self.discovered_item_types.add(prefab)
                base = 5.0 * (3 if (is_day1 and prefab in _DAY1_PRIORITY) else 1)
                reward += base
                purpose = get_purpose(prefab)
                if purpose != "unknown" and purpose not in self.discovered_purposes:
                    self.discovered_purposes.add(purpose)
                    reward += PURPOSE_DISCOVERY_BONUS

        mult = 1.5 if self.consecutive_idle_steps > 50 else 1.0

        if bump_occurred:
            reward += BUMP_PENALTY * mult

        if cur_hp_pct < self.previous_hp_pct:
            reward += PAIN_PENALTY
            reward -= damage_mag * mult

        if deja_vu_clamped > 0:
            reward -= min(deja_vu_mag, 0.5) * mult

        if env_penalty < 0:
            reward += env_penalty * mult

        reward -= idle_mag

        log_prob = self._log_prob(out["action_probs"], action_idx)
        self.rollout.append({
            "obs":      obs,
            "action":   action_idx,
            "reward":   reward,
            "value":    out["value"],
            "log_prob": log_prob,
        })

        self.item_tracker.update(inv_prefabs_list, reward)

        self.previous_x_z             = (cur_x, cur_z)
        self.previous_inventory_count = cur_inv_count
        self.previous_hp_pct          = cur_hp_pct
        self.previous_inv_prefabs     = set(inv_prefabs_list)

        self.step += 1

        return {
            "action":      action,
            "action_label":ACTION_LABELS[action],
            "emotion":     out["emotion"],
            "deja_vu":     self._last_deja_vu,
            "thought":     thought,
            "value":       out["value"],
            "reward":      reward,
        }

    # ── Death ─────────────────────────────────────────────────────────────────

    def on_death(self, state: dict):
        """
        Called when Wilson dies.
        The only hardcoded punishment. Everything else is learned.
        """
        self.total_deaths += 1
        self.episode      += 1
        self.alive         = False

        # Imprint this death into memory
        out = self.net.forward(self.perceive(state))
        self.death_memory.record(
            hidden  = self.net.h,
            emotion = out["emotion"],
            obs     = self.perceive(state),
            state   = state,
        )

        # Apply death reward to last rollout step
        if self.rollout:
            self.rollout[-1]["reward"] = DEATH_REWARD

        # Train on this episode's rollout
        self._train_on_rollout()

        # Reset for next life
        self.net.reset_hidden()
        self.rollout = []
        self.alive   = True
        self.step    = 0
        self.previous_x_z              = None
        self.previous_inventory_count  = 0
        self.previous_hp_pct           = 1.0
        self.previous_inv_prefabs      = set()
        self._pending_mastery_reward   = 0.0
        self._pending_curiosity_reward = 0.0

        # Save after each death
        self._save()

        print(f"[Wilson] Episode {self.episode} ended. "
              f"Steps: {self.step}. "
              f"Total deaths: {self.total_deaths}. "
              f"{self.death_memory.summary()}")

    # ── Training (PPO-lite) ───────────────────────────────────────────────────

    def _train_on_rollout(self):
        """
        Simplified PPO update on the collected rollout.
        Pure Python — no autograd. Uses finite differences for gradients.
        For real training speed, replace with numpy/torch version.
        """
        if len(self.rollout) < 2:
            return

        # Compute returns and advantages (GAE)
        returns    = self._compute_returns()
        advantages = self._compute_advantages(returns)

        # Simple policy gradient step (REINFORCE with baseline)
        # Full PPO clipping omitted for v1 — can be added once torch is integrated
        # This nudges weights in the right direction even without autograd
        self._nudge_weights(advantages)

    def _compute_returns(self) -> list:
        returns = []
        R = 0.0
        for step in reversed(self.rollout):
            R = step["reward"] + self.gamma * R
            returns.insert(0, R)
        return returns

    def _compute_advantages(self, returns: list) -> list:
        values = [s["value"] for s in self.rollout]
        advantages = [r - v for r, v in zip(returns, values)]
        # Normalise
        if len(advantages) > 1:
            mean = sum(advantages) / len(advantages)
            std  = max((sum((a-mean)**2 for a in advantages)/len(advantages))**0.5, 1e-8)
            advantages = [(a - mean) / std for a in advantages]
        return advantages

    def _nudge_weights(self, advantages: list):
        """
        Finite-difference weight nudge. Placeholder for proper backprop.
        Replace with torch autograd for real training.
        """
        import random
        # Nudge a random subset of policy head weights proportional to advantage
        mean_adv = sum(advantages) / max(len(advantages), 1)
        scale    = self.lr * mean_adv * 0.01

        policy_W = self.net.policy_head.W
        for i in range(min(50, len(policy_W.data))):
            idx = random.randrange(len(policy_W.data))
            policy_W.data[idx] += scale * random.gauss(0, 1)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sample_action(self, probs: list) -> int:
        import random
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                return i
        return len(probs) - 1

    def _log_prob(self, probs: list, action: int) -> float:
        return math.log(max(probs[action], 1e-10))

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        self.net.save(str(self.save_dir / "wilson_net.pkl"))
        self.death_memory.save(str(self.save_dir / "death_memory.json"))
        cell_visit_serialized = {
            f"{k[0]},{k[1]}": v for k, v in self.cell_visit_count.items()
        }
        meta = {
            "step":                 self.step,
            "episode":              self.episode,
            "total_deaths":         self.total_deaths,
            "ever_crafted":         list(self.ever_crafted),
            "item_knowledge":       self.item_tracker.to_dict(),
            "entity_encounter_count": self.entity_encounter_count,
            "cell_visit_count":     cell_visit_serialized,
            "discovered_purposes":  list(self.discovered_purposes),
        }
        with open(self.save_dir / "wilson_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def _load_if_exists(self):
        net_path  = self.save_dir / "wilson_net.pkl"
        mem_path  = self.save_dir / "death_memory.json"
        meta_path = self.save_dir / "wilson_meta.json"

        if net_path.exists():
            try:
                self.net.load(str(net_path))
                expected = self.net.obs_dim * self.net.hidden_dim
                if len(self.net.enc1.W.data) != expected:
                    raise ValueError(f"enc1 shape mismatch: got {len(self.net.enc1.W.data)}, expected {expected}")
            except Exception as e:
                print(f"[Wilson] Incompatible checkpoint ({e}); starting fresh.")
                self.net = WilsonNet(
                    obs_dim     = OBS_DIM,
                    hidden_dim  = self.hidden_dim,
                    emotion_dim = self.emotion_dim,
                    n_actions   = N_ACTIONS,
                    vocab_size  = VOCAB_SIZE,
                )
        if mem_path.exists():
            self.death_memory.load(str(mem_path))
        if meta_path.exists() and meta_path.stat().st_size > 0:
            with open(meta_path) as f:
                meta = json.load(f)
            self.step         = meta.get("step", 0)
            self.episode      = meta.get("episode", 0)
            self.total_deaths = meta.get("total_deaths", 0)
            self.ever_crafted = set(meta.get("ever_crafted", []))

            if "item_knowledge" in meta:
                self.item_tracker.from_dict(meta["item_knowledge"])
            self.entity_encounter_count = meta.get("entity_encounter_count", {})
            self.seen_entity_types = set(self.entity_encounter_count.keys())
            raw_cells = meta.get("cell_visit_count", {})
            self.cell_visit_count = {
                tuple(int(x) for x in k.split(",")): v
                for k, v in raw_cells.items()
            }
            self.explored_cells = set(self.cell_visit_count.keys())
            self.discovered_purposes = set(meta.get("discovered_purposes", []))

            print(f"[Wilson] Resumed from episode {self.episode}, "
                  f"{self.total_deaths} deaths. "
                  f"Knows {len(self.item_tracker.utility)} items, "
                  f"{len(self.discovered_purposes)} purposes discovered.")

    def status(self) -> str:
        top = self.item_tracker.top_items(3)
        top_str = ", ".join(f"{p}={v:.2f}" for p, v in top) if top else "none"
        return (f"Episode {self.episode} | Step {self.step} | "
                f"Deaths {self.total_deaths} | "
                f"Deja vu {self._last_deja_vu:.3f} | "
                f"Purposes {len(self.discovered_purposes)}/7 | "
                f"Top items: [{top_str}] | "
                f"{self.death_memory.summary()}")


if __name__ == "__main__":
    wilson = Wilson(save_dir="/tmp/wilson_test")

    # Simulate a short life
    fake_state = {
        "self":  {"hp": 100, "hp_max": 150, "hunger": 80, "hunger_max": 150,
                  "sanity": 180, "sanity_max": 200, "wetness": 0,
                  "x": 0.0, "z": 0.0},
        "world": {"day": 1, "phase": "day", "season": "autumn", "is_raining": False},
        "inventory": [],
        "perceived": [{"prefab":"tree","dist":5.0,"angle":90,"hp_pct":None,"radius":"near"}],
        "remembered": [],
        "sounds": {k: 0 for k in ["hound_near","hound_attack","low_hp","low_sanity",
                                   "low_hunger","dusk","dawn","rain","thunder",
                                   "boss_near","darkness","damage_taken"]},
        "deaths": 0,
    }

    print("Simulating a short life...")
    for i in range(5):
        decision = wilson.decide(fake_state)
        print(f"  Step {i+1}: {decision['action_label']:20s} "
              f"value={decision['value']:.3f} "
              f"deja_vu={decision['deja_vu']:.3f}")
        if decision["thought"]:
            print(f"  Thought: {decision['thought']}")

    print("\nSimulating death...")
    wilson.on_death(fake_state)
    print(f"\nStatus: {wilson.status()}")
