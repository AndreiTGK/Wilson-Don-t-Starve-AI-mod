"""
pipeline.py
───────────
Headless agent runner for Wilson. Reads state from the Lua mod,
calls the agent, and writes the action back — no terminal UI.

Run with --run to start the loop, or import and call run() directly.
Running without --run performs a smoke test (init only).

Cannot run simultaneously with wilson_watch.py — both write to the
same action file.
"""

import json
import sys
import time
from pathlib import Path

MOD_DIR    = Path(__file__).parent
STATE_FILE = MOD_DIR / "wilson_ai_state"
ACTION_FILE= MOD_DIR / "wilson_ai_action.txt"

sys.path.insert(0, str(MOD_DIR))

from agent.perceptor      import Perceptor
from agent.wilson         import Wilson
from agent.observation    import encode_state
from memory.passive_memory import PassiveMemory

SAVE_INTERVAL = 500   # steps between periodic saves


def _read_state() -> dict | None:
    """Return parsed state dict, or None if file is missing / mid-write / unreadable."""
    try:
        if not STATE_FILE.exists():
            return None
        raw = STATE_FILE.read_text(encoding="utf-8")
        if not raw.endswith("--END--"):
            return None  # Lua write still in progress
        raw = raw[:raw.rfind("\n--END--")].strip()
        if not raw:
            return None
        return json.loads(raw)
    except (json.JSONDecodeError, OSError):
        return None


def _write_action(action_idx: int):
    ACTION_FILE.write_text(str(int(action_idx)))


def run():
    """Main headless loop. Runs until interrupted."""
    perceptor = Perceptor(save_path=str(MOD_DIR / "perceptor.json"))
    wilson    = Wilson(save_dir=str(MOD_DIR))
    wilson.perceptor = perceptor

    passive   = PassiveMemory(path=str(MOD_DIR / "passive_memory.jsonl"))

    last_death_count = 0
    last_state_mtime = 0.0
    steps_since_save = 0

    print(f"[Pipeline] Started. State: {STATE_FILE}  Action: {ACTION_FILE}")
    print(f"[Pipeline] Passive memory: {len(passive)} entries on disk")

    try:
        while True:
            try:
                mtime = STATE_FILE.stat().st_mtime
            except OSError:
                time.sleep(0.05)
                continue

            if mtime == last_state_mtime:
                time.sleep(0.05)
                continue

            state = _read_state()
            if state is None:
                time.sleep(0.05)
                continue

            last_state_mtime = mtime

            death_count = state.get("death_count", 0)
            is_death    = death_count > last_death_count

            if is_death:
                wilson.on_death(state)
                raw_obs = encode_state(state)
                reward  = -1000.0
                passive.record(state, action=-1, reward=reward, done=True)
                perceptor.update(raw_obs, reward)
                perceptor.save()
                last_death_count = death_count
                steps_since_save = 0
                continue

            decision   = wilson.decide(state)
            action_idx = int(decision["action"])
            _write_action(action_idx)

            # Pull the reward wilson just computed and stored in rollout
            raw_obs = encode_state(state)
            reward  = wilson.rollout[-1]["reward"] if wilson.rollout else 0.0

            perceptor.update(raw_obs, reward)
            passive.record(state, action=action_idx, reward=reward, done=False)

            steps_since_save += 1
            if steps_since_save >= SAVE_INTERVAL:
                perceptor.save()
                steps_since_save = 0
                print(f"[Pipeline] Step {wilson.step} | "
                      f"Top features: {perceptor.top_features(5)}")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[Pipeline] Interrupted — saving...")
        perceptor.save()
        print("[Pipeline] Done.")


if __name__ == "__main__":
    if "--run" in sys.argv:
        run()
    else:
        # Smoke test: just initialise everything and verify wiring
        print("── Pipeline smoke test ──────────────────────────────────────")
        perceptor = Perceptor(save_path=str(MOD_DIR / "perceptor.json"))
        wilson    = Wilson(save_dir=str(MOD_DIR))
        wilson.perceptor = perceptor

        assert wilson.perceptor is perceptor, "perceptor not wired into wilson"
        print("  wilson.perceptor wired correctly  ✓")

        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            tmp = tf.name
        try:
            passive = PassiveMemory(path=tmp)
            assert len(passive) == 0
            passive.record({}, action=0, reward=0.1, done=False)
            assert len(passive) == 1
            print("  PassiveMemory record/len  ✓")
        finally:
            os.unlink(tmp)

        print("\nPipeline OK  (pass --run to start the agent loop)")
