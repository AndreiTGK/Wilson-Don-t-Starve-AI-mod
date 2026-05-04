# Wilson AI Observer — Don't Starve / DST Mod

A Don't Starve (Together) mod that turns Wilson into an AI-driven agent. The Lua mod exports Wilson's in-game perception to disk, and a Python pipeline reads that state, makes decisions, and writes actions back — creating a feedback loop between the game and an external AI agent.

No omniscience. Wilson only knows what he can observe locally, the same way a real player would.

---

## How It Works

```
[Don't Starve / DST]
   Lua mod (modmain.lua)
        │
        │  writes wilson_ai_state (JSON + --END-- sentinel)
        ▼
[Python Pipeline]
   pipeline.py / wilson_watch.py
        │
        ├─ Perceptor   — encodes raw game state into features
        ├─ Wilson      — decision-making agent
        └─ PassiveMemory — logs (state, action, reward) to disk
        │
        │  writes wilson_ai_action.txt (action index)
        ▼
[Don't Starve / DST]
   Lua mod reads action and executes it
```

The Lua side is purely observational and makes no gameplay changes on its own — it only watches and reports. The Python side drives all decision-making.

---

## Features

- **Observational Lua mod** — exports Wilson's local perception (health, hunger, sanity, nearby entities, inventory, etc.) without giving the AI any god-mode knowledge
- **Headless Python agent** (`pipeline.py`) — runs in the background, no terminal UI required
- **Passive memory** — every `(state, action, reward)` tuple is logged to a `.jsonl` file for offline analysis or training
- **Death handling** — the agent detects death events and applies a large negative reward, persisting what it learned from the run
- **Perceptor** — tracks and ranks the most informative game state features over time
- **Interactive watcher** (`wilson_watch.py`) — alternative runner for active monitoring sessions
- **Compatible with** Don't Starve, Don't Starve Together, Reign of Giants, Shipwrecked, and Hamlet

---

## Project Structure

```
Wilson-Don-t-Starve-AI-mod/
├── agent/
│   ├── wilson.py           # Core decision-making agent
│   ├── perceptor.py        # State encoding & feature tracking
│   └── observation.py      # encode_state() — raw state → feature vector
├── language/               # Natural language / dialogue layer
├── memory/
│   └── passive_memory.py   # Records (state, action, reward) to .jsonl
├── utils/                  # Helper utilities
├── modmain.lua             # Lua mod — hooks into the game, exports state
├── modinfo.lua             # Mod metadata
├── pipeline.py             # Headless agent runner (recommended)
├── wilson_watch.py         # Interactive watcher (alternative runner)
├── death_memory.json       # Persisted death events
└── __init__.py
```

---

## Installation

### 1. Install the Lua mod

Copy the entire repository folder into your Don't Starve mods directory:

**Don't Starve Together:**
```
# Windows
C:\Program Files (x86)\Steam\steamapps\common\Don't Starve Together\mods\

# Linux
~/.steam/steam/steamapps/common/Don't Starve Together/mods/
```

Enable the **Wilson AI Observer** mod from the in-game mod menu before starting a world.

### 2. Install Python dependencies

Python 3.10+ is required.

```bash
cd Wilson-Don-t-Starve-AI-mod
pip install -r requirements.txt  # if present, otherwise the agent uses stdlib + json only
```

---

## Usage

### Headless mode (recommended)

Run the pipeline before or after launching the game:

```bash
python pipeline.py --run
```

The agent will wait for the game to write a state file, then begin making decisions automatically. Stop it at any time with `Ctrl+C` — it saves state on exit.

### Smoke test (no game required)

Run without `--run` to verify everything is wired correctly:

```bash
python pipeline.py
```

### Interactive watcher

```bash
python wilson_watch.py
```

> **Note:** `pipeline.py` and `wilson_watch.py` cannot run at the same time — both write to the same action file.

---

## State & Action Files

| File | Written by | Purpose |
|------|-----------|---------|
| `wilson_ai_state` | Lua mod | Current game state as JSON, terminated with `--END--` |
| `wilson_ai_action.txt` | Python pipeline | Integer action index for the mod to execute |
| `passive_memory.jsonl` | Python pipeline | Full history of (state, action, reward) tuples |
| `death_memory.json` | Python pipeline | Persisted record of death events |
| `perceptor.json` | Python pipeline | Saved perceptor feature weights |

---

## Mod Compatibility

| DLC / Version | Compatible |
|--------------|-----------|
| Don't Starve (base) | ✅ |
| Don't Starve Together | ✅ |
| Reign of Giants | ✅ |
| Shipwrecked | ✅ |
| Hamlet | ✅ |

The mod is server-side only (`client_only_mod = false`) and does not require all clients to install it (`all_clients_require_mod = false`).

---

## Author

Made by **AndreiTGK**.

MIT License

Copyright (c) 2026 AndreiTGK

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
