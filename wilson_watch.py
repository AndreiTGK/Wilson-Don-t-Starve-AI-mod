"""
wilson_watch.py
───────────────
Read-only telemetry dashboard for the Wilson AI mod.

Polls wilson_ai_state for live game state and tails passive_memory.jsonl
and wilson_thoughts.jsonl for output from the pipeline.

This process NEVER writes to wilson_ai_action.txt and NEVER loads the
neural network. Run pipeline.py as the agent brain alongside this.

  +/-     speed up / slow down (writes wilson_ai_speed, read by Lua)
  Ctrl+C  quit
"""
import atexit
import json, math, os, re, sys, time, platform, threading
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))
from agent.actions import ACTION_LABELS

# ── ANSI ──────────────────────────────────────────────────────────────────────
R="\033[91m"; G="\033[92m"; Y="\033[93m"; B="\033[94m"
M="\033[95m"; C="\033[96m"; W="\033[97m"; DIM="\033[2m"
IT="\033[3m"; RST="\033[0m"

# ── Speed ─────────────────────────────────────────────────────────────────────
SPEED_STEPS   = [1, 2, 3, 5, 8, 10, 15, 20]
current_speed = 1

def speed_up():
    global current_speed
    idx = SPEED_STEPS.index(current_speed) if current_speed in SPEED_STEPS else 0
    if idx < len(SPEED_STEPS) - 1:
        current_speed = SPEED_STEPS[idx + 1]

def speed_down():
    global current_speed
    idx = SPEED_STEPS.index(current_speed) if current_speed in SPEED_STEPS else 0
    if idx > 0:
        current_speed = SPEED_STEPS[idx - 1]

def write_speed(save_dir: Path):
    try:
        (Path(__file__).parent / "wilson_ai_speed").write_text(str(current_speed), encoding="utf-8")
    except Exception:
        pass

def render_speed_bar() -> str:
    bar = "".join(f"{Y}█{RST}" if s <= current_speed else f"{DIM}░{RST}" for s in SPEED_STEPS)
    lbl = f"{Y}{current_speed}x{RST}" if current_speed > 1 else f"{G}1x  normal{RST}"
    return f"  Speed  [{bar}] {lbl}   {DIM}(+/-){RST}"

# ── Keyboard (non-blocking) ───────────────────────────────────────────────────
_key_q    = []
_key_lock = threading.Lock()

def _key_reader():
    try:
        if platform.system() == "Windows":
            import msvcrt
            while True:
                if msvcrt.kbhit():
                    with _key_lock: _key_q.append(msvcrt.getwch())
                time.sleep(0.05)
        else:
            import termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)

            def restore_terminal():
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            atexit.register(restore_terminal)

            new_settings = termios.tcgetattr(fd)
            new_settings[3] = new_settings[3] & ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)

            while True:
                ch = sys.stdin.read(1)
                with _key_lock: _key_q.append(ch)
    except Exception:
        pass

def poll_keys(save_dir: Path):
    changed = False
    with _key_lock:
        keys = list(_key_q); _key_q.clear()
    for ch in keys:
        if ch in ("+", "="): speed_up();   changed = True
        elif ch in ("-", "_"): speed_down(); changed = True
        elif ch == "\x03": raise KeyboardInterrupt
    if changed:
        write_speed(save_dir)

# ── Thought log (tailed from wilson_thoughts.jsonl) ───────────────────────────
thought_log  = deque(maxlen=8)
thought_lock = threading.Lock()
_last_thought_mtime = 0.0

def poll_thoughts(mod_dir: Path):
    global _last_thought_mtime
    path = mod_dir / "wilson_thoughts.jsonl"
    if not path.exists():
        return
    try:
        mtime = path.stat().st_mtime
        if mtime == _last_thought_mtime:
            return
        _last_thought_mtime = mtime
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        with thought_lock:
            thought_log.clear()
            for line in lines[-8:]:
                try:
                    thought_log.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass

# ── Passive memory tail (from passive_memory.jsonl) ───────────────────────────
passive_log  = deque(maxlen=8)
passive_lock = threading.Lock()
_last_passive_mtime = 0.0

def poll_passive_memory(mod_dir: Path):
    global _last_passive_mtime
    path = mod_dir / "passive_memory.jsonl"
    if not path.exists():
        return
    try:
        mtime = path.stat().st_mtime
        if mtime == _last_passive_mtime:
            return
        _last_passive_mtime = mtime
        with open(path, "rb") as f:
            f.seek(0, 2)
            pos = f.tell()
            buf = b""
            found = 0
            while pos > 0 and found < 9:
                chunk = min(4096, pos)
                pos  -= chunk
                f.seek(pos)
                buf   = f.read(chunk) + buf
                found = buf.count(b"\n")
        entries = []
        for part in buf.split(b"\n"):
            part = part.strip()
            if part:
                try:
                    entries.append(json.loads(part))
                except Exception:
                    pass
        with passive_lock:
            passive_log.clear()
            passive_log.extend(entries[-8:])
    except Exception:
        pass

def _last_passive_entry() -> dict:
    with passive_lock:
        return dict(passive_log[-1]) if passive_log else {}

# ── Pipeline action (read-only peek at wilson_ai_action.txt) ──────────────────
def read_current_action(mod_dir: Path) -> int:
    try:
        return int((mod_dir / "wilson_ai_action.txt").read_text().strip())
    except Exception:
        return -1

# ── Save dir helpers ──────────────────────────────────────────────────────────
def find_save_dir() -> Path | None:
    sys_ = platform.system()
    if sys_ == "Windows":
        base = Path(os.environ.get("USERPROFILE", "C:/Users/User"))
        candidates = [
            base / "Documents/Klei/DoNotStarve",
            base / "Documents/Klei/DoNotStarveTogether",
        ]
    elif sys_ == "Darwin":
        base = Path.home()
        candidates = [
            base / "Documents/Klei/DoNotStarve",
            base / "Documents/Klei/DoNotStarveTogether",
        ]
    else:
        base = Path.home()
        steam_userdata = base / ".local/share/Steam/userdata"
        steam_candidates = []
        if steam_userdata.exists():
            for uid_dir in sorted(steam_userdata.iterdir()):
                if not uid_dir.is_dir():
                    continue
                ds_path  = uid_dir / "219740/remote"
                dst_path = uid_dir / "322330/remote"
                if ds_path.exists():  steam_candidates.append(ds_path)
                if dst_path.exists(): steam_candidates.append(dst_path)

        proton  = base / ".local/share/Steam/steamapps/compatdata/322330/pfx/drive_c/users/steamuser"
        flatpak = base / ".var/app/com.valvesoftware.Steam/data/Steam/steamapps/compatdata/322330/pfx/drive_c/users/steamuser"
        candidates = steam_candidates + [
            base / ".klei/DoNotStarve",
            base / ".klei/DoNotStarveTogether",
            proton / "Documents/Klei/DoNotStarveTogether",
            proton / "Documents/Klei/DoNotStarve",
            proton / "My Documents/Klei/DoNotStarveTogether",
            proton / "My Documents/Klei/DoNotStarve",
            flatpak / "Documents/Klei/DoNotStarveTogether",
            flatpak / "My Documents/Klei/DoNotStarveTogether",
        ]
    for p in candidates:
        if p.exists(): return p
    return None

# ── Render helpers ────────────────────────────────────────────────────────────
def bar(value, maximum, width=20) -> str:
    if maximum <= 0: return "[" + "?"*width + "]"
    filled = max(0, min(width, int((value/maximum)*width)))
    pct    = value/maximum
    color  = R if pct < 0.25 else Y if pct < 0.5 else G
    return f"[{color}{'█'*filled}{DIM}{'░'*(width-filled)}{RST}] {value:.0f}/{maximum:.0f}"

def render_entity(ent: dict) -> str:
    tag = f"{DIM}(mem){RST}" if ent.get("radius") == "memory" else ""
    hp  = f" hp:{ent['hp_pct']:.0%}" if ent.get("hp_pct") is not None else ""
    return f"  {C}{ent['prefab']:20s}{RST} {ent['dist']:5.1f}t  {ent['angle']:4d}° {hp} {tag}"

def render_sounds(sounds: dict) -> str:
    if not sounds: return f"  {DIM}silence{RST}"
    active   = [k for k, v in sounds.items() if v == 1]
    inactive = [k for k, v in sounds.items() if v == 0]
    line = "  "
    for k in active:   line += f"{R}▐{k}  {RST}"
    for k in inactive: line += f"{DIM}·{k}  {RST}"
    return line

# ── Radar ─────────────────────────────────────────────────────────────────────
RADAR_ROWS   = 21
RADAR_COLS   = 41
RADAR_VISION = 40.0

_RADAR_PRIORITY = {"W": 99, "!": 5, "f": 4, "$": 3, "t": 2, "#": 1, "·": 0, ".": -1}
_RADAR_COLORS   = {
    "!": "\033[31m", "f": "\033[32m", "$": "\033[33m",
    "#": "\033[34m", "t": "\033[96m", ".": "\033[2m",
    "·": "\033[90m", "?": "\033[2m",
}

_HOSTILE_WORDS  = ("spider","hound","bee","tallbird","deerclops","bearger","dragonfly","leif","warg")
_BOULDER_WORDS  = ("rock","boulder","marble","stalagmite")
_VALUABLE_WORDS = ("gold","flint","axe","pickaxe","shovel","spear","hammer","razor")
_FOOD_WORDS     = ("berries","carrot","mushroom","dragonfruit","pomegranate","durian",
                   "eggplant","banana","honey","morsel","meat","fish","egg")
_TREE_WORDS     = ("tree","evergreen","deciduous","lumpy","log","pinecone")

_ANSI_RE = re.compile(r'\033\[[^m]*m')

_history: list     = []
_phase_track: dict = {"phase": "", "wall_time": 0.0}

_DIR_ARROWS = ["^", "↗", ">", "↘", "v", "↙", "<", "↖"]

def color_text(text: str, color_code: int) -> str:
    return f"\033[{color_code}m{text}\033[0m"

def _visual_len(s: str) -> int:
    return len(_ANSI_RE.sub("", s))

def _pad_to(s: str, width: int) -> str:
    return s + " " * max(0, width - _visual_len(s))

def _entity_symbol(prefab: str) -> str:
    p = prefab.lower()
    if any(w in p for w in _HOSTILE_WORDS):  return "!"
    if any(w in p for w in _BOULDER_WORDS):  return "#"
    if any(w in p for w in _VALUABLE_WORDS): return "$"
    if any(w in p for w in _FOOD_WORDS):     return "f"
    if any(w in p for w in _TREE_WORDS):     return "t"
    return "?"

def _heading_from_history() -> str:
    if len(_history) < 2:
        return "o"
    prev_x, prev_z = _history[-2]
    curr_x, curr_z = _history[-1]
    dx, dz = curr_x - prev_x, curr_z - prev_z
    if abs(dx) < 0.3 and abs(dz) < 0.3:
        return "o"
    angle = (math.degrees(math.atan2(dx, dz)) + 360) % 360
    return _DIR_ARROWS[int((angle + 22.5) / 45) % 8]

def _bresenham_line(r0: int, c0: int, r1: int, c1: int) -> list:
    pts = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
    err = dr - dc
    r, c = r0, c0
    while True:
        pts.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc: err -= dc; r += sr
        if e2 < dr:  err += dr; c += sc
    return pts

def _time_until_next_phase(phase: str, game_speed: int) -> tuple:
    if phase != _phase_track["phase"]:
        _phase_track.update({"phase": phase, "wall_time": time.time()})
    elapsed   = time.time() - _phase_track["wall_time"]
    base_dur  = {"day": 270, "dusk": 90, "night": 120}.get(phase, 120)
    adj_dur   = max(1.0, base_dur / max(1, game_speed))
    remaining = max(0, int(adj_dur - elapsed))
    next_phase= {"day": "dusk", "dusk": "night", "night": "day"}.get(phase, "?")
    return next_phase, remaining


def render_radar(state: dict, step_count: int,
                 last_action: int = -1, pipeline_reward: float = 0.0):
    s      = state.get("self", {})
    inv    = state.get("inventory", [])
    w      = state.get("world", {})
    hp     = s.get("hp",     0); hp_max  = s.get("hp_max",     150) or 150
    hunger = s.get("hunger", 0); hun_max = s.get("hunger_max", 150) or 150
    sanity = s.get("sanity", 0); san_max = s.get("sanity_max", 200) or 200
    phase      = w.get("phase", "day")
    game_speed = state.get("speed", 1) or 1
    wx, wz     = s.get("x", 0.0), s.get("z", 0.0)

    sys.stdout.write("\033[2J\033[H")

    with thought_lock:
        snap = list(thought_log)
    thought_text = (snap[-1].get("thought", "...") if snap else "...")[: RADAR_COLS - 2]

    action_label = ACTION_LABELS.get(last_action, f"#{last_action}") if last_action >= 0 else "…"

    # ── Grid ────────────────────────────────────────────────────────────────
    cr, cc = RADAR_ROWS // 2, RADAR_COLS // 2
    grid   = [["." for _ in range(RADAR_COLS)] for _ in range(RADAR_ROWS)]

    for bx, bz in _history[:-1]:
        col = cc + round((bx - wx) / RADAR_VISION * cc)
        row = cr - round((bz - wz) / RADAR_VISION * cr)
        if 0 <= row < RADAR_ROWS and 0 <= col < RADAR_COLS:
            if _RADAR_PRIORITY["·"] > _RADAR_PRIORITY.get(grid[row][col], -1):
                grid[row][col] = "·"

    perceived  = state.get("perceived", [])
    remembered = state.get("remembered", [])
    hostiles_in_range: list = []
    nearest_hostile_name = ""
    nearest_hostile_dist = float("inf")

    for ent in perceived + remembered:
        dist   = float(ent.get("dist",  0))
        angle  = float(ent.get("angle", 0))
        prefab = ent.get("prefab", "")
        sym    = _entity_symbol(prefab)
        dx_w   = dist * math.sin(math.radians(angle))
        dz_w   = dist * math.cos(math.radians(angle))
        col    = cc + round(dx_w / RADAR_VISION * cc)
        row    = cr - round(dz_w / RADAR_VISION * cr)
        if 0 <= row < RADAR_ROWS and 0 <= col < RADAR_COLS:
            if _RADAR_PRIORITY.get(sym, 0) > _RADAR_PRIORITY.get(grid[row][col], -1):
                grid[row][col] = sym
            if sym == "!" and dist <= 15.0:
                hostiles_in_range.append((row, col))
        if sym == "!" and dist < nearest_hostile_dist:
            nearest_hostile_dist = dist
            nearest_hostile_name = prefab

    for hr, hc in hostiles_in_range:
        for r, c in _bresenham_line(cr, cc, hr, hc)[1:-1]:
            if 0 <= r < RADAR_ROWS and 0 <= c < RADAR_COLS:
                if _RADAR_PRIORITY["·"] > _RADAR_PRIORITY.get(grid[r][c], -1):
                    grid[r][c] = "·"

    grid[cr][cc] = _heading_from_history()

    colored_rows = [
        "".join(
            (W if (r == cr and c == cc) else _RADAR_COLORS.get(cell, RST)) + cell + RST
            for c, cell in enumerate(row)
        )
        for r, row in enumerate(grid)
    ]

    # ── Legend values ────────────────────────────────────────────────────────
    hp_pct  = int(100 * hp     / hp_max)
    hun_pct = int(100 * hunger / hun_max)
    san_pct = int(100 * sanity / san_max)

    def _pct(v: int) -> str:
        return color_text(f"{v}%", 31 if v < 25 else 33 if v < 50 else 32)

    next_phase, time_left = _time_until_next_phase(phase, game_speed)
    pc = {"day": Y, "dusk": M, "night": B}.get(phase, W)

    if nearest_hostile_name:
        target_str = color_text(f"{nearest_hostile_name} at {nearest_hostile_dist:.1f} units", 31)
    else:
        target_str = color_text("none", 32)

    hp_col     = R if hp_pct < 25 else Y if hp_pct < 50 else G
    reward_col = G if pipeline_reward >= 0 else R
    border_h   = "═" * RADAR_COLS

    h1 = f"  Step {Y}{step_count:>6d}{RST}  HP {hp_col}{hp:.0f}/{hp_max:.0f}{RST}  Inv {G}{len(inv):>2d}{RST}"
    h2 = f"  {DIM}pipeline▸{RST} {G}{action_label:18s}{RST} {DIM}r={RST}{reward_col}{pipeline_reward:+.2f}{RST}"
    h3 = f"  {IT}{DIM}{thought_text}{RST}"
    h4 = f"  {DIM}! hostile  f food  $ valuable  t tree  # ocean  · path{RST}"

    out_lines = [
        f"{W}╔{border_h}╗{RST}",
        f"{W}│{RST}{_pad_to(h1, RADAR_COLS)}{W}│{RST}",
        f"{W}│{RST}{_pad_to(h2, RADAR_COLS)}{W}│{RST}",
        f"{W}│{RST}{_pad_to(h3, RADAR_COLS)}{W}│{RST}",
        f"{W}│{RST}{_pad_to(h4, RADAR_COLS)}{W}│{RST}",
        f"{W}╠{border_h}╣{RST}",
        *[f"{W}│{RST}{crow}{W}│{RST}" for crow in colored_rows],
        f"{W}╚{border_h}╝{RST}",
        f"  [ STATUS ] HP: {_pct(hp_pct)} | Hunger: {_pct(hun_pct)} | Sanity: {_pct(san_pct)}",
        f"  [ WORLD  ] Phase: {pc}{phase}{RST} | Time until {next_phase}: {Y}{time_left}s{RST}",
        f"  [ TARGET ] Nearest Threat: {target_str}",
        f"  {render_speed_bar()}",
        "",
    ]

    sys.stdout.write("\n".join(out_lines))
    sys.stdout.flush()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global current_speed
    print(f"{Y}Wilson Watch  {DIM}[read-only telemetry — run pipeline.py for the agent]{RST}")

    save_dir = find_save_dir()
    if not save_dir:
        print(f"{R}Could not find Don't Starve save directory.{RST}")
        manual = input("Enter path manually (or Enter for current dir): ").strip()
        save_dir = Path(manual) if manual else Path.cwd()

    mod_dir    = Path(__file__).parent
    state_file = mod_dir / "wilson_ai_state"

    print(f"{G}Mod dir:   {mod_dir}{RST}")
    print(f"{DIM}Watching state file. This process is read-only.{RST}")

    threading.Thread(target=_key_reader, daemon=True).start()

    last_state_mtime = 0.0
    step_count = 0
    spinner    = list("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")

    while True:
        try:
            poll_keys(save_dir)
            poll_thoughts(mod_dir)
            poll_passive_memory(mod_dir)

            if not state_file.exists():
                t = int(time.time() * 10) % len(spinner)
                sys.stdout.write(f"\r{DIM}{spinner[t]} Waiting for game state...{RST}")
                sys.stdout.flush()
                time.sleep(0.05)
                continue

            try:
                mtime = state_file.stat().st_mtime
                if mtime == last_state_mtime:
                    time.sleep(0.05)
                    continue
                raw = state_file.read_text(encoding="utf-8")
            except Exception:
                time.sleep(0.05)
                continue

            if not raw.endswith("--END--"):
                time.sleep(0.02)  # partial write in progress
                continue

            last_state_mtime = mtime  # mark as seen; do NOT unlink

            raw = raw[: raw.rfind("\n--END--")].strip()
            if not raw:
                continue

            brace = raw.find("{")
            if brace > 0:
                raw = raw[brace:]
            elif raw.startswith('"') and raw.endswith('"'):
                raw = raw[1:-1].replace('\\"', '"').replace("\\n", "\n")

            try:
                state = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"{R}JSON parse error: {e}{RST}")
                continue

            _s = state.get("self", {})
            _history.append((_s.get("x", 0.0), _s.get("z", 0.0)))
            if len(_history) > 10:
                _history.pop(0)

            step_count += 1

            if step_count == 1 or step_count % 40 == 0:
                last_action   = read_current_action(mod_dir)
                last_reward   = _last_passive_entry().get("reward", 0.0)
                render_radar(state, step_count, last_action, last_reward)

        except KeyboardInterrupt:
            current_speed = 1
            write_speed(save_dir)
            print(f"\n{Y}Wilson Watch stopped.{RST}")
            sys.exit(0)
        except Exception as e:
            print(f"{R}Error: {e}{RST}")
            time.sleep(1)

if __name__ == "__main__":
    if platform.system() == "Windows":
        os.system("color")
    main()
