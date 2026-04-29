"""
item_knowledge.py
─────────────────
Item purpose taxonomy + learned utility tracker.

Purposes are hardcoded semantic facts (food is food).
Values are NOT hardcoded — Wilson learns them via ItemValueTracker
by observing which items correlate with positive reward.
"""

PURPOSES = ["food", "material", "tool", "weapon", "armor", "light", "structure", "unknown"]
N_PURPOSES = len(PURPOSES)
_PURPOSE_IDX = {p: i for i, p in enumerate(PURPOSES)}

PURPOSE_MAP: dict[str, str] = {
    # Food
    "berries": "food", "roasted_berries": "food",
    "carrot": "food", "roasted_carrot": "food",
    "meat": "food", "cookedmeat": "food",
    "morsel": "food", "cookedmorsel": "food",
    "egg": "food", "cookedegg": "food",
    "honey": "food", "honeycomb": "food",
    "mushroom": "food", "cookedmushroom": "food",
    "pumpkin": "food", "cookedpumpkin": "food",
    "watermelon": "food", "fruitmedley": "food",
    "meatballs": "food", "butterflymuffin": "food",
    "trailmix": "food", "honeynuggets": "food",
    "wetgoop": "food",
    # Materials
    "twigs": "material", "cutgrass": "material",
    "log": "material", "flint": "material",
    "rocks": "material", "goldnugget": "material",
    "charcoal": "material", "ash": "material",
    "silk": "material", "pigskin": "material",
    "beefalo_wool": "material", "tentaclespots": "material",
    "snurtleshell": "material", "reeds": "material",
    "nightmarefuel": "material", "spidergland": "material",
    "fireflies": "material", "rope": "material",
    "boards": "material", "cutstone": "material",
    "papyrus": "material", "poop": "material",
    "purplegem": "material", "bluegem": "material", "redgem": "material",
    # Tools
    "axe": "tool", "goldenaxe": "tool",
    "pickaxe": "tool", "goldenpickaxe": "tool",
    "shovel": "tool", "goldenshovel": "tool",
    "fishingrod": "tool", "birdtrap": "tool",
    "trap": "tool", "razor": "tool",
    "hammer": "tool", "backpack": "tool",
    "bedroll_straw": "tool", "healaniment": "tool",
    "amulet": "tool", "lifegiving_amulet": "tool",
    # Weapons
    "spear": "weapon", "tentaclespike": "weapon",
    "hambat": "weapon", "boomerang": "weapon",
    "blowdart_sleep": "weapon", "blowdart_fire": "weapon",
    # Armor
    "armorgrass": "armor", "armorwood": "armor",
    "armorsnurtle": "armor", "footballhat": "armor",
    "tophat": "armor", "beefalohat": "armor",
    "winterhat": "armor",
    # Light
    "torch": "light", "lantern": "light",
    # Structure (placeable items)
    "campfire": "structure", "firepit": "structure",
    "researchlab": "structure", "researchlab2": "structure",
    "icebox": "structure", "crockpot": "structure",
    "chest": "structure", "dryer": "structure",
}


def get_purpose(prefab: str) -> str:
    return PURPOSE_MAP.get(prefab, "unknown")


def get_purpose_idx(prefab: str) -> int:
    return _PURPOSE_IDX.get(get_purpose(prefab), _PURPOSE_IDX["unknown"])


class ItemValueTracker:
    """
    Wilson's learned sense of item value.

    Each step: observe which items were in inventory and what reward
    was received. Run EMA so items held during good outcomes float up,
    items held during bad outcomes float down.

    Survives across episodes so the knowledge accumulates over many lives.
    """

    def __init__(self, alpha: float = 0.005):
        self.alpha = alpha
        self.utility: dict[str, float] = {}
        self.encounter_count: dict[str, int] = {}

    def update(self, inv_prefabs: list[str], step_reward: float):
        for prefab in set(inv_prefabs):
            if not prefab:
                continue
            if prefab not in self.utility:
                self.utility[prefab] = 0.0
                self.encounter_count[prefab] = 0
            self.encounter_count[prefab] += 1
            self.utility[prefab] += self.alpha * (step_reward - self.utility[prefab])

    def get_utility(self, prefab: str) -> float:
        return self.utility.get(prefab, 0.0)

    def top_items(self, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self.utility.items(), key=lambda x: x[1], reverse=True)[:n]

    def to_dict(self) -> dict:
        return {
            "utility": self.utility,
            "encounter_count": self.encounter_count,
        }

    def from_dict(self, d: dict):
        self.utility = d.get("utility", {})
        self.encounter_count = d.get("encounter_count", {})
