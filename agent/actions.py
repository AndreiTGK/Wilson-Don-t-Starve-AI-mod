"""
actions.py
──────────
Wilson's complete action space. Every decision he can make.
Now upgraded to a 40-discrete-action spatial system.
"""

from enum import IntEnum

class Action(IntEnum):
    # ── Walk (0-7) ──
    WALK_N  = 0
    WALK_NE = 1
    WALK_E  = 2
    WALK_SE = 3
    WALK_S  = 4
    WALK_SW = 5
    WALK_W  = 6
    WALK_NW = 7

    STOP    = 8
    NO_OP_9 = 9

    # ── Interact Spatial (10-17) ──
    INTERACT_N  = 10
    INTERACT_NE = 11
    INTERACT_E  = 12
    INTERACT_SE = 13
    INTERACT_S  = 14
    INTERACT_SW = 15
    INTERACT_W  = 16
    INTERACT_NW = 17

    NO_OP_18 = 18
    NO_OP_19 = 19

    # ── Attack Spatial (20-27) ──
    ATTACK_N  = 20
    ATTACK_NE = 21
    ATTACK_E  = 22
    ATTACK_SE = 23
    ATTACK_S  = 24
    ATTACK_SW = 25
    ATTACK_W  = 26
    ATTACK_NW = 27

    NO_OP_28 = 28
    NO_OP_29 = 29

    # ── Inventory Use/Equip (30-39) ──
    INV_1  = 30
    INV_2  = 31
    INV_3  = 32
    INV_4  = 33
    INV_5  = 34
    INV_6  = 35
    INV_7  = 36
    INV_8  = 37
    INV_9  = 38
    INV_10 = 39

    # ── Craft Slots (40-47) ──
    CRAFT_0 = 40
    CRAFT_1 = 41
    CRAFT_2 = 42
    CRAFT_3 = 43
    CRAFT_4 = 44
    CRAFT_5 = 45
    CRAFT_6 = 46
    CRAFT_7 = 47

    # ── Place Spatial (48-55) ──
    PLACE_N  = 48
    PLACE_NE = 49
    PLACE_E  = 50
    PLACE_SE = 51
    PLACE_S  = 52
    PLACE_SW = 53
    PLACE_W  = 54
    PLACE_NW = 55

    PLACE_HERE = 56

N_ACTIONS = len(Action)

# Human-readable labels for the terminal UI
ACTION_LABELS = {
    Action.WALK_N:  "Walk N",  Action.WALK_NE: "Walk NE",
    Action.WALK_E:  "Walk E",  Action.WALK_SE: "Walk SE",
    Action.WALK_S:  "Walk S",  Action.WALK_SW: "Walk SW",
    Action.WALK_W:  "Walk W",  Action.WALK_NW: "Walk NW",

    Action.STOP:    "Stop",
    Action.NO_OP_9: "(thinking)",

    Action.INTERACT_N:  "Interact N",  Action.INTERACT_NE: "Interact NE",
    Action.INTERACT_E:  "Interact E",  Action.INTERACT_SE: "Interact SE",
    Action.INTERACT_S:  "Interact S",  Action.INTERACT_SW: "Interact SW",
    Action.INTERACT_W:  "Interact W",  Action.INTERACT_NW: "Interact NW",

    Action.NO_OP_18: "(thinking)", Action.NO_OP_19: "(thinking)",

    Action.ATTACK_N:  "Attack N",  Action.ATTACK_NE: "Attack NE",
    Action.ATTACK_E:  "Attack E",  Action.ATTACK_SE: "Attack SE",
    Action.ATTACK_S:  "Attack S",  Action.ATTACK_SW: "Attack SW",
    Action.ATTACK_W:  "Attack W",  Action.ATTACK_NW: "Attack NW",

    Action.NO_OP_28: "(thinking)", Action.NO_OP_29: "(thinking)",

    Action.INV_1: "Use Slot 1", Action.INV_2: "Use Slot 2",
    Action.INV_3: "Use Slot 3", Action.INV_4: "Use Slot 4",
    Action.INV_5: "Use Slot 5", Action.INV_6: "Use Slot 6",
    Action.INV_7: "Use Slot 7", Action.INV_8: "Use Slot 8",
    Action.INV_9: "Use Slot 9", Action.INV_10:"Use Slot 10",

    Action.CRAFT_0: "Craft slot 0", Action.CRAFT_1: "Craft slot 1",
    Action.CRAFT_2: "Craft slot 2", Action.CRAFT_3: "Craft slot 3",
    Action.CRAFT_4: "Craft slot 4", Action.CRAFT_5: "Craft slot 5",
    Action.CRAFT_6: "Craft slot 6", Action.CRAFT_7: "Craft slot 7",

    Action.PLACE_N:  "Place N",  Action.PLACE_NE: "Place NE",
    Action.PLACE_E:  "Place E",  Action.PLACE_SE: "Place SE",
    Action.PLACE_S:  "Place S",  Action.PLACE_SW: "Place SW",
    Action.PLACE_W:  "Place W",  Action.PLACE_NW: "Place NW",

    Action.PLACE_HERE: "Place here",
}
