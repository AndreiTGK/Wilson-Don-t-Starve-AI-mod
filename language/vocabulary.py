SPECIAL = [
    "<pad>",
    "<unk>",
    "<start>",
    "<end>",
    "...",
]

SENSORY_WORDS = [
    "canine_aggressor",
    "arachnid_entity",
    "thermal_combustion",
    "shrouded_locale",
    "ursine_proximity",
    "avian_presence",
    "swine_collective",
    "bovine_formation",
    "aquatic_entity",
    "shadow_manifestation",
    "tentacular_threat",
    "nocturnal_phase",
    "diurnal_phase",
    "crepuscular_phase",
    "precipitation_event",
    "tempest_condition",
    "arid_atmosphere",
    "frigid_environment",
    "luminous_source",
    "botanical_resource",
    "ligneous_cluster",
    "mineral_deposit",
    "crystalline_formation",
    "floral_presence",
    "fungal_specimen",
    "subterranean_cache",
    "elevated_terrain",
    "aquatic_proximity",
    "dense_foliage",
    "barren_expanse",
    "adjacent_threat",
    "distant_entity",
    "sonic_disturbance",
    "olfactory_signal",
    "spectral_presence",
    "hostile_proximity",
    "passive_fauna",
    "dormant_resource",
    "seasonal_autumn",
    "seasonal_winter",
    "seasonal_spring",
    "seasonal_summer",
    "storm_imminent",
    "lunar_darkness",
    "ignition_source",
    "constructed_structure",
    "open_terrain",
    "enclosed_space",
    "familiar_location",
    "unexplored_territory",
]

STATE_WORDS = [
    "caloric_deficit",
    "vitality_critical",
    "psychological_distress",
    "caloric_surplus",
    "vitality_optimal",
    "psychological_equilibrium",
    "satiation_adequate",
    "satiation_depleted",
    "thermal_discomfort",
    "moisture_saturation",
    "locomotive_fatigue",
    "cognitive_clarity",
    "cognitive_impairment",
    "existential_dread",
    "somatic_pain",
    "immune_compromise",
    "energy_depleted",
    "energy_restored",
    "paranoia_elevated",
    "temporal_disorientation",
    "memory_resonance",
    "memory_resonance_stigma",
    "deja_vu_activation",
    "mortality_imminent",
    "survival_sustained",
    "resource_deprived",
    "resource_adequate",
    "threat_proximity_detected",
    "physiological_stress",
    "neurological_disturbance",
    "vitality_stable",
    "caloric_marginal",
    "sanity_degrading",
    "sanity_restored",
    "fear_response",
    "pain_acute",
    "pain_chronic",
    "exhaustion_onset",
    "alertness_heightened",
    "vigilance_active",
    "dormancy_impulse",
]

DIRECTIVE_WORDS = [
    "tactical_retreat",
    "synthesize_apparatus",
    "resource_extraction",
    "territorial_advance",
    "predator_evasion",
    "nutritional_acquisition",
    "combustion_initiation",
    "structural_construction",
    "arboreal_harvest",
    "lapidary_extraction",
    "nocturnal_shelter",
    "diurnal_exploration",
    "threat_neutralization",
    "flora_cultivation",
    "tool_fabrication",
    "sustenance_prioritization",
    "perimeter_assessment",
    "coordinate_triangulation",
    "inventory_optimization",
    "emergency_protocol",
    "stasis_preservation",
    "momentum_maintenance",
    "reconnaissance_sweep",
    "cache_retrieval",
    "proximity_analysis",
    "defensive_posture",
    "offensive_engagement",
    "path_recalculation",
    "sustenance_processing",
    "combat_initiation",
    "environmental_assessment",
    "resource_cataloguing",
    "shelter_construction",
    "fire_maintenance",
    "strategic_withdrawal",
    "rapid_displacement",
    "nutritional_synthesis",
    "threat_monitoring",
    "spatial_mapping",
    "spatial_recalibration",
]

ALL_WORDS = SPECIAL + SENSORY_WORDS + STATE_WORDS + DIRECTIVE_WORDS

VOCAB = ALL_WORDS
WORD_TO_IDX = {w: i for i, w in enumerate(VOCAB)}
IDX_TO_WORD = {i: w for i, w in enumerate(VOCAB)}

VOCAB_SIZE  = len(VOCAB)
PAD_IDX     = WORD_TO_IDX["<pad>"]
UNK_IDX     = WORD_TO_IDX["<unk>"]
START_IDX   = WORD_TO_IDX["<start>"]
END_IDX     = WORD_TO_IDX["<end>"]

SENSORY_SET   = set(SENSORY_WORDS)
STATE_SET     = set(STATE_WORDS)
DIRECTIVE_SET = set(DIRECTIVE_WORDS)


def encode(word: str) -> int:
    return WORD_TO_IDX.get(word, UNK_IDX)


def decode(idx: int) -> str:
    return IDX_TO_WORD.get(idx, "<unk>")


def encode_sequence(words: list) -> list:
    return [encode(w) for w in words]


def decode_sequence(indices: list, strip_special=True) -> list:
    words = [decode(i) for i in indices]
    if strip_special:
        words = [w for w in words if w not in ("<pad>", "<start>", "<end>")]
    return words


if __name__ == "__main__":
    print(f"Wilson's vocabulary: {VOCAB_SIZE} words")
    print(f"Sensory: {len(SENSORY_WORDS)}, State: {len(STATE_WORDS)}, Directive: {len(DIRECTIVE_WORDS)}")
    print(f"First 10 words: {VOCAB[:10]}")
    print(f"Last 10 words:  {VOCAB[-10:]}")
