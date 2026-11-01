"""
Classification Rules
====================
All rule definitions, category mappings, and the multi-signal scoring
function for the Part 3 change interpretation engine.

Design philosophy
-----------------
* Primary signal  : object_type  (from Part 2 object detection)
* Secondary signal: change_type  (optional hint -- do NOT rely on it alone)
* Context signals : region_type, cd_change_magnitude, od_detection_confidence

Weights must sum to <= 1.0 for each component group.
"""

from typing import Optional

# ---------------------------------------------------------------------------
# CATEGORY LABELS
# ---------------------------------------------------------------------------

CATEGORY_HUMAN_MADE = "Human-made"
CATEGORY_NATURAL    = "Natural"
CATEGORY_UNCERTAIN  = "Uncertain"
VALID_CATEGORIES    = {CATEGORY_HUMAN_MADE, CATEGORY_NATURAL, CATEGORY_UNCERTAIN}

# ---------------------------------------------------------------------------
# OBJECT TYPE SETS  (Part 2 object-detection vocabulary)
# ---------------------------------------------------------------------------

HUMAN_MADE_OBJECT_TYPES: frozenset = frozenset({
    # New construction
    "building", "structure", "rooftop",
    # Demolition
    "building_ruins", "bare_ground", "debris",
    # Roads
    "road", "pathway", "pavement", "highway", "bridge",
    # Vehicle infrastructure
    "parking_lot", "vehicle_cluster", "transport_hub",
})

NATURAL_OBJECT_TYPES: frozenset = frozenset({
    # Vegetation change
    "cleared_vegetation", "bare_land", "logged_area",
    "dense_vegetation", "shrubland", "regrowth_patch",
    # Water
    "flooded_area", "inundated_land", "water_body",
    "reservoir", "lake_expansion",
    # Mass movement
    "debris_flow", "slope_failure",
})

UNCERTAIN_OBJECT_TYPES: frozenset = frozenset({
    "unclassified_region", "unknown_object", "unresolved_area",
    "blurred_region", "uncertain_feature", "ambiguous_patch",
    "mixed_land_use", "boundary_uncertainty", "unresolved_transition",
})

# ---------------------------------------------------------------------------
# CHANGE TYPE SETS  (used as a secondary hint, not the sole signal)
# ---------------------------------------------------------------------------

HUMAN_MADE_CHANGE_TYPES: frozenset = frozenset({
    "new_building_construction",
    "building_demolition",
    "road_construction",
    "road_expansion",
    "vehicle_infrastructure_changes",
})

NATURAL_CHANGE_TYPES: frozenset = frozenset({
    "vegetation_loss",
    "vegetation_growth",
    "flooding",
    "water_body_expansion",
    "landslide",
})

UNCERTAIN_CHANGE_TYPES: frozenset = frozenset({
    "unknown_object_change",
    "low_confidence_change",
    "ambiguous_geographical_change",
})

# ---------------------------------------------------------------------------
# REGION TYPE ASSOCIATIONS
# ---------------------------------------------------------------------------

HUMAN_MADE_REGIONS: frozenset = frozenset({"urban", "suburban", "industrial"})
NATURAL_REGIONS:    frozenset = frozenset({"forest", "rural", "coastal", "agricultural"})

# ---------------------------------------------------------------------------
# SCORING WEIGHTS   (all between 0 and 1; groups sum to 1)
# ---------------------------------------------------------------------------

WEIGHT_OBJECT_TYPE    = 0.40  # strongest signal -- from Part 2
WEIGHT_CHANGE_TYPE    = 0.20  # secondary hint
WEIGHT_REGION_TYPE    = 0.10  # geographical context
WEIGHT_CD_MAGNITUDE   = 0.15  # change magnitude from Part 1
WEIGHT_OD_CONFIDENCE  = 0.15  # object detection confidence from Part 2

# ---------------------------------------------------------------------------
# DECISION THRESHOLDS
# ---------------------------------------------------------------------------

CONFIDENCE_UNCERTAIN_THRESHOLD  = 0.50   # below this -> override to Uncertain
OD_CONF_LOW_THRESHOLD           = 0.40   # low OD confidence -> uncertain signal
OD_CONF_HIGH_THRESHOLD          = 0.75   # high OD confidence -> reinforce leader
CD_MAGNITUDE_LOW_THRESHOLD      = 0.30   # weak change -> uncertain signal
CD_MAGNITUDE_HIGH_THRESHOLD     = 0.60   # strong change -> reinforce leader
OD_CONF_HARD_OVERRIDE           = 0.30   # below this -> force Uncertain
CD_SCORE_HARD_OVERRIDE          = 0.25   # below this -> force Uncertain


# ---------------------------------------------------------------------------
# SCORING FUNCTION
# ---------------------------------------------------------------------------

def score_record(
    object_type:             str,
    change_type:             Optional[str] = None,
    region_type:             Optional[str] = None,
    cd_change_magnitude:     Optional[float] = None,
    od_detection_confidence: Optional[float] = None,
    **_kwargs,
) -> dict:
    """
    Compute per-category scores for a single change record using all
    available signals.

    Parameters
    ----------
    object_type             : Detected object class (from Part 2). Required.
    change_type             : Change type label (optional hint).
    region_type             : Geographical region type (optional context).
    cd_change_magnitude     : Change magnitude [0, 1] from Part 1 (optional).
    od_detection_confidence : Object detection confidence [0, 1] from Part 2 (optional).

    Returns
    -------
    dict
        human_score    : float
        natural_score  : float
        uncertain_score: float
        components     : list[dict] -- one entry per scoring decision
    """
    human     = 0.0
    natural   = 0.0
    uncertain = 0.0
    components: list[dict] = []

    def _add(target: str, weight: float, component: str, reason: str) -> None:
        nonlocal human, natural, uncertain
        if target == CATEGORY_HUMAN_MADE:
            human += weight
        elif target == CATEGORY_NATURAL:
            natural += weight
        else:
            uncertain += weight
        components.append(
            {"component": component, "signal": target, "weight": weight, "reason": reason}
        )

    # ---- 1. Object type (weight 0.40) ----------------------------------------
    if object_type in HUMAN_MADE_OBJECT_TYPES:
        _add(CATEGORY_HUMAN_MADE, WEIGHT_OBJECT_TYPE, "object_type",
             f"Object '{object_type}' is a human-made structure")
    elif object_type in NATURAL_OBJECT_TYPES:
        _add(CATEGORY_NATURAL, WEIGHT_OBJECT_TYPE, "object_type",
             f"Object '{object_type}' is associated with a natural process")
    elif object_type in UNCERTAIN_OBJECT_TYPES:
        _add(CATEGORY_UNCERTAIN, WEIGHT_OBJECT_TYPE, "object_type",
             f"Object '{object_type}' cannot be definitively classified")
    else:
        # Unrecognised object type -> half-weight uncertain signal
        _add(CATEGORY_UNCERTAIN, WEIGHT_OBJECT_TYPE * 0.5, "object_type",
             f"Object '{object_type}' is not in the known taxonomy")

    # ---- 2. Change type hint (weight 0.20) ------------------------------------
    if change_type:
        if change_type in HUMAN_MADE_CHANGE_TYPES:
            _add(CATEGORY_HUMAN_MADE, WEIGHT_CHANGE_TYPE, "change_type",
                 f"Change type '{change_type}' indicates human activity")
        elif change_type in NATURAL_CHANGE_TYPES:
            _add(CATEGORY_NATURAL, WEIGHT_CHANGE_TYPE, "change_type",
                 f"Change type '{change_type}' indicates a natural process")
        elif change_type in UNCERTAIN_CHANGE_TYPES:
            _add(CATEGORY_UNCERTAIN, WEIGHT_CHANGE_TYPE, "change_type",
                 f"Change type '{change_type}' is inherently uncertain")
        # Unknown change_type -> no additional score

    # ---- 3. Region type (weight 0.10) ----------------------------------------
    if region_type:
        if region_type in HUMAN_MADE_REGIONS:
            _add(CATEGORY_HUMAN_MADE, WEIGHT_REGION_TYPE, "region_type",
                 f"Region '{region_type}' is associated with human infrastructure")
        elif region_type in NATURAL_REGIONS:
            _add(CATEGORY_NATURAL, WEIGHT_REGION_TYPE, "region_type",
                 f"Region '{region_type}' is associated with natural processes")
        else:
            _add(CATEGORY_UNCERTAIN, WEIGHT_REGION_TYPE * 0.5, "region_type",
                 f"Region '{region_type}' is ambiguous")

    # ---- 4. Change magnitude (weight 0.15) ------------------------------------
    if cd_change_magnitude is not None:
        if cd_change_magnitude >= CD_MAGNITUDE_HIGH_THRESHOLD:
            # Strong change reinforces whichever category leads so far
            leading_score = max(human, natural)
            if human >= leading_score and human > 0:
                _add(CATEGORY_HUMAN_MADE, WEIGHT_CD_MAGNITUDE, "cd_magnitude",
                     f"High magnitude ({cd_change_magnitude:.2f}) reinforces Human-made")
            elif natural > 0:
                _add(CATEGORY_NATURAL, WEIGHT_CD_MAGNITUDE, "cd_magnitude",
                     f"High magnitude ({cd_change_magnitude:.2f}) reinforces Natural")
            else:
                _add(CATEGORY_UNCERTAIN, WEIGHT_CD_MAGNITUDE * 0.5, "cd_magnitude",
                     f"High magnitude ({cd_change_magnitude:.2f}) but no clear leader")
        elif cd_change_magnitude < CD_MAGNITUDE_LOW_THRESHOLD:
            # Weak change -> uncertain signal
            _add(CATEGORY_UNCERTAIN, WEIGHT_CD_MAGNITUDE * 0.75, "cd_magnitude",
                 f"Low magnitude ({cd_change_magnitude:.2f}) -- weak or uncertain change")
        # Moderate magnitude -> neutral, no extra score

    # ---- 5. Object detection confidence (weight 0.15) ------------------------
    if od_detection_confidence is not None:
        if od_detection_confidence >= OD_CONF_HIGH_THRESHOLD:
            # High OD confidence reinforces leading category
            if human >= natural and human > 0:
                _add(CATEGORY_HUMAN_MADE, WEIGHT_OD_CONFIDENCE, "od_confidence",
                     f"High OD confidence ({od_detection_confidence:.2f}) reinforces Human-made")
            elif natural > 0:
                _add(CATEGORY_NATURAL, WEIGHT_OD_CONFIDENCE, "od_confidence",
                     f"High OD confidence ({od_detection_confidence:.2f}) reinforces Natural")
            else:
                _add(CATEGORY_UNCERTAIN, WEIGHT_OD_CONFIDENCE * 0.5, "od_confidence",
                     "High OD confidence but no leading non-uncertain category")
        elif od_detection_confidence < OD_CONF_LOW_THRESHOLD:
            _add(CATEGORY_UNCERTAIN, WEIGHT_OD_CONFIDENCE, "od_confidence",
                 f"Low OD confidence ({od_detection_confidence:.2f}) makes classification uncertain")
        # Moderate confidence -> neutral

    return {
        "human_score":    round(human, 4),
        "natural_score":  round(natural, 4),
        "uncertain_score": round(uncertain, 4),
        "components":     components,
    }


def check_object_change_consistency(object_type: str, change_type: Optional[str]) -> bool:
    """
    Return True when object_type and change_type belong to the same category,
    or when change_type is None (no contradiction possible).
    """
    if not change_type:
        return True

    obj_human    = object_type in HUMAN_MADE_OBJECT_TYPES
    obj_natural  = object_type in NATURAL_OBJECT_TYPES
    obj_uncertain = object_type in UNCERTAIN_OBJECT_TYPES

    ct_human    = change_type in HUMAN_MADE_CHANGE_TYPES
    ct_natural  = change_type in NATURAL_CHANGE_TYPES
    ct_uncertain = change_type in UNCERTAIN_CHANGE_TYPES

    if obj_human    and ct_human:    return True
    if obj_natural  and ct_natural:  return True
    if obj_uncertain and ct_uncertain: return True
    return False
