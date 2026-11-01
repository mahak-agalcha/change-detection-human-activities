"""
ClassificationResult -- data model for a single change classification output.
Standalone module: no imports from the rest of Part 3 engine.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClassificationResult:
    """
    Represents the full output of classifying one geospatial change record.

    All fields are populated by ChangeClassifier; downstream consumers
    (Part 4 API, database layer) should read from to_dict() or to_geojson_properties().
    """

    # Identity
    change_id: str

    # Primary output
    category: str           # "Human-made" | "Natural" | "Uncertain"
    confidence_score: float # [0.0, 1.0]

    # Evidence
    evidence: list[str]
    evidence_count: int
    explanation: str
    verification_recommendation: str

    # Scoring breakdown (for transparency / explainability)
    human_score: float
    natural_score: float
    uncertain_score: float

    # Consistency diagnostics
    object_change_consistent: bool
    rule_triggered: str

    # Provenance
    classification_source: str = "rule_based_v1"

    # ------------------------------------------------------------------ #
    # Serialisation helpers
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Flat dict suitable for a CSV row or JSON record."""
        return {
            "change_id":                     self.change_id,
            "clf_category":                  self.category,
            "clf_confidence_score":          round(self.confidence_score, 4),
            "clf_evidence":                  "; ".join(self.evidence),
            "clf_evidence_count":            self.evidence_count,
            "clf_explanation":               self.explanation,
            "clf_verification_recommendation": self.verification_recommendation,
            "clf_human_score":               round(self.human_score, 4),
            "clf_natural_score":             round(self.natural_score, 4),
            "clf_uncertain_score":           round(self.uncertain_score, 4),
            "clf_object_change_consistent":  self.object_change_consistent,
            "clf_rule_triggered":            self.rule_triggered,
            "clf_classification_source":     self.classification_source,
        }

    def to_geojson_properties(self) -> dict:
        """Return only classification columns for embedding in a GeoJSON feature."""
        return {k: v for k, v in self.to_dict().items() if k != "change_id"}
