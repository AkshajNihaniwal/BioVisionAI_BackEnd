"""
Trajectory modeling for BIOVISION-AI.

- Stage classifier: single-snapshot -> stage + trend
- Sequence model: multiple timepoints -> future stage (placeholder/skeleton)

Clinical constraint: We estimate stage and likely evolution, not exact
reconstruction of past appearance or replacement of histopathology.
"""

from biovision_ai.models.trajectory.stage_classifier import StageClassifier
from biovision_ai.models.trajectory.sequence_model import LesionSequenceModel

__all__ = ["StageClassifier", "LesionSequenceModel"]
