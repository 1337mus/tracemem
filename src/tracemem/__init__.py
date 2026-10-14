"""Local, evidence-grounded conversational memory primitives."""

from .schema import EvidenceRecord
from .storage import EvidenceStore

__all__ = ["EvidenceRecord", "EvidenceStore"]
