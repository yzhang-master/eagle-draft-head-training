"""EAGLE-style draft-head training and speculative decoding."""

from eagle_draft.draft import EAGLEDraftHead
from eagle_draft.model import TinyTargetModel
from eagle_draft.speculative import speculative_decode, target_decode
from eagle_draft.train import train_draft, train_target

__all__ = [
    "EAGLEDraftHead",
    "TinyTargetModel",
    "speculative_decode",
    "target_decode",
    "train_draft",
    "train_target",
]
__version__ = "0.1.0"
