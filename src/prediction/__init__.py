"""
src/prediction/__init__.py
"""
from .evidential_head import EvidentialHead, opinion_to_feature_vector
from .baselines import SingleModalityBaseline, MonolithicFusionBaseline, LLMArbitratedBaseline

__all__ = [
    "EvidentialHead", "opinion_to_feature_vector",
    "SingleModalityBaseline", "MonolithicFusionBaseline", "LLMArbitratedBaseline",
]
