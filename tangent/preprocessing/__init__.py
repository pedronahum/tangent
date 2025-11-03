"""Preprocessing module for Tangent checkpointing.

This module provides AST preprocessing to transform loops into checkpoint-ready form.
"""

from tangent.preprocessing.checkpoint_preprocessor import CheckpointPreprocessor

__all__ = ['CheckpointPreprocessor']
