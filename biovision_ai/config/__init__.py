"""
Configuration management for BIOVISION-AI.

Loads YAML/JSON configs for model, training, API, and deployment.
"""

from biovision_ai.config.loader import load_config, get_default_config

__all__ = ["load_config", "get_default_config"]
