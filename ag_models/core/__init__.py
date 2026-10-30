"""
Core infrastructure for `ag_models`: Registry, BaseModel, ModelConfig, ModelOutput, and Layer Factories.
"""

from ag_models.core.base import BaseModel, ModelConfig, ModelOutput
from ag_models.core.layers import (
    get_activation,
    get_adaptive_avg_pool,
    get_adaptive_max_pool,
    get_avg_pool,
    get_conv,
    get_conv_transpose,
    get_interp_mode,
    get_max_pool,
    get_norm,
    interpolate_nd,
    make_divisible,
)
from ag_models.core.registry import (
    ATTENTION_REGISTRY,
    BLOCK_REGISTRY,
    HEAD_REGISTRY,
    LOSS_REGISTRY,
    MODEL_REGISTRY,
    Registry,
    build_attention,
    build_block,
    build_loss,
    build_model,
    list_models,
)

__all__ = [
    "BaseModel",
    "ModelConfig",
    "ModelOutput",
    "Registry",
    "MODEL_REGISTRY",
    "BLOCK_REGISTRY",
    "ATTENTION_REGISTRY",
    "HEAD_REGISTRY",
    "LOSS_REGISTRY",
    "build_model",
    "build_block",
    "build_attention",
    "build_loss",
    "list_models",
    "make_divisible",
    "get_conv",
    "get_conv_transpose",
    "get_norm",
    "get_activation",
    "get_adaptive_avg_pool",
    "get_adaptive_max_pool",
    "get_max_pool",
    "get_avg_pool",
    "get_interp_mode",
    "interpolate_nd",
]
