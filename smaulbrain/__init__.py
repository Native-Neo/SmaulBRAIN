"""SmaulBRAIN: Byte-level Recurrent Adaptive Intelligent Network.

Recurrent byte-level LM with linear attention, dynamic MoE experts,
adaptive halting, FP8/BF16/FP32 precision policy, and D2R/R2VR/D2VR paging.
"""

from .config import SmaulBrainConfig

__all__ = ["SmaulBrainConfig"]
__version__ = "0.1.0"
