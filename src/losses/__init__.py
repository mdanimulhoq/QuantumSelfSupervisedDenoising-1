"""
Losses package for N2LN-QEM.
"""
from .distribution import (
    kl_divergence,
    total_variation_distance,
    chi2_divergence,
    DistributionLoss,
    SNDLoss,
    HNEELoss,
)
from .physicality import (
    PhysicalityLoss,
    EntropyRegularization,
)
from .consistency import (
    ConsistencyLoss,
    CrossStageConsistency,
    JointConsistencyLoss,
)
