"""QSNN-QGAN building blocks layered on top of the existing QSNN backend."""

from .autoencoder import (
    ProbabilityAutoencoder,
    load_autoencoder_artifact,
    save_autoencoder_artifact,
)
from .encoding import (
    area_downsample,
    embed_binary_label_density,
    pad_density_dimension,
    padding_mass,
    probability_amplitude_encode,
    probabilities_from_density,
)
from .generators import PQCGenerator
from .checkpoint import load_checkpoint, runtime_metadata, save_checkpoint
from .metrics import (
    density_fidelity,
    hellinger_distance,
    physicality_diagnostics,
    purity,
    trace_distance,
    total_variation_distance,
    trainable_parameter_count,
)
from .objectives import (
    direct_success_value,
    discriminator_loss,
    generator_loss,
    output_statistics,
    partition_output_statistics,
    trace_z_value,
)
from .qsnn_discriminator import QSNNDiscriminator
from .trainer import QGANTrainer
from .vqc_discriminator import VQCDiscriminator

__all__ = [
    "QSNNDiscriminator",
    "VQCDiscriminator",
    "PQCGenerator",
    "ProbabilityAutoencoder",
    "QGANTrainer",
    "area_downsample",
    "direct_success_value",
    "density_fidelity",
    "discriminator_loss",
    "embed_binary_label_density",
    "generator_loss",
    "hellinger_distance",
    "output_statistics",
    "partition_output_statistics",
    "pad_density_dimension",
    "padding_mass",
    "physicality_diagnostics",
    "probability_amplitude_encode",
    "probabilities_from_density",
    "purity",
    "load_checkpoint",
    "load_autoencoder_artifact",
    "runtime_metadata",
    "save_checkpoint",
    "save_autoencoder_artifact",
    "trace_distance",
    "trace_z_value",
    "total_variation_distance",
    "trainable_parameter_count",
]
