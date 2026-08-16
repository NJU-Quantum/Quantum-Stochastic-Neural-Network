"""Quantum-native state-discrimination tasks for QSNN experiments.

The NumPy reference backend remains importable in deliberately minimal
environments.  PyTorch exports are added when PyTorch is installed.
"""

from .numpy_reference import NumpyBinaryEnsemble, NumpyTrainConfig

__all__ = ["NumpyBinaryEnsemble", "NumpyTrainConfig"]

try:
    from .bounds import (
        HelstromResult,
        best_fixed_pauli_success,
        helstrom_measurement,
        measurement_success,
    )
    from .experiment import TrainConfig, evaluate_discriminator, train_discriminator
    from .models import (
        QubitHelstromQSNN,
        UnitaryQubitDiscriminator,
        effective_povm,
        povm_diagnostics,
    )
    from .states import (
        BinaryStateEnsemble,
        amplitude_damping,
        depolarize,
        make_nonorthogonal_qubit_ensemble,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.extend(
        [
            "BinaryStateEnsemble",
            "HelstromResult",
            "QubitHelstromQSNN",
            "TrainConfig",
            "UnitaryQubitDiscriminator",
            "amplitude_damping",
            "best_fixed_pauli_success",
            "depolarize",
            "effective_povm",
            "evaluate_discriminator",
            "helstrom_measurement",
            "make_nonorthogonal_qubit_ensemble",
            "measurement_success",
            "povm_diagnostics",
            "train_discriminator",
        ]
    )
