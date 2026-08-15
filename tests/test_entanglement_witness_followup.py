import torch

from scripts.run_entanglement_witness_followup import runtime_config, update_ema


def test_ema_updates_floating_tensors_and_copies_integer_buffers():
    ema = {
        "weight": torch.tensor([1.0, 3.0]),
        "index": torch.tensor([1, 2], dtype=torch.long),
    }
    current = {
        "weight": torch.tensor([3.0, 7.0]),
        "index": torch.tensor([4, 5], dtype=torch.long),
    }
    update_ema(ema, current, decay=0.75)
    assert torch.allclose(ema["weight"], torch.tensor([1.5, 4.0]))
    assert torch.equal(ema["index"], current["index"])


def test_runtime_config_applies_candidate_without_mutating_base_config():
    base = {
        "generator": {"components": 16},
        "base_discriminators": {
            "qsnn": {"target_layer_mass": 0.9},
            "vqc": {"layers": 8},
        },
        "training": {
            "lr_decay_fraction": 0.3,
            "lr_decay_factor": 0.3,
            "lr_g": 0.002,
        },
        "target": {"dense_points": 101},
        "certification": {"tolerance": 5e-4},
    }
    candidate = {
        "name": "mass99",
        "target_layer_mass": 0.99,
        "lr_d": 0.001,
        "generator_steps": 10,
    }
    configured = runtime_config(base, "qsnn", candidate, epochs=600, target_p=0.36)
    assert configured["discriminators"]["qsnn"]["target_layer_mass"] == 0.99
    assert configured["training"]["lr_decay_start"] == 180
    assert configured["training"]["generator_steps"] == 10
    assert configured["target"]["werner_p"] == 0.36
    assert base["base_discriminators"]["qsnn"]["target_layer_mass"] == 0.9
