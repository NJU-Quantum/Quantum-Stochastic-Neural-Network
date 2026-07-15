import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class HighDimConfigTests(unittest.TestCase):
    def load(self, name):
        return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))

    def test_autoencoder_and_qgan_dimensions_match(self):
        for dim in (128, 256):
            autoencoder = self.load(f"autoencoder_mnist0_{dim}.yaml")
            self.assertEqual(autoencoder["model"]["latent_dim"], dim)
            for discriminator in ("qsnn", "vqc"):
                qgan = self.load(f"mnist0_ae{dim}_{discriminator}.yaml")
                self.assertEqual(qgan["model"]["input_dim"], dim)
                self.assertTrue(qgan["training"]["statevector_training"])

    def test_full_resolution_uses_explicit_lossless_embedding(self):
        for discriminator in ("qsnn", "vqc"):
            config = self.load(f"mnist0_full784_{discriminator}.yaml")
            self.assertEqual(config["data"]["representation"], "zero_padded_pixels")
            self.assertEqual(config["data"]["valid_feature_dim"], 784)
            self.assertEqual(config["data"]["encoding_eps"], 0.0)
            self.assertEqual(config["model"]["input_dim"], 1024)
            self.assertGreater(config["training"]["padding_mass_penalty"], 0.0)
            self.assertTrue(config["training"]["statevector_training"])


if __name__ == "__main__":
    unittest.main()
