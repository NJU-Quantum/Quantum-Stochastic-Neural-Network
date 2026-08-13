import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from photonic_qsnn import (
    PhotonicQSNNClassifier,
    PhotonicQSNNConfig,
    encode_xy,
    label_bits,
    make_circle_dataset,
    set_seed,
)


class PhotonicQSNNTests(unittest.TestCase):
    def setUp(self):
        set_seed(11)
        self.config = PhotonicQSNNConfig(bins_per_axis=3, num_hidden=4, seed=11)

    def test_encoding_and_label_layout(self):
        xy = torch.tensor([[-1.0, 1.0], [0.5, -0.25]])
        encoded = encode_xy(xy, self.config)
        self.assertEqual(encoded.shape, (2, self.config.num_input))
        self.assertTrue(torch.all((encoded == 0) | (encoded == 1)))
        labels = label_bits(torch.tensor([0, 1]))
        self.assertTrue(torch.equal(labels, torch.tensor([[1.0, 0.0], [0.0, 1.0]])))

    def test_train_predict_save_and_export(self):
        xy, labels = make_circle_dataset(40, seed=11)
        classifier = PhotonicQSNNClassifier(self.config, sampler_kind="local")
        history = classifier.fit(
            xy,
            labels,
            epochs=2,
            batch_size=20,
            reads=1,
            gibbs_steps=2,
            verbose=False,
        )
        probabilities = classifier.predict_proba(xy[:4], reads=4, gibbs_steps=3)
        self.assertEqual(probabilities.shape, (4, 2))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(4)))
        self.assertTrue(torch.isfinite(probabilities).all())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            ising = root / "ising.npy"
            classifier.save(checkpoint, history)
            classifier.export_ising(ising)
            self.assertTrue(checkpoint.exists())
            self.assertEqual(
                np.load(ising).shape,
                (self.config.num_nodes + 1, self.config.num_nodes + 1),
            )


if __name__ == "__main__":
    unittest.main()
