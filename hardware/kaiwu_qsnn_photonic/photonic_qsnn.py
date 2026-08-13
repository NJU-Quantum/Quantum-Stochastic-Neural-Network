from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_CHECKPOINT = DEFAULT_OUTPUT_DIR / "photonic_qsnn_checkpoint.pt"


@dataclass(frozen=True)
class PhotonicQSNNConfig:
    """Fixed topology for a QSNN-inspired photonic energy classifier."""

    bins_per_axis: int = 4
    num_hidden: int = 8
    output_penalty: float = 1.5
    seed: int = 7

    @property
    def num_input(self) -> int:
        # Thermometer bits for x/y plus sign(x*y) and radius bits.
        return 2 * self.bins_per_axis + 2

    @property
    def num_output(self) -> int:
        return 2

    @property
    def num_nodes(self) -> int:
        return self.num_input + self.num_output + self.num_hidden

    @property
    def out0(self) -> int:
        return self.num_input

    @property
    def out1(self) -> int:
        return self.num_input + 1


class LocalBoltzmannMachine(torch.nn.Module):
    """Small API-compatible fallback used when Kaiwu is not installed."""

    def __init__(self, num_nodes: int, device: torch.device) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.device = device
        self.quadratic_coef = torch.nn.Parameter(
            0.02 * torch.randn(num_nodes, num_nodes, device=device)
        )
        self.linear_bias = torch.nn.Parameter(torch.zeros(num_nodes, device=device))

    def symmetrized_quadratic_coef(self) -> torch.Tensor:
        upper = self.quadratic_coef.triu(1)
        return upper + upper.transpose(0, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        coupling = self.symmetrized_quadratic_coef()
        return -state @ self.linear_bias - 0.5 * torch.sum(
            (state @ coupling) * state, dim=-1
        )

    def gibbs_sample(
        self,
        num_steps: int,
        s_visible: torch.Tensor,
    ) -> torch.Tensor:
        """Sample unclamped nodes while preserving the visible prefix."""
        with torch.no_grad():
            batch = s_visible.shape[0]
            num_visible = s_visible.shape[1]
            state = torch.bernoulli(
                torch.full((batch, self.num_nodes), 0.5, device=self.device)
            )
            state[:, :num_visible] = s_visible
            coupling = self.symmetrized_quadratic_coef()

            for _ in range(num_steps):
                for node in torch.randperm(self.num_nodes, device=self.device):
                    if int(node) < num_visible:
                        continue
                    activation = state @ coupling[:, node] + self.linear_bias[node]
                    probability = torch.sigmoid(activation)
                    state[:, node] = torch.bernoulli(probability)
            return state

    def get_ising_matrix(self) -> np.ndarray:
        """Use the same binary-to-Ising conversion as Kaiwu's BM class."""
        with torch.no_grad():
            coupling = self.symmetrized_quadratic_coef()
            column_sums = coupling.sum(dim=0)
            matrix = torch.zeros(
                self.num_nodes + 1,
                self.num_nodes + 1,
                device=self.device,
            )
            matrix[:-1, :-1] = coupling / 8.0
            ising_bias = self.linear_bias / 4.0 + column_sums / 8.0
            matrix[:-1, -1] = ising_bias
            matrix[-1, :-1] = ising_bias
            return matrix.cpu().numpy()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_circle_dataset(
    samples: int,
    seed: int,
    noise: float = 0.07,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate an inner/outer ring binary classification dataset."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(samples) % 2
    angles = 2.0 * torch.pi * torch.rand(samples, generator=generator)
    base_radius = torch.where(labels == 0, 0.35, 0.80)
    radius = base_radius + noise * torch.randn(samples, generator=generator)
    xy = torch.stack((radius * torch.cos(angles), radius * torch.sin(angles)), dim=1)
    xy += 0.02 * torch.randn(samples, 2, generator=generator)
    permutation = torch.randperm(samples, generator=generator)
    return xy[permutation].float(), labels[permutation].long()


def encode_xy(xy: torch.Tensor, config: PhotonicQSNNConfig) -> torch.Tensor:
    """Map continuous samples to binary optical/Ising input spins."""
    xy = xy.clamp(-1.0, 1.0)
    thresholds = torch.linspace(
        -0.8,
        0.8,
        config.bins_per_axis,
        device=xy.device,
        dtype=xy.dtype,
    )
    x_bits = (xy[:, [0]] > thresholds).float()
    y_bits = (xy[:, [1]] > thresholds).float()
    cross = ((xy[:, 0] * xy[:, 1]) > 0.0).float().unsqueeze(1)
    radius = ((xy.square().sum(dim=1)) > 0.34).float().unsqueeze(1)
    return torch.cat((x_bits, y_bits, cross, radius), dim=1)


def label_bits(labels: torch.Tensor) -> torch.Tensor:
    """Encode class 0/1 as two mutually exclusive output nodes."""
    result = torch.zeros(labels.shape[0], 2, device=labels.device)
    result[torch.arange(labels.shape[0], device=labels.device), labels] = 1.0
    return result


def create_kaiwu_sampler(kind: str, task_name: str):
    """Create an official Kaiwu SDK sampler."""
    load_dotenv(SCRIPT_DIR.parent.parent / ".env", override=False)
    try:
        import kaiwu as kw
        from kaiwu.cim import CIMOptimizer, PrecisionReducer
        from kaiwu.classical import SimulatedAnnealingOptimizer
    except ImportError as exc:
        raise RuntimeError(
            "Kaiwu SDK/plugin is unavailable. Install the official dependencies "
            "described in README.md, or use --sampler local."
        ) from exc

    if kind == "kaiwu-sa":
        return SimulatedAnnealingOptimizer(alpha=0.999, size_limit=100)

    if kind == "kaiwu-cim":
        access_key = os.environ.get("WUYUE_ACCESS_KEY_ID", "").strip()
        secret_key = os.environ.get("WUYUE_ACCESS_KEY_SECRET", "").strip()
        device_id = os.environ.get(
            "WUYUE_PHOTONIC_DEVICE_ID", "WuYue-QPU-Qboson-1000"
        ).strip()
        if not access_key or not secret_key:
            raise RuntimeError("WuYue AK/SK are required for Kaiwu CIM submissions.")
        kw.common.CheckpointManager.save_dir = str(DEFAULT_OUTPUT_DIR / "kaiwu_tasks")
        cim = CIMOptimizer(
            task_name=task_name,
            wait=True,
            access_key=access_key,
            secret_key=secret_key,
            device_id=device_id,
        )
        return PrecisionReducer(
            cim,
            precision=8,
            truncated_precision=10,
            target_bits=1000,
            only_feasible_solution=False,
        )

    raise ValueError(f"Unsupported Kaiwu sampler: {kind}")


class PhotonicQSNNClassifier:
    """Joint input/label energy model with two QSNN-style output attractors."""

    def __init__(
        self,
        config: PhotonicQSNNConfig,
        sampler_kind: str = "local",
        device: str = "cpu",
        task_name: str = "QSNNPhotonicBinary",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.sampler_kind = sampler_kind
        self.external_sampler = None

        if sampler_kind == "local":
            self.model = LocalBoltzmannMachine(config.num_nodes, self.device)
        else:
            try:
                from kaiwu.torch_plugin import BoltzmannMachine
            except ImportError as exc:
                raise RuntimeError(
                    "kaiwu-torch-plugin is required for Kaiwu samplers."
                ) from exc
            self.model = BoltzmannMachine(config.num_nodes, device=self.device)
            self.external_sampler = create_kaiwu_sampler(sampler_kind, task_name)

        self.model.to(self.device)
        self._initialize_output_constraint()

    def _initialize_output_constraint(self) -> None:
        """Bias the output pair toward a one-hot real/fake state."""
        out0, out1 = self.config.out0, self.config.out1
        penalty = self.config.output_penalty
        with torch.no_grad():
            self.model.linear_bias[out0] = penalty
            self.model.linear_bias[out1] = penalty
            self.model.quadratic_coef[out0, out1] = -2.0 * penalty

    def _local_conditioned(
        self,
        visible: torch.Tensor,
        reads: int,
        gibbs_steps: int,
    ) -> torch.Tensor:
        expanded = visible.repeat_interleave(reads, dim=0)
        return self.model.gibbs_sample(num_steps=gibbs_steps, s_visible=expanded)

    def _kaiwu_conditioned(self, visible: torch.Tensor, reads: int) -> torch.Tensor:
        samples: List[torch.Tensor] = []
        for row in visible:
            states = self.model.condition_sample(self.external_sampler, row.unsqueeze(0))
            if states.shape[0] >= reads:
                states = states[:reads]
            else:
                repeats = (reads + states.shape[0] - 1) // states.shape[0]
                states = states.repeat(repeats, 1)[:reads]
            samples.append(states)
        return torch.cat(samples, dim=0)

    def conditioned_samples(
        self,
        visible: torch.Tensor,
        reads: int,
        gibbs_steps: int,
    ) -> torch.Tensor:
        visible = visible.to(self.device)
        if self.sampler_kind == "local":
            return self._local_conditioned(visible, reads, gibbs_steps)
        return self._kaiwu_conditioned(visible, reads)

    def fit(
        self,
        xy: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 40,
        batch_size: int = 32,
        learning_rate: float = 0.04,
        reads: int = 1,
        gibbs_steps: int = 20,
        weight_decay: float = 1e-3,
        verbose: bool = True,
    ) -> List[float]:
        """Conditional contrastive training with clamped input/label phases."""
        encoded = encode_xy(xy.to(self.device), self.config)
        labels = labels.to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        history: List[float] = []

        for epoch in range(epochs):
            permutation = torch.randperm(encoded.shape[0], device=self.device)
            epoch_loss = 0.0
            batches = 0

            for start in range(0, encoded.shape[0], batch_size):
                index = permutation[start : start + batch_size]
                batch_x = encoded[index]
                batch_y = labels[index]

                positive_visible = torch.cat((batch_x, label_bits(batch_y)), dim=1)
                positive = self.conditioned_samples(
                    positive_visible, reads=reads, gibbs_steps=gibbs_steps
                )
                negative = self.conditioned_samples(
                    batch_x, reads=reads, gibbs_steps=gibbs_steps
                )

                optimizer.zero_grad()
                contrastive = self.model(positive).mean() - self.model(negative).mean()
                regularization = weight_decay * (
                    self.model.quadratic_coef.square().mean()
                    + self.model.linear_bias.square().mean()
                )
                loss = contrastive + regularization
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    self.model.quadratic_coef.data.clamp_(-2.0, 2.0)
                    self.model.linear_bias.data.clamp_(-2.0, 2.0)

                epoch_loss += float(loss.detach())
                batches += 1

            mean_loss = epoch_loss / max(batches, 1)
            history.append(mean_loss)
            if verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs):
                print(f"epoch={epoch + 1:03d} objective={mean_loss:.6f}")

        return history

    @torch.no_grad()
    def predict_proba(
        self,
        xy: torch.Tensor,
        reads: int = 32,
        gibbs_steps: int = 40,
    ) -> torch.Tensor:
        encoded = encode_xy(xy.to(self.device), self.config)
        states = self.conditioned_samples(encoded, reads=reads, gibbs_steps=gibbs_steps)
        outputs = states[:, self.config.out0 : self.config.out1 + 1]
        outputs = outputs.view(encoded.shape[0], reads, 2)

        vote0 = (outputs[:, :, 0] > outputs[:, :, 1]).float()
        vote1 = (outputs[:, :, 1] > outputs[:, :, 0]).float()
        ties = (outputs[:, :, 0] == outputs[:, :, 1]).float()
        p0 = (vote0 + 0.5 * ties).mean(dim=1)
        p1 = (vote1 + 0.5 * ties).mean(dim=1)
        return torch.stack((p0, p1), dim=1)

    def save(self, path: Path, history: Optional[List[float]] = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "state_dict": self.model.state_dict(),
                "history": history or [],
            },
            path,
        )

    def load(self, path: Path) -> Dict[str, Any]:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["state_dict"])
        return payload

    def export_ising(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.model.get_ising_matrix())
        return path


def accuracy(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = probabilities.argmax(dim=1).cpu()
    return float((predictions == labels.cpu()).float().mean())


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def train_demo(args: argparse.Namespace) -> None:
    config = PhotonicQSNNConfig(
        bins_per_axis=args.bins_per_axis,
        num_hidden=args.hidden,
        seed=args.seed,
    )
    set_seed(config.seed)
    xy, labels = make_circle_dataset(args.samples, args.seed)
    split = int(0.8 * args.samples)
    train_xy, test_xy = xy[:split], xy[split:]
    train_y, test_y = labels[:split], labels[split:]

    classifier = PhotonicQSNNClassifier(
        config,
        sampler_kind=args.sampler,
        device=args.device,
        task_name=args.task_name,
    )
    history = classifier.fit(
        train_xy,
        train_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        reads=args.train_reads,
        gibbs_steps=args.gibbs_steps,
    )
    probabilities = classifier.predict_proba(
        test_xy,
        reads=args.eval_reads,
        gibbs_steps=args.gibbs_steps,
    )
    test_accuracy = accuracy(probabilities, test_y)

    classifier.save(args.checkpoint, history)
    ising_path = classifier.export_ising(args.output_dir / "photonic_qsnn_ising.npy")
    summary = {
        "platform": "Kaiwu coherent photonic Ising/Boltzmann sampler",
        "sampler": args.sampler,
        "config": asdict(config),
        "train_samples": int(train_xy.shape[0]),
        "test_samples": int(test_xy.shape[0]),
        "test_accuracy": test_accuracy,
        "final_objective": history[-1],
        "checkpoint": str(args.checkpoint),
        "ising_matrix": str(ising_path),
    }
    write_json(args.output_dir / "photonic_qsnn_result_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def predict_one(args: argparse.Namespace) -> None:
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = PhotonicQSNNConfig(**payload["config"])
    classifier = PhotonicQSNNClassifier(
        config,
        sampler_kind=args.sampler,
        device=args.device,
        task_name=args.task_name,
    )
    classifier.model.load_state_dict(payload["state_dict"])
    xy = torch.tensor([[args.x, args.y]], dtype=torch.float32)
    probabilities = classifier.predict_proba(
        xy,
        reads=args.eval_reads,
        gibbs_steps=args.gibbs_steps,
    )[0]
    result = {
        "x": args.x,
        "y": args.y,
        "p0": float(probabilities[0]),
        "p1": float(probabilities[1]),
        "pred": int(probabilities.argmax()),
    }
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="QSNN-inspired binary classifier for Kaiwu coherent photonic samplers."
    )
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument(
        "--sampler",
        choices=["local", "kaiwu-sa", "kaiwu-cim"],
        default="local",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task-name", default="QSNNPhotonicBinary")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples", type=int, default=320)
    parser.add_argument("--bins-per-axis", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--train-reads", type=int, default=1)
    parser.add_argument("--eval-reads", type=int, default=32)
    parser.add_argument("--gibbs-steps", type=int, default=20)
    parser.add_argument("--x", type=float, default=0.2)
    parser.add_argument("--y", type=float, default=-0.4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "train":
        train_demo(args)
    else:
        predict_one(args)


if __name__ == "__main__":
    main()
