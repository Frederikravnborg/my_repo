from __future__ import annotations

from pathlib import Path

import torch
import typer
from torch.utils.data import Dataset


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


def _find_train_shards(raw_dir: Path) -> list[int]:
    image_idxs: set[int] = set()
    target_idxs: set[int] = set()

    for p in raw_dir.glob("train_images_*.pt"):
        try:
            image_idxs.add(int(p.stem.split("_")[-1]))
        except ValueError:
            continue

    for p in raw_dir.glob("train_target_*.pt"):
        try:
            target_idxs.add(int(p.stem.split("_")[-1]))
        except ValueError:
            continue

    missing_images = sorted(target_idxs - image_idxs)
    missing_targets = sorted(image_idxs - target_idxs)
    if missing_images or missing_targets:
        raise FileNotFoundError(
            "Mismatched shard files in raw_dir. "
            f"Missing train_images shards: {missing_images}. "
            f"Missing train_target shards: {missing_targets}."
        )

    idxs = sorted(image_idxs & target_idxs)
    if not idxs:
        raise FileNotFoundError(f"No train shards found in {raw_dir}. Expected train_images_*.pt + train_target_*.pt")
    return idxs


def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    """Process raw data (sharded train + test) and save it to processed directory."""
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    shard_idxs = _find_train_shards(raw_path)

    train_images, train_target = [], []
    for i in shard_idxs:
        train_images.append(torch.load(raw_path / f"train_images_{i}.pt"))
        train_target.append(torch.load(raw_path / f"train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(raw_path / "test_images.pt")
    test_target: torch.Tensor = torch.load(raw_path / "test_target.pt")

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_images = normalize(train_images)
    test_images = normalize(test_images)

    torch.save(train_images, processed_path / "train_images.pt")
    torch.save(train_target, processed_path / "train_target.pt")
    torch.save(test_images, processed_path / "test_images.pt")
    torch.save(test_target, processed_path / "test_target.pt")


def corrupt_mnist(processed_dir: str = "data/processed") -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST from the processed directory."""
    processed_path = Path(processed_dir)
    train_images = torch.load(processed_path / "train_images.pt")
    train_target = torch.load(processed_path / "train_target.pt")
    test_images = torch.load(processed_path / "test_images.pt")
    test_target = torch.load(processed_path / "test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


class MyDataset(Dataset):
    """A simple Dataset wrapper around the raw sharded training data."""

    def __init__(self, raw_dir: str):
        raw_path = Path(raw_dir)
        shard_idxs = _find_train_shards(raw_path)

        images, targets = [], []
        for i in shard_idxs:
            images.append(torch.load(raw_path / f"train_images_{i}.pt"))
            targets.append(torch.load(raw_path / f"train_target_{i}.pt"))

        self.images = torch.cat(images).unsqueeze(1).float()
        self.targets = torch.cat(targets).long()

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.targets[idx]


if __name__ == "__main__":
    typer.run(preprocess_data)