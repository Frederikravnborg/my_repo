from __future__ import annotations

from pathlib import Path

import torch


def test_train_creates_artifacts(monkeypatch, tmp_path: Path) -> None:
    # Import inside the test so pytest's conftest has already set MPLBACKEND and sys.path.
    import my_project.train as train_mod

    # Avoid writing into the repo; train.py writes relative paths.
    monkeypatch.chdir(tmp_path)

    # Make sure report directory exists (train.py currently doesn't mkdir it).
    (tmp_path / "reports" / "figures").mkdir(parents=True, exist_ok=True)

    # Tiny fake dataset: 8 samples of MNIST-shaped images + labels.
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    tiny_ds = torch.utils.data.TensorDataset(x, y)

    def fake_corrupt_mnist(*args, **kwargs):
        return tiny_ds, tiny_ds

    monkeypatch.setattr(train_mod, "corrupt_mnist", fake_corrupt_mnist)

    # 1 epoch, small batch => should be fast.
    train_mod.train(epochs=1, batch_size=4, lr=1e-3)

    assert (tmp_path / "models" / "model.pth").is_file()
    assert (tmp_path / "reports" / "figures" / "training_statistics.png").is_file()


