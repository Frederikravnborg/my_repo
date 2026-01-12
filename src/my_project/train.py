import matplotlib.pyplot as plt
import torch
import typer
from pathlib import Path

# Allow running both as a script (`uv run src/my_project/train.py`) and as an importable module (`import my_project.train`).
try:
    from my_project.data import corrupt_mnist
    from my_project.model import MyAwesomeModel
except ModuleNotFoundError:  # pragma: no cover
    from data import corrupt_mnist  # type: ignore[import-not-found]
    from model import MyAwesomeModel  # type: ignore[import-not-found]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            # Validate input shape early to give a clearer error than a later PyTorch shape mismatch.
            # Expected: (batch, 1, 28, 28)
            if img.ndim != 4 or tuple(img.shape[1:]) != (1, 28, 28):
                raise ValueError(f"Expected img shape (batch, 1, 28, 28), got {tuple(img.shape)}")
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    # Save a CPU checkpoint so it can be loaded in environments without MPS/CUDA (e.g. Docker CPU images).
    Path("models").mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu")
    torch.save(model_cpu.state_dict(), "models/model.pth")
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")


if __name__ == "__main__":
    typer.run(train)