import sys
from pathlib import Path
# Add src directory to Python path so we can import my_project
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import pytest
from my_project.model import MyAwesomeModel
import torch

def test_dims():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = MyAwesomeModel().to(DEVICE)

    img = torch.load('data/raw/train_images_0.pt')[0].to(DEVICE)
    y_pred = model(img.unsqueeze(0).unsqueeze(0))
    assert y_pred.shape == torch.Size([1, 10])


def test_model_raises_on_bad_input_shape():
    model = MyAwesomeModel()

    with pytest.raises(ValueError, match="4D tensor"):
        model(torch.randn(28, 28))

    with pytest.raises(ValueError, match=r"\[1, 28, 28\]"):
        model(torch.randn(1, 2, 28, 28))



